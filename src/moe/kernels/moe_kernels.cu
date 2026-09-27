#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace vllm_moe {

// ---------------------------------------------------------------------------
// Align
// ---------------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ int get_local_expert_id(size_t idx,
                                                  const scalar_t* __restrict__ topk_ids,
                                                  int32_t num_experts) {
  int expert_id = static_cast<int>(topk_ids[idx]);
  if (expert_id >= num_experts || expert_id < 0) return -1;
  return expert_id;
}

template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t padded_num_experts, int32_t experts_per_warp,
    int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded, int32_t max_num_m_blocks) {
  extern __shared__ int32_t shared_counts[];

  if (blockIdx.x % 2) {
    for (size_t it = threadIdx.x; it < static_cast<size_t>(max_num_tokens_padded);
         it += blockDim.x) {
      sorted_token_ids[it] = static_cast<int32_t>(numel);
    }
    return;
  }

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }
  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = get_local_expert_id(i, topk_ids, num_experts);
    if (expert_id != -1) {
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }
  }
  __syncthreads();

  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[expert_id] = cumsum_val;
  }
  if (expert_id == num_experts) {
    total_tokens_post_pad[0] = cumsum_val;
  }
  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
  const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start_idx; i < static_cast<size_t>(max_num_m_blocks);
       i += blockDim.x) {
    expert_ids[i] = -1;
  }
}

template <typename scalar_t, int32_t fill_threads>
__device__ void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t block_size, size_t numel,
    int32_t max_num_tokens_padded, int32_t max_num_m_blocks) {
  if (threadIdx.x < fill_threads) {
    for (size_t it = threadIdx.x; it < static_cast<size_t>(max_num_tokens_padded);
         it += fill_threads) {
      sorted_token_ids[it] = static_cast<int32_t>(numel);
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = shared_mem + num_experts + 1;

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = get_local_expert_id(i, topk_ids, num_experts);
    if (expert_id != -1) {
      tokens_cnts[(tid + 1) * num_experts + expert_id] += 1;
    }
  }
  __syncthreads();

  if (tid < static_cast<size_t>(num_experts)) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= static_cast<int>(stride); ++i) {
      tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
    }
  }
  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
    }
    total_tokens_post_pad[0] = cumsum[num_experts];
  }
  __syncthreads();

  if (tid < static_cast<size_t>(num_experts)) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = static_cast<int32_t>(tid);
    }
  }
  const size_t fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (size_t i = fill_start_idx; i < static_cast<size_t>(max_num_m_blocks);
       i += stride) {
    expert_ids[i] = -1;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = get_local_expert_id(i, topk_ids, num_experts);
    if (expert_id != -1) {
      int32_t rank_post_pad =
          tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
      sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}

template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = get_local_expert_id(i, topk_ids, num_experts);
    if (expert_id != -1) {
      int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
      sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
    }
  }
  (void)max_num_tokens_padded;
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t padded_num_experts, int32_t experts_per_warp,
    int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded) {
  _moe_align_block_size(topk_ids, sorted_token_ids, expert_ids,
                        total_tokens_post_pad, num_experts, padded_num_experts,
                        experts_per_warp, block_size, numel, cumsum,
                        max_num_tokens_padded,
                        CEILDIV(max_num_tokens_padded, block_size));
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded) {
  _count_and_sort_expert_tokens(topk_ids, sorted_token_ids, cumsum_buffer, numel,
                                num_experts, max_num_tokens_padded);
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t block_size, size_t numel,
    int32_t max_num_tokens_padded) {
  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, num_experts,
      block_size, numel, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size));
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be CUDA");
  TORCH_CHECK(sorted_token_ids.dtype() == torch::kInt32);
  TORCH_CHECK(experts_ids.dtype() == torch::kInt32);
  TORCH_CHECK(num_tokens_post_pad.dtype() == torch::kInt32);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  TORCH_CHECK(padded_num_experts < 1024, "padded_num_experts must be < 1024");

  AT_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size", [&] {
    const bool small_batch =
        (topk_ids.numel() < 1024) && (num_experts <= 64);
    if (small_batch) {
      const int32_t threads = std::max<int32_t>(static_cast<int32_t>(num_experts),
                                                WARP_SIZE);
      const int32_t shared_mem_size =
          ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
      constexpr int32_t fill_threads = 256;
      auto kernel =
          moe_align_block_size_small_batch_expert_kernel<scalar_t, fill_threads>;
      kernel<<<1, fill_threads + threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(), experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          static_cast<int32_t>(num_experts), static_cast<int32_t>(block_size),
          static_cast<size_t>(topk_ids.numel()),
          static_cast<int32_t>(sorted_token_ids.size(0)));
    } else {
      auto cumsum_buffer =
          torch::empty({num_experts + 1}, topk_ids.options().dtype(torch::kInt32));
      int threads = 1024;
      threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
      size_t num_warps = CEILDIV(padded_num_experts, WARP_SIZE);
      size_t shared_mem_size = num_warps * WARP_SIZE * sizeof(int32_t);
      auto align_kernel = moe_align_block_size_kernel<scalar_t>;
      align_kernel<<<2, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(), experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          static_cast<int32_t>(num_experts),
          static_cast<int32_t>(padded_num_experts), WARP_SIZE,
          static_cast<int32_t>(block_size),
          static_cast<size_t>(topk_ids.numel()),
          cumsum_buffer.data_ptr<int32_t>(),
          static_cast<int32_t>(sorted_token_ids.size(0)));

      const int block_threads = std::min(256, threads);
      const int num_blocks =
          (topk_ids.numel() + block_threads - 1) / block_threads;
      const int actual_blocks = std::min(num_blocks, 65535);
      dim3 gridDims(1, actual_blocks);
      count_and_sort_expert_tokens_kernel<scalar_t>
          <<<gridDims, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(),
              static_cast<size_t>(topk_ids.numel()),
              static_cast<int32_t>(num_experts),
              static_cast<int32_t>(sorted_token_ids.size(0)));
    }
  });
}

// ---------------------------------------------------------------------------
// moe_sum: input [M, topk, H] -> output [M, H]
// ---------------------------------------------------------------------------

template <typename scalar_t>
constexpr int MOE_SUM_VEC = 16 / sizeof(scalar_t);

template <typename scalar_t, int TOPK>
__global__ void moe_sum_vec_kernel(scalar_t* __restrict__ out,
                                   const scalar_t* __restrict__ input,
                                   int64_t num_tokens, int d,
                                   int64_t stride_token, int64_t stride_topk) {
  constexpr int VEC = MOE_SUM_VEC<scalar_t>;
  const int64_t n_vec = d / VEC;
  const int64_t total = num_tokens * n_vec;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t token = i / n_vec;
    const int64_t v = i % n_vec;
    const scalar_t* in_tok = input + token * stride_token + v * VEC;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const scalar_t* src = in_tok + k * stride_topk;
#pragma unroll
      for (int j = 0; j < VEC; ++j) acc[j] += static_cast<float>(src[j]);
    }
    scalar_t* dst = out + token * d + v * VEC;
#pragma unroll
    for (int j = 0; j < VEC; ++j) dst[j] = static_cast<scalar_t>(acc[j]);
  }
}

template <typename scalar_t>
__global__ void moe_sum_vec_dynamic_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    int64_t num_tokens, int d, int topk, int64_t stride_token,
    int64_t stride_topk) {
  constexpr int VEC = MOE_SUM_VEC<scalar_t>;
  const int64_t n_vec = d / VEC;
  const int64_t total = num_tokens * n_vec;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t token = i / n_vec;
    const int64_t v = i % n_vec;
    const scalar_t* in_tok = input + token * stride_token + v * VEC;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) acc[j] = 0.f;
    for (int k = 0; k < topk; ++k) {
      const scalar_t* src = in_tok + k * stride_topk;
#pragma unroll
      for (int j = 0; j < VEC; ++j) acc[j] += static_cast<float>(src[j]);
    }
    scalar_t* dst = out + token * d + v * VEC;
#pragma unroll
    for (int j = 0; j < VEC; ++j) dst[j] = static_cast<scalar_t>(acc[j]);
  }
}

template <typename scalar_t>
__global__ void moe_sum_scalar_kernel(scalar_t* __restrict__ out,
                                      const scalar_t* __restrict__ input, int d,
                                      int topk, int64_t stride_token,
                                      int64_t stride_topk,
                                      int64_t stride_hidden) {
  const int64_t token_idx = blockIdx.x;
  const scalar_t* in_tok = input + token_idx * stride_token;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = 0.f;
    for (int k = 0; k < topk; ++k) {
      x += static_cast<float>(in_tok[k * stride_topk + idx * stride_hidden]);
    }
    out[token_idx * d + idx] = static_cast<scalar_t>(x);
  }
}

void moe_sum(torch::Tensor input, torch::Tensor output) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda());
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(input.dim() == 3 && output.dim() == 2);
  TORCH_CHECK(input.size(0) == output.size(0));
  TORCH_CHECK(input.size(2) == output.size(1));

  const int hidden_size = static_cast<int>(input.size(2));
  const int64_t num_tokens = output.size(0);
  const int topk = static_cast<int>(input.size(1));
  const int64_t stride_token = input.stride(0);
  const int64_t stride_topk = input.stride(1);
  const int64_t stride_hidden = input.stride(2);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "moe_sum", [&] {
        auto* out_ptr = output.data_ptr<scalar_t>();
        auto* in_ptr = input.data_ptr<scalar_t>();
        constexpr int VEC = MOE_SUM_VEC<scalar_t>;
        constexpr int WIDTH = VEC * sizeof(scalar_t);
        const bool can_vec =
            (stride_hidden == 1) && (hidden_size % VEC == 0) &&
            (stride_token % VEC == 0) && (stride_topk % VEC == 0) &&
            (reinterpret_cast<uintptr_t>(in_ptr) % WIDTH == 0) &&
            (reinterpret_cast<uintptr_t>(out_ptr) % WIDTH == 0);
        if (can_vec) {
          const int64_t n_vec = hidden_size / VEC;
          const int64_t total = num_tokens * n_vec;
          const int block = 256;
          const dim3 grid(std::min<int64_t>((total + block - 1) / block, 65535));
          switch (topk) {
            case 1:
              moe_sum_vec_kernel<scalar_t, 1><<<grid, block, 0, stream>>>(
                  out_ptr, in_ptr, num_tokens, hidden_size, stride_token,
                  stride_topk);
              break;
            case 2:
              moe_sum_vec_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
                  out_ptr, in_ptr, num_tokens, hidden_size, stride_token,
                  stride_topk);
              break;
            case 4:
              moe_sum_vec_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
                  out_ptr, in_ptr, num_tokens, hidden_size, stride_token,
                  stride_topk);
              break;
            case 8:
              moe_sum_vec_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(
                  out_ptr, in_ptr, num_tokens, hidden_size, stride_token,
                  stride_topk);
              break;
            default:
              moe_sum_vec_dynamic_kernel<scalar_t><<<grid, block, 0, stream>>>(
                  out_ptr, in_ptr, num_tokens, hidden_size, topk, stride_token,
                  stride_topk);
              break;
          }
        } else {
          dim3 grid(num_tokens);
          dim3 block(std::min(hidden_size, 1024));
          moe_sum_scalar_kernel<scalar_t><<<grid, block, 0, stream>>>(
              out_ptr, in_ptr, hidden_size, topk, stride_token, stride_topk,
              stride_hidden);
        }
      });
}

// ---------------------------------------------------------------------------
// silu_and_mul / mul_and_silu
//   silu_and_mul:  out = silu(x[..., :d]) * x[..., d:]     // [gate|up]
//   mul_and_silu:  out = x[..., :d] * silu(x[..., d:])     // [up|gate]
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float silu_f(float x) {
  return x / (1.f + ::expf(-x));
}

template <typename scalar_t, bool act_first>
__global__ void act_and_mul_kernel(scalar_t* __restrict__ out,
                                   const scalar_t* __restrict__ input, int d) {
  const int64_t token = blockIdx.x;
  const scalar_t* x_ptr = input + token * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float a = static_cast<float>(x_ptr[i]);
    float b = static_cast<float>(y_ptr[i]);
    float r;
    if constexpr (act_first) {
      r = silu_f<scalar_t>(a) * b;
    } else {
      r = a * silu_f<scalar_t>(b);
    }
    out_ptr[i] = static_cast<scalar_t>(r);
  }
}

void launch_act_and_mul(torch::Tensor& out, torch::Tensor& input, bool act_first) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda());
  TORCH_CHECK(input.size(-1) % 2 == 0);
  const int d = static_cast<int>(input.size(-1) / 2);
  TORCH_CHECK(out.size(-1) == d);
  const int64_t num_tokens = input.numel() / input.size(-1);
  TORCH_CHECK(out.numel() / d == num_tokens);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "act_and_mul", [&] {
        if (act_first) {
          act_and_mul_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
        } else {
          act_and_mul_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
        }
      });
}

void silu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  launch_act_and_mul(out, input, /*act_first=*/true);
}

void mul_and_silu(torch::Tensor& out, torch::Tensor& input) {
  launch_act_and_mul(out, input, /*act_first=*/false);
}

}  // namespace vllm_moe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_block_size", &vllm_moe::moe_align_block_size,
        "MoE align block size (CUDA)");
  m.def("moe_sum", &vllm_moe::moe_sum, "MoE sum over topk (CUDA)");
  m.def("silu_and_mul", &vllm_moe::silu_and_mul, "silu(gate)*up for [gate|up]");
  m.def("mul_and_silu", &vllm_moe::mul_and_silu, "up*silu(gate) for [up|gate]");
}
