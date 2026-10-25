"""DSpark 草稿模型（vendored 自 deepseek-ai/DeepSpec，MIT License）。

来源与版本（检索日 2026-10-25，main 分支）：

```
deepspec/modeling/dspark/qwen3/modeling.py    → src/spec/dspark/modeling.py
deepspec/modeling/dspark/qwen3/config.py      → src/spec/dspark/config.py
deepspec/modeling/dspark/common.py            → src/spec/dspark/common.py
deepspec/modeling/dspark/markov_head.py       → src/spec/dspark/markov_head.py
deepspec/utils/sampling.py（仅 sample_tokens） → src/spec/dspark/_compat.py
deepspec/utils/metrics.py（仅 add_metric，训练期用） → src/spec/dspark/_compat.py
```

**本地改动只限三处**，算法与结构未改：

1. `deepspec.*` 的 import 换成相对导入（本项目不引入 deepspec 训练框架，它依赖
   38 TB 级 target cache 与 8 卡训练脚本）；
2. `torch.nn.attention.flex_attention.create_block_mask` 改为**函数内惰性导入**
   ——它只在训练/教师强制路径用到，推理不碰；放在模块顶层会让推理环境因为一个
   训练依赖而整个 import 失败；
3. 每个文件加 provenance 头（便于审计与追上游更新）。

为什么 vendor 而不是 `pip install deepspec`：上游是训练框架，会把 datasets /
accelerate / tensorboard 等一并拉进来，且其 `deepspec.utils` 有副作用式初始化；
本引擎只需要那 4 个建模文件（共 ~40 KB）。

使用方式见 `src/spec/dspark/loader.py`（Phase 3 实现）：
`Qwen3DSparkModel.from_pretrained(checkpoint_dir)` 直接加载官方
`dspark_qwen3_8b_block7`，无需 transformers 5.x（`Qwen3DSparkModel` 是这里的类，
config 里的 `transformers_version` 只是保存时版本，不参与校验）。
"""

from __future__ import annotations

from .common import (
    DSparkForwardOutput,
    extract_context_feature,
    validate_target_layer_ids,
)
from .modeling import Qwen3DSparkModel

__all__ = [
    "DSparkForwardOutput",
    "Qwen3DSparkModel",
    "extract_context_feature",
    "validate_target_layer_ids",
]
