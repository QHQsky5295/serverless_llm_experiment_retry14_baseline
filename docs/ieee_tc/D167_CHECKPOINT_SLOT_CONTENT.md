# D167：从原始 checkpoint 独立核对 GPU 槽位内容

## 问题、复用范围与实现依据

D166 已完成真实 GPU 槽位与已注册 CPU 张量的精确比较，但参考张量来自被测
加载器本身。本项从**原始 safetensors 与 adapter/base config 独立构造期望值**，
与 D166 保存的注册张量摘要、实际 GPU 读回摘要比较。不再次运行 GPU，不修改
原始结果、LoRA 工件、trace、服务算法或 IEEE 九个公式。

复用原 22 请求、12 个逻辑 adapter、4 个权重/config 内容类别和 44 个前后快照。
内容类别归并以实际文件 SHA 为依据，不把逻辑 ID 当成独立训练权重。全部源路径
与 SHA 记录于 curated JSON；D166 的两次失败仍由原始报告保留，不在本项删除。

核查 vLLM 0.30.0 的原始实现：

- [LoRA 加载器](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)：加载到目标 dtype。
- [LoRA 权重变换](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_weights.py)：将缩放并入 B。
- [PEFT 参数](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/peft_helper.py)：当前非 rsLoRA 的缩放为 alpha/r。
- [Llama 层映射](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/models/llama.py)：Q/K/V 和 gate/up 的原生排列。

本地已安装文件的 SHA 另存，网页版本描述不代替本地身份核验。没有调用原生
加载器生成参考，而在既有 `ieee_tc_preflight.py` 中小幅扩展 NumPy 离线检查：
按 base config 推导层数与形状、读单个原有张量、先 FP16 转换、再对 B 缩放、
按原生布局补零。每个记录要求形状、来源有无、有限性、非零计数及两个 SHA
全部吻合。缺失、重复或额外张量直接拒绝；当前只支持已核查的 dense TP=1、
非 GQA Llama、q/k/v/o、无 rsLoRA/DoRA 的 FP16 合同，不泛化到未知加载方式。

## 结果表

| 检查 | 结果 | 解释 |
|---|---:|---|
| 单元与既有工件审核测试 | 14/14 通过，其中 7 项新增 | 包括错位、错 SHA、形状错误、缺失/重复及非法配置反例 |
| 复用原始请求 | 22 | 没有新增 GPU 回放 |
| 生成前后快照 | 44 | 复用 D166 的同步读回，不是独立重复 |
| checkpoint 推导内容与槽张量比较 | 19,888 | 每快照 452 个，覆盖 padding、缺失模块、embedding/head |
| 不一致张量 | 0 | 同时匹配注册参考摘要和实际 GPU 内容摘要 |
| medical 错配到 finance 的反例 | 256 个张量不符，拒绝 | 非零 adapter 身份可区分 |
| 零权重错配到 finance 的反例 | 256 个张量不符，拒绝 | 不能以 base/零更新混充 finance |
| 省略 B 缩放的反例 | 128 个张量不符，拒绝 | 检查能够识别实际变换错误 |
| 离线分析耗时 / 最大 RSS | 18.84 s / 1,180,204 KiB | 不属于服务性能或 GPU 生命周期结果 |
| 分析 high/max/OOM/swap | 0 | 使用独立 3/4 GiB、swap=0、固定辅助 CPU 域 |

本项新增测试没有新建实验权重池；仅用临时微型数组测试检查器逻辑。实际分析
只读取现有文件，所有工件 SHA 与 D166 输入身份吻合，147 项历史保护检查通过。
按计划 §11 与 academic-plotting，此项交付正确性表格和 CSV，不画性能排名。

## 支持与不支持的结论

支持：这组既有工件经当前加载/放置链路后，D166 生成前后实际 GPU 槽内容与
从原始 checkpoint 独立推导的内容完全一致。相较 D166 单独使用已注册张量，
本项补齐了这组快照的外部内容来源证据。

不支持：每个 token 期间的实际执行映射与内核算术、全部 500 ID 的语义资格、
Full 回放的数值正确请求数、共同 warm-SLO、G1/G2 或性能领先。仍保留
`n_correct=null`；不追改 D164/D166 的资格标记，不把快照数作为性能重复数。
零权重逻辑 ID 之间不能通过权重内容相互区分，仍需独立请求/adapter 映射证据。

服务实现完全未改，当前任务不是新优化候选，因此不为此再跑普通 Full。
后续沿 7B 正确性与共同 warm-reference 主线推进；达到新指标下旧/新 Prime
目标后才能进入 3B，最后恢复外部基线。缓存授权已由 D78/D80 完成，仍只复用
既有一次性只读交付缓存，不重建、不在请求路径打包。

产物：`paper_results/ieee_tc/p2_backend/20261002_d167_checkpoint_slot_content.{json,csv}`；
原始分析与完整派生摘要在 `results/ieee_tc/p2_backend_qualification/d167_20261002/`。
