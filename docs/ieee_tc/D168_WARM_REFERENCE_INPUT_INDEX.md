# D168：共同 warm-SLO 标定的既有输入索引

## 本项完成什么

复用 D164 已封存的 4,000 请求投影与 seed42 原 trace，固定 V1 共同 warm
参考的输入分组和抽样索引。不重跑 D164、不解析其大原始结果、不生成新 trace、
不复制工件，不修改服务实现、九个公式或冻结指标 V1。

这是共同测量的准备步骤，不是外部 vLLM 基线比较，也不是性能优化结果。
真正的 warm 数值、batch 可行性、数值身份、Resident 预算和 7B G1/G2 仍未合格。
已完成的 D166/D167 内容检查不重做；仅用名字与 token 数不能补造执行映射。

## 选择规则与参考依据

按 V1 §5，以全部原请求的**原生实际执行输入长度**求 Type-1 四分位点，合并
相等边界。组内依照原始到达顺序，等距选 256 个不同的既有请求；索引为
`floor(i * eligible_count / 256)`，`i=0..255`。三轮重复相同输入索引，只反映
运行变异，不声称三个新 workload。抽样完全不读取 TTFT、TPOT、输出内容或胜负。

输入依赖逐项绑定原 request/adapter/target、canonical prompt SHA、native
prompt-token SHA 与源行 SHA。输入全集不全、重复/外来 ID、合同不一致、错误
hash 或超长上下文均拒绝。TPOT 标定要求 target≥2；不足 256 个不同合格请求的
分组显式报错，不复制样本凑数、不生成替代请求。本次没有单 token 排除项。

前置检查仅验证输入身份。单元测试刻意改变所有时延和历史 success 字段而不影响
索引，是为了验证无性能结果选择；这**不表示失败运行可以用作已合格 warm 样本**。
实际测量必须另验每个请求原生完成、GPU-ready、时间边界与资源释放。

[HydraServe 原文](https://www.usenix.org/system/files/nsdi26-lou.pdf)提供所采用的
5×TTFT/2×TPOT 倍率背景，本项目的分组与样本规则仍以 V1 为准。
[vLLM 0.30.0 AsyncLLM 官方源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)
是后续复用原生生成/计量边界的实现依据，不借用其结果作为本机测量。
沿既有 preflight/runtime 接口扩展，不新建回放框架。

## 输入核查表（不是性能结果）

| 组 | 全部请求 | 执行输入长度 | 固定抽样数 | 抽样实际 adapter 数 | 抽样 target 范围 |
|---|---:|---:|---:|---:|---:|
| 0 | 1,039 | 14–616 | 256 | 88 | 6–244 |
| 1 | 2,961 | 626–760 | 256 | 94 | 6–256 |

原始四分位点为 `[616,760,760]`，最终有限上界 `[616]`。不存在第三、第四个
不同长度区间；不得人为拆开相同长度、重复计作四个独立组。后续按各请求所属
组使用阈值，不以两组等量抽样重加权主负载的 4,000 请求分母。

选中输入范围分别是 17–616 与 666–760。760 是后端实际输入长度（含特殊
token），不意味着改变内容最多 759 tokens 的生成合同。选中的原 prompt 和
target 不改写。后续执行须重新核对两个 prompt SHA 和实际长度。

每组 256×3，共计划 1,536 个测量调用；不是本项已经执行了 1,536 次推理。
按原顺序八请求分批，两组最大不同 adapter 均为 8；最大声明上下文总量分别
为 5,311 与 7,865 tokens。它们只是实际容量资格的输入需求，不是 KV 可行性
结论。不能拿旧 Full 的 cap4/profile 直接宣布 batch8 不可行，或无验证提高并发。

| 检查 | 结果 |
|---|---|
| 新增纯输入选择测试 | 7 项通过 |
| 既有 preflight 回归 | 72 项通过 |
| 原请求/adapter/target/输入 hash 绑定 | 4,000 条完整 |
| 新 GPU 推理、远端操作、工件/负载生成 | 0 |
| 历史保护文件 | 147 项通过 |
| CPU 测试及分析 high/max/OOM/swap | 全部为 0 |
| 共同 warm 时延、阈值、batch 资格 | 尚未测量/未冻结 |

按计划 §11 与 `academic-plotting` 的数据表达原则，交付输入/状态表和 CSV，
不把准备工作画成性能提升。仅增加 preflight 的纯索引函数，既有函数与服务
源码不改。输出 JSON 的 `measured_warm_reference` 与 `formal_g1_g2_qualified`
均为 false，`batch_size` 和 `thresholds` 均为 null。

## 下一项（仍为 7B 主线）

复用此固定索引，在既有受限启动与原生 runtime 上实现 GPU-ready warm 测量；
先核验八请求批次的实际 slot/KV/调度可行性，再冻结最大共同可行 batch≤8。
计量采用原生 dispatch→first 和 first→last，不将 Prime 的路由、准备或队列
延迟加入 warm 参考。批次之间完全 drain，加载/引用建立放在该参考的测量边界
之外并单独记录，整体 GPU 持有仍如实保留。不得为获得更宽 SLO 而选慢样本。

实际请求→adapter 执行映射证据仍需补齐；不能把本索引或之前的槽位内容检查
追改为完整正确性。完成共同参考和必要旧/新 Prime 测量后，才按新核心指标验收
7B；然后 3B，最后外部基线。一次性远端交付缓存继续复用 D78/D80，不重建。

产物：`paper_results/ieee_tc/p2_backend/20261002_d168_warm_input_index.{json,csv}`；
原始测试/索引分析：`results/ieee_tc/p2_backend_qualification/d168_20261002/`。
