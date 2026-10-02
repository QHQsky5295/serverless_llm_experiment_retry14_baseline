# D165：原生 LoRA 槽位内容检查入口

## 要回答的问题与证据边界

D164 完整 7B 回放结束并封存，但数值身份、共同 warm-SLO 和 G1/G2 尚未通过。
D63 的独立 HF/PEFT 对照中，错误 adapter/base 也通过同一概率容差；重复该提示
或事后改变容差不能补齐身份验证。现有 worker inventory 证明槽位归属和容量，
没有逐项读取实际槽内数值。本项补后者，不是新的性能优化或运行成绩。

可证伪检查：对已加载且保持引用的 adapter，在无未完成原生工作的独立诊断中，
同步设备，然后将每个实际 GPU 槽的 A/B 张量读回，和已注册 CPU 张量按原生
setter 的形状、零填充规则逐项精确比较；不使用概率容差或文本作为代替。
错误内容、非零残留、缺失 slice 未归零、非有限值或归属变化必须不能通过。

## 依据及本次实现范围

- 复用 `IEEEWorkerObservationExtension.ieee_worker_observation`，新增显式
  `audit_adapter_ids`；缺少 `synchronize=True` 即拒绝。默认观测不读回张量。
- 复用既有 vLLM 0.30 TP=1 HOST-copy contract，拒绝未知 setter/布局/版本；
  dense、merged packed、缺失模块的所有 A/B slice 与低秩 padding 均检查。
- 每次仅读回一个槽张量并构造其 CPU 期望值，不复制整池，不创建 GPU 张量。
  保存有限性、逐项不一致数、非零数及期望/实际 SHA；正负零按数值比较，
  SHA 另存，不将数值相等冒称所有位均相等。
- 前后检查槽映射和注册对象身份，不触碰 LRU 或载入/驱逐策略。
- 现有 `backend-model-check` 增加 opt-in `native_slot_content`，仍用既有
  trace 前缀、工件、独立运行时、物理资源分配与收尾。保持原生引用，在生成
  前后各检查一次；先确认 scheduler drained，不将设备屏障加入普通 Full。
- 此入口是本地现有工件的独立正确性诊断，不冒称真实远端主比较；性能结果
  明确不可复用。远端 D78/D80 已发布缓存直接保留，不重建、不启动服务。

官方实现中的零化、slice copy 及 LoRA 运算入口已联网并与本机版本对照：
[BaseLinearLayerWithLoRA](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/layers/base_linear.py)、
[MergedColumnParallelLinearWithLoRA](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/layers/column_parallel_linear.py)、
[官方 LoRA 层测试](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/tests/lora/test_layers.py)。
这里检查 setter 的实际产物，不照搬上游随机权重为新的实验工件。

## 验证状态

| 项目 | 当前状态 | 可支持的结论 |
|---|---|---|
| CPU 精确比较与错误注入测试 | 13 项通过，0.173 s | 仅检查器逻辑；不是实际 GPU 证据 |
| 既有 worker/RPC/生命周期回归 | 811 项通过，74.901 s | 默认路径及接口兼容性；含 9 项新增测试，不重复加总为独立测试 |
| 完整 preflight 检查回归 | 72 项通过，1.045 s | 包含新增 4 项；使用具备原安全接口的系统 Python |
| 7B 原生槽位内容运行 | 尚未执行 | 需要独立运行与真实收尾记录 |
| checkpoint→已注册张量 | 未完成 | 不由路径/名称或此比较反推 |
| 每 token 的实际执行映射与算术路径 | 未完成 | 生成前后槽相同不证明生成期间映射正确 |
| 500-ID / 完整语义资格 | 未完成 | 零权重输出等价不能作为身份区分证据 |
| common warm-SLO / G1/G2 | 未完成 | 不放宽冻结 V1，不填虚构值 |

依据计划 §11 与 academic-plotting，此阶段采用正确性状态表，无性能排名图。
未增加 Full 性能候选，不重复 D164。完成并备份检查器测试后，再进行一次必要
的 7B 实际资格检查及后续共同 reference 测量；外部基线和 3B 仍暂停。

## 安全与来源

父检查点 `70029ba3ff68c4914e1d4a08c40b18298758aa56`；普通 Full runtime
`1acd8f43676e5e1fc25afada5736c0607032ebb0`。只改独立诊断接口与测试，
IEEE 九公式、服务策略、配置、原始结果、V1 和只读交付缓存不变。
原始测试回执放 `results/ieee_tc/p2_backend_qualification/d165_20261002/`。
推理机当前可用盘空间低于 150 GiB 新重型任务门槛，因此本阶段只运行受限
CPU 测试；不能因此降低门槛或删除唯一证据来启动 GPU。

三项测试均在 CPU `2,3,26,27`、MemoryHigh/Max=3/4 GiB、swap=0 的独立
资源域完成，退出码均为 0；最高单进程 RSS 分别为 1,164,888 / 1,199,536 /
39,552 KiB，命令总耗时分别 10.92 / 87.26 / 1.22 秒。high/max/OOM 和 swap
计数均为 0，实际资源域已移除，无 GPU 作业。没有失败运行或盲目重试。
准备回归命令时发现一个猜测的测试文件名不存在，在执行前换为实际现有
source-profile 相关测试所在的 launch/request-lifecycle 模块；不涉及服务修复。

原生运行仍未执行。本记录只交付通过测试的检查器，不将 CPU fixture 转写为
GPU 测量。D164 的 115 条 output-hash 差异、正确请求数和共同 SLO 仍未闭合。
