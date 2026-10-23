# D761：D760 内存停顿的有界归因（非性能重跑）

问题：D760 的 1478 个失败不能在未核查适配问题时用来证明 PrimeLoRA 优势。
固定 32 GiB store 不能单独解释后续增长；正常完成的 request_trace 会删除，
原生 TokenizerGroup 的 LRU 容量为 4，不能把它直接称为无界缓存。

本次唯一待证伪假设：3B 工件携带同一 backbone 的 tokenizer 文件，但原生
vLLM 0.10.2 按 LoRA ID 反复构造 tokenizer。即使 LRU 保留四个实例，构造
过程的临时对象、动态缓存类或原生分配是否延迟回收，仍须实际检查。

先复用 D760 的逐秒资源、请求、resolver 和五个 backend epoch 记录。另做
一次 CPU-only 原生 TokenizerGroup 诊断：使用同一环境、原工件与某个实际
backend 的前 128 次 resolver 顺序，不加载模型、不开 GPU、不启动远端、不
生成负载或工件。记录 RSS、GC 计数、LRU 数量及 tokenizer/动态类弱引用。
循环中不主动 GC；结束时单独进行一次 GC，仅用于区分仍存活与可回收对象，
不将其作为被测系统补丁。RSS 达 2.5 GiB 则结束循环，外层仍为 3/4 GiB、
swap=0 的受限辅助域，最长 180 秒。

停止规则：诊断只回答该具体假设。未证实时保留未解限制，不追加无目标的
整轮重跑、提高内存/超时或开展 baseline 参数搜索。证实时仍须区分原生行为
与必要适配缺陷，再决定最小修复；不能仅凭与停顿相关就宣布全因果闭口。

依据：已读实际执行源码，并在线核对同版本官方
[TokenizerGroup](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/transformers_utils/tokenizer_group.py)
及 [tokenizer 缓存实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/transformers_utils/tokenizer.py)。
其他版本的 GitHub memory-leak 报告与本路径不匹配，不用其充当本机根因证据。

## 结果（已完成一次原生 CPU 诊断）

使用原 backend PID 990137 的 resolver 访问顺序。第 120 次访问达到
预登记 RSS 停止线，实际发生 79 次原生 tokenizer cache miss；不是完成
128 次，也不是一次端到端实验失败。整个检查 91.03 秒，未加载权重、未
启动 GPU/远端。受限域 peak=2533646336 B，high/max/OOM/swap 均为 0。

| 状态 | 进程 RSS（B） | LRU 实例数 | 尚存活的动态缓存类 |
|---|---:|---:|---:|
| 原生组初始化后 | 860295168 | 0 | 0 |
| 120 次访问结束 | 2726866944 | 4 | 35 |
| 仅用于归因的完整 GC 后 | 2429382656 | 4 | 4 |

RSS 增加 1866571776 B；GC 后减少 297484288 B，但没有回到初始化水平。
弱引用显示淘汰的 tokenizer 实例确实释放，而部分动态类仍存活；完整 GC
回收了这些类。官方 `get_cached_tokenizer` 的动态类方法闭包持有原始
tokenizer 和 vocabulary，因此缓存字典容量不是该路径的全部内存上界。
剩余 RSS 不能仅由此判成不可回收泄漏：可能包含分配器保留、碎片及其他
原生状态。本检查未取得 D760 运行时堆快照，不能量化其对总停顿的贡献。

D760 原日志的独立时间线：

- 首次 memory.high：启动后 1928.075625 秒，当时已有 1956 个响应。
- 两个相邻响应间最长空档：3450.220913 秒；空档前 2489 个响应，之后 33 个。
- 资源增长发生于持续服务阶段，而不只是初始 store 分配。全部 resolver
  完成分布为 PID 990137:1056、991443:1006、992035:1001、1034817:744、
  992808:193。固定 300 秒分箱见 `timeline_300s.csv`，不按结果选择窗口。
- 4000/2522/1478 的 offered/响应/失败账目不变；未修改 D760 数据或判优身份。

第一次离线分箱读取了错误字段 `adapter_id`，真实 resolve_completed 使用
`tc_artifact_id`。保留原脚本、失败回执和关闭记录，只修正离线读字段再分析；
CPU 原生诊断没有重复运行，所有失败保留，不涉及生成或性能计量改变。

## 决定与下一主线

1. **已支持的观察**：相同 tokenizer 随 LoRA ID 的反复构造可产生可观测的
   额外常驻内存；不能继续用“LRU=4”关闭该问题。
2. **尚未支持的推断**：它是 D760 全部超时的唯一原因，或该现象是所有
   ServerlessLLM 部署的必然机制缺陷。均不作此主张。
3. **必要适配候选**：按内容身份复用共同 tokenizer，而不是周期性强制 GC、
   重启、放宽保护或改 baseline 调度。既有远端内容索引显示 500 个 adapter
   的 tokenizer.json、tokenizer_config、special_tokens_map、config 和
   generation_config 各只有一个 SHA；正式候选仍须验证它们与 backbone
   一致，且不更改权重、LoRA ID、token 输出与真实传输合同。
4. 后续 vLLM 0.11.0 官方已采用共同 base tokenizer，并明确弃用逐 LoRA
   tokenizer（[同版本源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/vllm/v1/engine/processor.py)）。
   这是表示方式的参考，不是擅自升级 ServerlessLLM 后端、也不是这次异常
   已在该版本被证明修复。保留现有 vLLM 0.10.2、RR、原生 store 和扩缩容。
5. 下一步只验证这一内容等价适配（原生 CPU 编解码/有界状态、必要 mixed-LoRA
   路径），通过后重做受影响的完整 W0 新版本键，再接同配置 W1/W2。
   不重复 500/1000 整套资格、不扩展 baseline 性能搜索。若等价条件或改善
   假设不成立，明确记录边界后回到矩阵，不堆叠补丁。

原始证据：`results/ieee_tc/serverless_qualification/d761_20261023/`；
curated：`paper_results/ieee_tc/baseline_audit/d761_3b_memory_cause/`。
本轮没有修改被测服务代码，也未改变 IEEE 公式、评价协议和其他未完成实验。
