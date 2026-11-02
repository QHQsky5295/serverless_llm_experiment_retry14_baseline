# D1025：发布执行与串行文件状态等待的有界组件观察

2026-11-03。接续 D1024；完成诊断计时、20项相关回归和三个既有工件的
CPU组件观察。**没有新GPU回放，没有新的性能候选被接受，没有修改九式、
配置、阈值、远端或共享k3s。** 用户dirty的EXECUTION_STATUS.md不覆盖，
当前交接以本文件为准；完整目标继续active。

## 1. 主线与本轮问题

完整重读目标、PLAN、冻结V1/单轮V2、存储规范；复核D1024、D887、D850、
D628/D631和D844/D860。使用analyze-results区分观察与因果解释；使用
run-experiment复用现有组件和资源包络，不重建GPU实验框架。

当前完整3B W2仍是D1022：4000请求完成原生合同，3770联合达标（94.25%），
8062.415716物理GPU-s，P95 TTFT 1997.601528ms。距3800还差30；独立数值
正确性未知，n_correct仍为null，不宣布G1/G2最优。D810已完成的外部基线
观测及HydraServe隔离适配直接保留，不因延迟收到集群答复重新部署共享k3s。

D1024恢复了479条共享预取关联。发布/发布后交接占相关TTFT违约请求准备
包络的57.50%，但不能等同可删除锁等待。本轮只检查：文件状态进程的一次
发布操作本身耗时多少，以及排在它后面的来源查询是否主要在等待执行。

## 2. 计时实现及边界

仅扩展既有`faaslora/memory/file_state_process.py`，新增显式
`FileStateProcess(..., diagnostic_timing=True)`。默认false；生产facade、
runner和冻结配置不接入这个开关。没有另一个账本、调度输入或无限记录队列。

每个已接收回执包含：

1. parent_submitted_ns：提交入口，包含随后序列化及父侧锁等待；
2. worker_started_ns：子进程开始命令，包括参数解码；
3. worker_ended_ns：原操作和回执构造结束；
4. parent_observed_ns：调用者第一次取得回执，早于文件变化回调；
5. worker_cpu_ns：上述子进程命令的进程CPU时间。

四个wall边界来自同机monotonic_ns；CPU量单列，不再加进wall分解。
submit→worker包含IPC、executor排队、进程初始化等，不命名为纯队列时间；
worker→parent包含IPC、反序列化、调度及调用者延后join，不命名为纯通信时间。
identity首调用包含spawn/import，不能作为稳态RPC成本。异常仍保留原类型；
取消后重join同一命令只记录第一次接收，不重复突变或事件。

20项现有/新增测试通过，19项既有依赖警告；覆盖默认不插桩、成功/失败边界、
取消重join、真实spawn、容量、损坏拒绝、原子来源获取、pidfd和进程失败。
测试使用现有vLLM环境；这次没有留存测试scope退出前峰值，不补造零事件。
夹具修正后又在显式taskset包络中重查3项计时/spawn测试，全部通过；它们是
上述20项中的复查，不额外计算为23项独立测试。

## 3. 复用真实工件的组件结果

入口扩展在原`tests/test_ieee_tc_file_state_process.py`，没有另建runner。
从既有500项内容index按总逻辑字节、ID排序，固定选最小、上中位、最大；
每次只复制一个工件到短暂私有workspace。所有文件逐字节校验并正常发布，
未改原工件、未生成权重/trace、未复制整池。源SHA与执行源码SHA在报告中。

这不是原生HTTP回放：没有归档/解包和后台请求，缓存初始为空，用prepare_copy
进入相同确认发布路径。八个来源查询被故意排在发布后，用于观察单worker依赖；
不能当作自然线上发生频率、8并发性能或4000请求的因果对照。

全部51个命令成功，三次发布内容与冻结清单相等，逐命令边界单调且分解恒等。
三个组件各执行一次，不作运行间均值/CI，不挑选一个作为正式结果。

| 工件 | 逻辑字节 | 发布执行wall ms | 发布CPU ms | 提交→执行 ms | 执行→取得回执 ms |
|---|---:|---:|---:|---:|---:|
| gaming_lora_0010 | 28015916 | 92.727130 | 92.712197 | 1.034615 | 0.401805 |
| medical_lora_0014 | 38501676 | 111.241301 | 111.223165 | 0.753954 | 0.316377 |
| code_lora_0495 | 56327468 | 152.256900 | 152.234557 | 1.341710 | 0.300102 |

下表均为同一组件内八个故意排队的查询，不是八轮实验。

| 工件 | 查询执行wall均值 ms | 查询提交→执行最小/最大 ms | workspace退役执行 ms |
|---|---:|---:|---:|
| gaming_lora_0010 | 0.912002 | 93.732060 / 100.439593 | 1.616241 |
| medical_lora_0014 | 0.921656 | 112.002177 / 118.946703 | 1.520145 |
| code_lora_0495 | 0.928581 | 153.590321 / 160.277469 | 0.518701 |

1. **观察：**发布在空缓存组件中已需93–152ms，CPU与wall接近；后续查询自身
   不到1ms，必须等前面的命令。**解释：**单worker串行事务确有这种依赖，
   不能把一次查询的完整等待都解释为查询自身扫描工作。
2. **边界：**这没有测出D1022在线发布包络的排队比例；不能从D1024的321/
   456ms直接减去这里的组件值。缓存大小、CPU竞争、环境及到达条件不同。
3. **边界：**publish_transfer包含库存、签名、完整内容SHA与rename；CPU接近
   wall不等于本轮已经把CPU时间全部精确归因到SHA。D627/D628的旧hash细分
   仅作为相同路径的历史旁证，不冒充本次同环境剖析。
4. **结论：**不据此修改SLO或再开同配置完整GPU诊断；此轮证实观测接口和
   一个具体串行依赖，为下一有界工作量优化提供方向，不宣称性能已提升。

## 4. 两个未成功尝试及资源事实

保留`component.json`和`attempt01_source.json`：首个夹具把2×逻辑字节加
1MiB误当预算，遭真实ext4预留规则拒绝。它尚未执行发布，不是性能负结果。
修正仅让夹具用既有_file_allocation_geometry/_file_allocation_ceiling求预算；
生产规则不改。最终三例预算168165376/231079936/338034688B，不是实际多写
这些字节。源快照存为JSON中的原始patch，base为7489cc9；可按patch SHA还原。

第一次仅设置AllowedCPUs，实际affinity仍0–47；memory.high=4091，峰值
3303809024B，max/OOM/kill/swap=0。不能把它说成已固定四CPU的测量。
随后显式taskset到2/3/26/27，保持3/4GiB、swap0，没有扩大资源门槛。

第二次采用vLLM环境并显式PYTHONPATH，项目sitecustomize自动导入原生worker，
实际出现humming构建子进程并持续触发memory.high。保存component02_abort.json
后，只对已核对身份的本次scope发送SIGTERM，exit143；没有OOM，没有完成
报告或可用性能点。原始消息包括退出时5个semaphore的resource_tracker清理
警告，不把它写成无异常退出。scope已inactive，空的私有host/nvme目录经
ls核查后用rmdir删除；没有删除任何测量、权重或其他项目数据。

第三次复用既有LLM CPU环境（不是替换正式vLLM后端），同一源码和三工件
组件完成。峰值1845735424B，high/max/OOM/kill/swap=0，父进程实测affinity
为2/3/26/27，资源限制实际读回3/4GiB和swap0。各spawn初始化7418–11077ms
单列，不与稳态发布比较。没有启动推理GPU工作进程、GPU回放或远端管理操作。

## 5. 后续单一方向与关闭方向

不重开D628/D631的解锁后hash、D850提前通知/预发租约、D851–D855 owner线程、
D844/D860 stat改写，也不再做已经完成的进程隔离。它们的负结果完整保留。

本轮在文档和git历史检索中未找到**保持发布事务锁不变、仅并行不同文件的
完整SHA**的已验证候选；这与移锁/少校验/提前发布不同。下一有界任务先复核
相关代码历史，再在同一既有组件中测试这个可证伪假设：每个文件SHA仍完全
相同，全部校验成功并join后才继续原后观察、容量及rename；只缩短独立文件
校验的串行执行时间。并行度应由本进程可用CPU和实际文件任务数约束，不按
W2结果调一个魔数，不拆分SHA后用另一种摘要冒充相同内容身份。

若组件无收益、身份/取消语义不成立或引入资源问题，就归档，不启动GPU。
若有可解释收益，才形成一个正式候选、完成受影响回归并用下一唯一Full W2
验证整体SLO/GPU-s；不同时叠加capture或控制策略候选。此处只是下一任务，
尚未实现并行校验，也不预先保证能补齐30个请求。

原始资料在2026-11-03重新核查：

- [CPython executor源码](https://raw.githubusercontent.com/python/cpython/v3.12.11/Lib/concurrent/futures/process.py)：提交、工作队列、子进程与回执是不同边界，支持分开观测，不给出本机耗时。
- [Python 3.12 time](https://docs.python.org/3.12/library/time.html)：monotonic与process CPU计时语义不同；不跨远端主机相减。
- [Python 3.12 hashlib](https://docs.python.org/3.12/library/hashlib.html)：大块update可释放GIL；因此不同文件的校验有并行可能，不代表必然获得端到端收益。
- [ServerlessLLM OSDI论文](https://www.usenix.org/conference/osdi24/presentation/fu)：把多层加载执行与调度分开讨论。本轮仅沿真实加载依赖定位，不借用其H100/其他条件下的加速数值。

## 6. 复现、保护与完整未完范围

成功组件入口（输出必须是新路径；不要覆盖本轮文件）：

```bash
taskset -c 2,3,26,27 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/home/qhq/serverless_llm_experiment_retry14_baseline \
  /home/qhq/anaconda3/envs/LLM/bin/python tests/test_ieee_tc_file_state_process.py \
  --component-diagnostic <new-output.json> \
  --content-index paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json \
  --artifact-root /home/qhq/serverless_llm_experiment/artifacts/frozen/llama32_3b_a500_v1_modelscope \
  --workspace-parent /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification
```

实际入口外层使用systemd辅助scope，MemoryHigh=3G/MemoryMax=4G/MemorySwapMax=0。
component03 SHA为2a767df1b475d28d79439ceca645371b7f475f4f5b63340d0487b79c1fe18a63。
当前FileStateProcess SHA为e45ca417cb5a14f205d8451b543546ed961100e271b86680c80f169577440442；
test文件SHA为ff69c8a2eb0bca96c0b11bfe0687ef624096dd5feed29aad3fe8024a80e7cb54。
residency_manager原SHA10fb41d67a31e6d22c840fd134a9e0dfc4182a97c2ee931cbde143d89ad1d333未变。

147项旧投稿保护通过；用户manifest与EXECUTION_STATUS的SHA分别仍为
3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df、
41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374，均不编辑或暂存。
PLAN/V1/V2仍为D1024记录的冻结SHA。最后可用磁盘161391173632B，约150.31GiB；
下轮重型任务仍需按预计增长量核查，不能把此刻余量视为足够新增完整回放。

未完范围不减：同版本两模型W0/W1/W2及G1/G2比较、独立数值正确性、A1–A5、
S1–S13、CE/Pareto、B1/B2/C1/C2/C3及两轮审稿证据、后期图表和最终V2集成。
已有baseline结果复用；不新增13B/SGLang、不改论文、不提前作图。

后置备份回执：9文件检查点7d1e14142fd113228b64d7fbc37d3af7d84556a7已推送并
ls-remote核验origin/retry14_continuous_queue_v2及faaslora_origin的
backup/d1025-file-owner-timing-20261103；没有强推。后者仍是可回退备份，
不是历史分歧V2的最终集成。提交路径、diff、源SHA和凭据模式检查通过；
两个用户dirty文件不变。收尾无D1025运行scope、无GPU compute进程；
可用磁盘161390657536B。本回执随下一正常检查点提交，不重复发布。
