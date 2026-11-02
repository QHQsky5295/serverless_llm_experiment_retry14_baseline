# D1026：保持原子发布的独立文件并行校验候选

2026-11-03，接续D1025；94项CPU回归和一次三工件组件观察完成，
具备进入D1027完整W2的局部证据，尚非接受的端到端性能改进。
本轮继续3B主线，不重新部署共享k3s，不修改远端，不增加GPU诊断负载。
用户dirty的EXECUTION_STATUS不覆盖，当前交接以本文件及D1025为准。

## 假设、历史与约束

D1022完整W2为3770/4000联合达标（94.25%），物理GPU占用8062.415716s，
距共同3800要求还差30。独立数值正确性仍未知，不能宣告G1/G2闭口。
D1025同路径CPU组件观察到发布执行93–152ms，故意排在发布后的来源查询
自身不足1ms而需等待事务。该组件不等于线上因果分解，也不保证能补齐30条。

历史D628/D631将hash移到锁外，未减少hash工作且完整回放无收益，已关闭；
D850提前通知/预发租约、D851–855 owner线程、D844/D860 stat改写也不重开。
本次保持owner事务锁、capacity、文件路径/大小/签名、发布前后完整观察和
rename顺序，全部独立文件的完整SHA均通过并join后才发布。只并行独立文件，
不拆分SHA身份、不采样、不跳过校验，不改论文九式或配置。

并行数=min(实际文件数，进程CPU affinity数)，单CPU直接串行；不按W2或
结果调参。每reader仍使用原1MiB块，暂存与线程数有界，继承既有服务资源域。
executor在成功、校验错误和提交错误时都join已提交reader；未退出不得退役
workspace或释放owner锁。不另建常驻线程队列或改变权威状态owner。

2026-11-03核查原始依据：

- [Python hashlib](https://docs.python.org/3.12/library/hashlib.html)：大块update释放GIL，支持独立SHA并行的可能性，不证明端到端收益。
- [CPython 3.12.12 executor实现](https://raw.githubusercontent.com/python/cpython/v3.12.12/Lib/concurrent/futures/thread.py)：shutdown(wait=True)等待线程退出；本实现使用context manager保留异常退出join。
- [ServerlessLLM OSDI论文](https://www.usenix.org/system/files/osdi24-fu.pdf)：借鉴在存储加载执行层利用独立工作并行的原则，不声称该系统采用这里的文件SHA方法，不引用其异构硬件加速数值。

## 有界验证和判定

1. 复用真实owner/HTTP测试，核查完整字节校验次数、持锁排他、无提前可见、
   错误时先join后清理、单CPU、损坏与替换保护；重查受影响process/HTTP路径。
2. 复用D1025原component入口、LLM CPU环境、同三工件、同四CPU与3/4GiB
   包络，一次测量；保留D1025原始对照，不创建新权重/trace，不覆盖输出。
3. 无可解释组件收益或正确性失败则归档；有收益才推进下一唯一完整Full W2，
   不另加短GPU剖析、不叠加capture/control候选。组件成功不称主目标达标。

协议仍为PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83，
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22，
单轮V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
当前代码base为7d1e14142fd113228b64d7fbc37d3af7d84556a7。
未完范围保留：同版本两模型完整W0/W1/W2及G1/G2比较、独立数值正确性、
A1–A5、S1–S13、CE/Pareto、设计与审稿文档、后期图表及最终V2集成。

## 首个测试批次的环境中止

95项测试合并在vLLM环境的3/4GiB辅助域运行；37项实际HTTP/owner发布检查
完成后，spawn进程测试出现两个失败标记，同时产生额外Python编译子进程和
memory.high回收。中止前快照peak=3417489408B，high=40825，max/OOM/kill/
swap=0；这是中途快照，不冒充退出终态。完整失败traceback/JUnit尚未产生，
不能把具体失败都断言为内存所致，也不能把整批写成通过。只向已核对的本次
scope发SIGINT、继而SIGTERM，命令exit143；之后scope inactive/dead。
原始状态保留在tests_attempt01_abort.json，不删除或混入性能结果。

D1025已发现vLLM解释器的项目sitecustomize可能导入GPU worker和编译任务；
本次合并测试仍遇到了该环境边界，故后续文件/HTTP检查统一复用既有LLM CPU
环境，保持源码、测试和3/4GiB包络，不提高上限、不顺便修改启动钩子。
这不是正式后端版本更换，也不重做GPU资格。完整输出保存tests_cpu.txt。

147项旧投稿保护在改动后核查通过；用户manifest和EXECUTION_STATUS的SHA
分别仍为3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df、
41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374，未修改/暂存。

## 组件结果与进入完整回放的决定

既有LLM CPU环境运行37项ConfirmedFilePublication、20项FileStateProcess及
37项AtomicArtifactPublication测试，94项全部通过，238.114秒；输出完整保留。
unittest未收集原pytest模块级旧roundtrip函数，不写成95项通过。
其中3个新增测试分别证明：独立reader实际并发、发布线程始终排他且目标未
提前可见；一个reader失败时仍等另一reader退出再清理；单CPU不创建线程池。
测试运行中最后记录peak=2742087680B，high/max/OOM/kill/swap=0，不作为
退出终态快照；scope已inactive。它不是新的GPU正确性或性能运行。

复用原component入口，三工件、内容清单、bytes、SHA、CPU2/3/26/27和资源
包络与D1025一致，51命令均成功，计时边界单调、内容身份一致、临时目录退出。
报告component.json SHA为205c1807d987b35c73fad50459534ebdbd96d876381a24fded1fdc8461464ab1。
表中每个工件/实现各一次，不作CI或重复均值；查询均值仅是该次8个排队命令。

| 工件 | 串行发布wall ms | 并行发布wall ms | wall下降 | 串行/并行累计CPU ms | 串行/并行查询自身均值 ms |
|---|---:|---:|---:|---:|---:|
| gaming_lora_0010 | 92.727130 | 82.368053 | 11.1716% | 92.712197 / 130.775691 | 0.912002 / 1.450096 |
| medical_lora_0014 | 111.241301 | 90.188130 | 18.9257% | 111.223165 / 182.680066 | 0.921656 / 0.950151 |
| code_lora_0495 | 152.256900 | 136.373622 | 10.4319% | 152.234557 / 218.941920 | 0.928581 / 1.464458 |

查询提交→执行最小/最大ms由93.732060/100.439593变为83.212429/94.552416，
112.002177/118.946703变为90.744142/97.882348，153.590321/160.277469变为
137.189530/148.624747。后续等待观测下降，但查询自身并非全部更快。
并行发布提交→执行为0.742126/0.546708/0.697077ms，取得回执尾部为
0.586249/0.562450/0.617332ms，不冒称纯IPC。

**解释与限制：**更短wall以更高累计CPU为代价；四逻辑CPU仅两物理核，不能
把这一组加速百分比直接搬到正式服务的40逻辑CPU包络，更不能从旧请求时间
扣去组件差值构造新SLO。现有三个工件均含多个大文件但大小不均，完整SHA
仍读取全部bytes。本轮已证明执行时间有局部改善，不证明端到端净收益。

component资源退出前peak=1721901056B，high/max/OOM/kill/swap均0，实际
affinity=[2,3,26,27]；Python3.12.12，LLM CPU环境。生产源码SHA：
residency_manager=6e98b7bafe53317428268c20bc833582a5e21169468373cb30d502a5efb3470a，
FileStateProcess仍为e45ca417cb5a14f205d8451b543546ed961100e271b86680c80f169577440442；
原component脚本SHA不变。请求生命周期测试SHA为
5101692934e2e11ca204a020ee7db4a80356f6f7f299f22a1c17953c474b1581。

决定：不追加另一CPU工作点或GPU短剖析，直接登记D1027一次完整4000请求W2，
除文件校验并行外保持D1022冻结配置/画像/原生后端和评价尺子不变，实际CPU
代价与GPU生命周期都进入测量。若整体退化，保留并否决，不重跑同键找胜出。
