# D445：Loquetier 单次完整 7B W0（预登记）

日期：2026-10-12。沿已批准主线，D442 原100、D443 全500-ID覆盖及 D444
原1000已完成，不重复。当前执行相同冻结配置的一次完整4000请求，不进行
baseline性能调参或因结果不达标重复运行。

## 研究问题与输入

检查完整原始负载下的原生生成、adapter绑定、共同服务要求及生命周期GPU占用。
沿用已有seed42 trace，不新建权重或负载。4000请求、实际132个adapter、
458224目标token，view SHA
`a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d`。
仍准备完整500逻辑adapter池，不利用未来热点缩池。500逻辑ID对应498零权重、
2非零ID、4权重SHA/6内容树类别的限制继续公开。

原生SMLM、FP16、batch2、context1024、顺序启动、四个125-ID静态分片保持不变。
配置 `configs/ieee_tc/loquetier_7b_static_native_v1.json` SHA
`d9f3b81d3d90340739ec66b45d36c1c71c706640dd2cd790a19d3ea4a80a617a`。
原生Python3.9、公共Python3.12、kernel/source-view/runtime/analyzer均不变。
这是Loquetier加静态分片部署，不冒称官方原生集群调度，也不是配置搜索得到的最优点。

共同部署通知后准备，业务在notice+60秒开始，不移动到ready之后。
复用D78/D80通知前只读gzip发布；不重新打包、发布或新增解压工件池。
真实读取、传输、竞争和本地物化计入服务时间。完整500准备传输只计一次，
后续请求引用不能另计为远端传输。

## 资源与执行

- 服务72/80GiB、swap2GiB、CPU4–23,28–47；外置3/4GiB、swap0、CPU2,3,26,27。
- 一次一个重任务；推理机磁盘150/100GiB、防OOM和独立watchdog不变。
- 预估新增20GiB；启动前重新核验所有资源及真实远端链路。
- 原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d445_20261012`；私有路径：`/tmp/tc-d445-7bl`。
- 复用D444执行/封存脚本，仅调整campaign和既有完整输入视图。
- 端口8510/8511、8523/8533/8543/8553，tmux `tc-d445-7b`。
- 保留资格/开发身份及1800秒请求保护，不把已有已知负载改写成独立留出评估。

启动前53项MAIN测试（0.447秒）和18项BASE测试（0.821秒）通过。
D444全部580来源SHA、147旧保护条目、三份用户修改和两原生checkout补丁未变。
检查scope实测peak49131520B，high/max/OOM/swap0；核验实际递归进程树为空和
InvocationID后停止。双仓库备份后才冻结实际执行commit，再过fresh gate和远端health。

## 验收、报告与下一步

严格分析使用 `loquetier_static_shards_v1`、`full4000`，不传D385原100数值参考。
必须完整请求/原生数量/来源/绑定/可靠计时，并验证实际GPU释放及所有失败。
共同D394阈值不变；联合至少3800/4000才可进入G1达标比较。
原始未知数值正确性标志不改写；内容/注册/指针/实际forward绑定与独立完整数值
语义证明分别说明。没有独立证明的部分保留未知，不能由HTTP200推定。

D444联合SLO6.6%仅是1000前缀观测，不能外推本次4000结果。无论本次是否达标，
均保留完整结果；不缩负载、换阈值、调优该baseline或重复寻找有利结果。
完整执行后：实际释放→身份核验并停止自有远端→保存双端证据→严格分析→封存及
双仓库备份→仅清理可重建下载副本→同合同主比较/HydraServe后续资格。
不重跑已封存PrimeLoRA、ServerlessLLM、vLLM、S-LoRA W0。
W1/W2/G2、3B、消融、机制、敏感性及文档仍OPEN；当前不画图、不修改论文。

## 实际启动（运行中，尚无完整性能结论）

2026-10-12 04:07:28（Asia/Shanghai）在tc-d445-7b中启动一次。
实际执行MAIN `9471e1ec60baabe78f6d9d00182c100b689e0719`、BASE
`311663aa501ea2f7b61a04f53fa471822272643f`，均在启动前推送并核验远端SHA。
输入、fresh资源检查和远端health通过；真实链路1000Mb/s/full。
服务InvocationID `26758f9ae61346babe00e308045d6be9`，独立监测已采样。
初始化准备尚在进行；继续监测同一运行，不启动重复4000或另一重型任务。
