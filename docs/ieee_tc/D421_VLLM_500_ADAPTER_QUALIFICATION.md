# D421：vLLM 7B 全工件功能检查（预登记）

2026-10-11。状态：16:13:38启动，约16:25完成，原生全池功能核验通过并收尾。原始结果唯一根目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d421_20261011`。

目的仅为补D420确认的500个逻辑ID原生执行覆盖。复用D419已验证的vLLM0.30.0
入口及公开配置，不修改引擎、路由、缓存、生成或资源策略。配置SHA
`361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5`。
baseline现存checkpoint `a56d79088ab1e431e12b5cfed95f53a1581d7037`；D418实际
源码checkpoint `3bd6088ce988a4108f397378c5b59a8e586cbc8f`。本轮不改baseline源码。

沿用现成`adapter_coverage_v1`，500请求、500既有ID、68925目标token，视图SHA
`286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d`。
源trace/pool分别为`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`、
`e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c`。
这是功能覆盖而非原W0性能点，不能用其时延或SLO决定工作点。

固定4张GPU、DP4/TP1、原序号round-robin，native LoRARequest/LRU，FP16、max-seqs2、
max-loras4、CPU-loras16、rank64、1024context/tokenbudget、graph、chunked prefill、
无prefixcache，seed0。全部在现有共同服务资源域内，预先独立open-loop到达，
60秒通知/1800秒资格保护，真实物理持卡至释放。原生编译磁盘缓存单独登记；
不预启动引擎，per-run LoRA cache为空。复用D78/D80只读发布对象，真实获取计时，
不在线打包、不重新发布。预计新增峰值20GiB，按150GiB启动/100GiB运行底线检查。

入口：raw目录`run7b.sh`，仍调用现有`ieee_tc_preflight.py gated-launch`和
`replay_vllm_engine_trace.py native-http-service`。原有远端管理/health/证据保存
脚本只替换campaign路径；不重建实验框架。只有本次明确所属服务可收尾。

验收：500/500 offered/terminal/native token/adapter/source/content/timing，
恰好覆盖冻结500-ID集合、无fallback；真实远端交付/收据与内容关联；四worker
来源和运行资源读回；物理leases全部关闭；旧保护清单零变化。现有analyzer的
`--scheduler-contract resident_rr_v1 --population adapter_coverage_v1`执行核验。
数值独立性限制、重复／零权重事实及原始unknown正确性字段照实保留。

结束顺序：实际GPU释放→停止本次远端服务→保存证据→严格分析→备份；再进入
原1000兼容和一次完整4000 W0。失败保留，不以短检查SLO不高作为重复的理由。
不重跑D419、不触动PrimeLoRA/ServerlessLLM封存点、不生成排名图。

实际执行main `aa76e000127a7cc19d275f8140d0bd8366b3ebc1`已推送并核对远端；
baseline SHA如上。tmux `tc-d421-7b`，服务InvocationID
`2afee10ae3244d309ce4ec5070cd5738`，通知时刻1213672.816455652（本机monotonic）。
启动前27项分析合同测试通过；本轮没有修改baseline运行代码。完整输入/编译缓存
初态以及实际资源身份记录在raw。后续只监控同一运行，不将等候误作重启理由。

## 终态与复用范围

| 功能检查 | 结果 |
|---|---:|
| offered／submitted／terminal／原生合同 |500／500／500／500|
| 实际执行逻辑adapter ID |500（与冻结清单集合完全相等）|
| 失败／原生输出token |0／68925|
| E2E、TPOT、原生边界最大重算误差 |0ms|
| 真实HTTP交付／请求时打包 |500／0|
| 线上压缩字节／已核验物化字节 |420740410／12985984450|
| 关闭物理租约／实际服务worker |4／4|
| 生命周期占用 |2722.8814599129837 GPU-s|

平均/P95 TTFT644.341327/2135.383901ms，平均E2E4069.208638ms，仅作为这次
功能视图的描述值，不与原W0性能或D419启动前缀作收益比较。原生编译缓存604
文件/53278235字节，SHA`b0c05a6113d73a38e1cc0e16949b0ef13ec0c2bfe62d9a6479abd701866050d5`；
磁盘缓存不是预热GPU或LoRA。原始`n_correct=null`和独立数值资格字段不改写；
此次证明原生绑定/调用/固定工作量/内容和计量合同，不证明500种独立训练模型。

服务内存峰值24619839488字节、主机可用内存最低99819220992字节；high/max/
OOM/swap均0。保护清单147条零变化。四卡原生context已释放，服务域移除，退出码0；
三个明确所属远端服务按InvocationID停止、MainPID0、端口无监听。resolver/worker/
remote证据已保留，远端源文件与副本SHA一致。暂未删除下载对象缓存。

严格审计：`paper_results/ieee_tc/baseline_audit/d421_7b_coverage/audit.json`，SHA
`6539a15a293aad9b0e63f4ee1e41b2689847498940468c44b8188a0c583d6277`。
568文件/35339691字节证据清单SHA
`293c40bd75e447a36945dfb548759acbe3da50cfc336829053f91ec2bcaa341e`。
新增CPU回执在raw的`cpu_contract_tests_receipt.json`及log：27项通过0.174秒；
源码与D419一致。这是新的终态后回归记录，不倒填D419旧0.090秒执行回执。

下一步：备份后，仅回收已核验可重建的本轮下载对象，保留原始证据和唯一工件，
恢复150GiB启动余量；再用现有入口执行原1000请求兼容和一次完整4000 W0。
500功能点不重复。PrimeLoRA D371、ServerlessLLM D414封存不变；U_ref仍开放。
