# D421：vLLM 7B 全工件功能检查（预登记）

2026-10-11。状态：准备，尚未启动。原始结果唯一根目录：
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
