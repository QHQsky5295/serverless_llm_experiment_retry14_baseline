# D419：D418 环境修复后的 7B 受影响路径资格

2026-10-11，启动前登记；尚未启动 GPU。
不是参数搜索或重复取最好结果。D417启动错误保留，D419为显式环境修复后的
新执行键；继续原100请求前缀、29个实际ID、17369个目标token，不重新生成数据。

执行 baseline 版本 `3bd6088ce988a4108f397378c5b59a8e586cbc8f` 已备份并核验。

配置仍为 vLLM0.30.0、DP4/TP1、原始请求序号RR、maxseq2、maxloras4、
CPU LoRA16、FP16、0.7GPU fraction、1024上下文/token预算、无prefix cache、
chunked prefill、原生图模式。仅工具链PATH绑定及显式MAX_JOBS2变化。
共同60秒通知、1800秒资格保护、固定输出、D394阈值不变。

原始根：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d419_20261011`

复用D417的 run/start/health/stop 入口，仅换本轮输出/监控/私有cache身份；
stop 对无传输日志的状态作显式记录，不把已完成停止误记为服务未关闭。
远端继续D78/D80预发布只读缓存，不请求内打包、不新增工件池或模拟延时。

CPU工具链探针已通过，尚无GPU结果。启动前记录实际相关原生编译缓存身份，
旧D417完整根缓存身份单独保留，不称重新全量核验。磁盘缓存不是GPU/LoRA预载。
本轮LoRA私有cache `/tmp/tc-d419-7bv` 必须全新；既定72/80GiB服务、2GiBswap、
4GiB/零swap辅助域、CPU集合、独立watchdog、150/100GiB磁盘门槛不变。

验收：100/100原生输入/生成/adapter合同，真实远端收据、可靠原生时间线、
实际进程归属、四卡完整owner/原生worker/release证据。未知数值正确性不补造；
短前缀不作全4000性能或U_ref。终态后释放、停止本轮远端、保全、审计、备份，
再返回7B比较主线；不因结果高低复跑本键，不跳3B。
