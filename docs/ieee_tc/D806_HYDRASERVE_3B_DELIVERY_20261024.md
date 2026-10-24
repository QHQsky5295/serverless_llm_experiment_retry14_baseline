# D806：HydraServe 3B 实际权重交付修正验证

本次是针对D805具体接收错误的单次正确性检查，不是主性能实验。复用已有
隔离rootless Podman平台，不改用户明确属于其他项目的k3s。原生控制器、
CPU provider、private SHM快速加载及GPU worker仍实际执行。

## 结果与证据边界

| 检查 | D805（保留失败） | D806 |
|---|---:|---:|
| 原trace前100请求终态/响应 | 100/100 | 100/100 |
| 原生固定长度与adapter绑定 | 100通过 | 100通过 |
| 全零输出序列 | 100 | 0 |
| req_00001前三token | 0, 0, 0 | 578, 2326, 1288 |
| 同输入独立参考前三token | 578, 2326, 1288 | 578, 2326, 1288 |
| 真实远端获取次数 | 29 | 29 |
| 线上传输字节 | 67,350,156 | 67,350,156 |
| 请求内打包 | 0 | 0 |

参考请求的477个prompt token IDs及adapter ID逐项相同。D803已有原生
Worker与独立HF/PEFT参考的局部数值验证，本次通过原CPU provider/共享内存
交付后该参考前缀一致，且不再出现D805的接收越界异常。相同100前缀上的
全零异常消失支持此次具体修正有效，不推广为所有输出token独立数值证明。
`n_correct=null`、`independent_full_correctness_proven=false`继续保留；
不把100请求检查、HTTP成功或原生绑定证明包装成4000请求全量数值证明。

唯一服务路径变化是已经备份、37项CPU测试通过的`bounded_shm_receive=True`：
按真实剩余分区限长接收，提前EOF失败，父进程核验接收子进程成功后才发布
完整进度。原共享池6GiB、batch8、30秒idle、1800秒请求保护与调度策略不变。
不增加任意padding、不更换LoRA权重、trace或后端，不做baseline性能搜索。
此前D804上下文与D805输出失败均保留，不追改为成功。

此次没有展示pipeline/consolidation收益；本机PP1交付检查不能声称验证了
未触发的跨GPU流水线机制。完整W0/W1/W2及其机制事件仍待执行。

## 测量与收尾

service与watchdog均returncode0；实际服务域递归为空并消失，GPU无残留。
服务峰值30,736,326,656B；high/max/OOM/oom_kill/swap均0。检查实际占用
242.9372435840778 GPU-s，仅作为此短检查的资源账，不外推完整性能。
prepare/remote_prelaunch/audit辅助域均精确关闭，audit峰值120,033,280B。

collector在冻结前改为实际3B clock_id与3B传输日志；停止时核验原InvocationID，
三个自有远端服务inactive/PID0、端口消失，传输/监测日志双边SHA一致。
允许无请求失败没有transfer journal，不伪造记录。本轮确有29条传输记录。
D80通知前一次性交付缓存继续复用，无请求内制作工件。

启动前已完成D805重复缓存安全回收：232个文件共1,170,025,980内容字节与
保留原池重哈希一致；回收1,170,915,328B。原失败/原始测量/transfer元数据
完整保留，私有失效view链接映射保存；147项旧投稿文件与用户dirty未变。
引用核查记录三项不可读同UID进程入口，未冒称完整系统级扫描。

## 来源与下一步

- RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d806_20261024`。
- 实际执行BASE：`e1c433bab982366a82ad188ef703270d89d6ea06`。
- 实际执行MAIN：`bef1572ae3f6d194ed6830637da07f23ec862424`，任务状态文档dirty记录保留。
- prepare SHA：`c6bcfa2e3a888f92427aece4231b5b19903ce08383b957ddf3890773f0c8b480`。
- source manifest SHA：`d3bcf4887dd2d89b95f6f8bf67c9c8d0eed4d2454f237a14536d7beae016879a`。
- launch SHA：`250c9289e005871609bd8b0e74feecc854914102930421cf4f74569b7ca0ba1c`。
- delivery audit SHA：`c46c1c9109d44fcf46a19e84b12afdbc5834a9ac02b4e704fc56d2f7fc7ed1cc`。

使用run-experiment/monitor-experiment复用既有单轮入口及实际进程核查；
analyze-results将输入绑定、局部数值证据与性能主张分开，github-sync保存
明确路径的检查点。没有绘图或新增统计显著性门槛。

下一完成检查点和必要重复缓存回收，直接运行HydraServe 3B完整W0/W1/W2，
不追加1000请求检查。随后全基线比较、PrimeLoRA有依据优化、同版本Full与
A1–A5/S1–S13、成本/CE/Pareto、审稿文档/图表/最终V2均保持完整主线。
