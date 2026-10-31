# D995：已到达请求的文件准备与推理容量分离

2026-10-31。当前只推进3B W2联合SLO缺口；不是重启baseline资格，也不修改
共享k3s。HydraServe已完成的隔离适配、D593/D810外部比较和所有失败保留。
PLAN SHA为d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。

## 假设、历史与来源

D992为3678/4000联合时延通过（91.95%）、物理U8268.639918 GPU-s，仍差
122条达到既定95%。D994从原始数据恢复503条自身Remote准备，其中135条
TTFT违约的平均准备为1895.226ms；这是顺序包络，不是可扣除的纯CPU时间。
源码的文件下载发生于全局dispatch permit和所选副本active_requests占用之后。
因此准备请求会占住尚未用于推理的容量，可能影响其他已经具备可执行LoRA的
请求。此次只改变这一依赖，检验整体联合SLO、尾延迟及实际GPU时间。

历史D313分离的是计量区间；D831（beb6a1bb）允许已核验文件在pressure结束
确认前消费，但仍在取得服务容量之后。两者没有完成本次的到达阶段分离。
D840解码器、D851–855线程包装、D844/D856/D874–886清点微优化不重开；
D888–912/D958文件状态进程已有实现，不包装成新候选。详见D994历史核查。

本轮核对的原始资料：

- [HydraServe §5.1](https://www.usenix.org/system/files/nsdi26-lou.pdf)：根据
  实际依赖重叠准备与运行时初始化。这里只借鉴消除无必要串行依赖，不引用其
  多节点收益作为本机结论，也不据此跳过文件/原生引用安全。
- [vLLM 0.30 worker_manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  原生adapter读取要求实际本地文件；保留其加载、校验与激活边界，不修改kernel。

## 单项候选与不变量

可选配置为`ieee_arrival_file_preparation: observed_remote_demand_v1`，默认关闭。
对实际已经到达、当前没有原生存在提示或已发布本地源的请求，先经过既有
OwnedMovementQueue执行Remote→NVMe，再进入原有dispatch permit、confirmed
routing、KV/active-LoRA admission及原生加载。其他请求可在下载期间继续服务。
这不是同时为同一请求预留多个副本，不是提前得知未来热点，也不承诺能消除
该请求本身的文件等待。使用过期的原生存在提示最多跳过这项可选准备，最终
决策仍需权威快照；提示不能证明GPU-ready。

- 论文九公式、同快照排序、EWMA、预算与原生可执行引用均不变。
- 实际请求到达后才提交文件任务；外部open-loop、原有截止时间及到达时钟不变。
- 文件队列仍合并相同目标/content，使用同一物理空间预算和传输并发，不新增常数。
- 已运行原生成员仍收到共享传输pressure；取消须等待实际writer结束再返还预算。
- 尚未选择副本的任务明确记录`target_replica=None, placement_phase=before_routing`，
  只允许NVMe demand；不能把未分配的HOST/GPU/handoff任务冒充合法选择。
- 需求兴趣从排队覆盖到后续请求结束，保护发布文件与原生读取之间的交接；
  这可能增加保留时间/竞争，必须计入实际副作用，不能声称无成本。
- 请求输出新增`gpu_reference_evidence.arrival_file_preparation`，保存观察、开始、
  完成、复用/存在提示和movement回执。dispatch tier仍描述实际路由时的层级；
  不能因早期准备使它变为NVMe，就抹去此前真实远端传输或声称原始到达即命中。
- 早期准备属于完整TTFT/E2E中的路由前等待，GPU物理生命周期照常积分；
  不把它作为该请求旧的admission后Remote→GPU成本样本、不做时间相减。

采用同一D992配置，只增加此策略及唯一输出/NVMe/HOST路径；保留预测控制、
图捕获、token预算、并发8、最多3实例、原生0.30后端、共同60秒通知和真实
已发布gzip。未改SLO、75%预算、原生token合同、工件及trace。
新配置SHA：74829d658bf5ed74390b486e9efef2c4239a90fc69ada56715958cc457356f54。

## 验证与一次完整回放

针对变化的请求生命周期检查82项及既有OwnedFileMovement6项通过；另补默认
关闭/非法策略和无所选engine但已有原生成员的pressure检查。初次测试中的
证据字段名误写、在IO线程读取原生线程专属计数器已修正测试，不改变生产
语义或放宽约束；两次失败记录保留于本说明。环境19条既有警告不隐藏。
没有改变kernel/模型执行边界，复用已完成的后端和数值诊断资格，不额外跑
重复100请求smoke。当前全部4000请求独立数值正确性仍为未知，不伪造n_correct。

复用原run_d992入口，仅增加显式配置覆盖；现对已存在运行目录立即拒绝，
防止在preflight之前截断旧launcher.log。执行命令：

```bash
FAASLORA_D992_RUN_ID=d995_20261031 \
FAASLORA_D992_RUN_TAG=d995_3b_arrival_preparation_full_w2 \
FAASLORA_D992_CONFIG=configs/ieee_tc/20261031_d995_3b_arrival_file_preparation.yaml \
bash scripts/ieee_tc/run_d992_3b_busy_service_rate_validation.sh
```

运行身份是已知trace上的单次开发候选，W2、4000请求，不称独立留出测试。
原始目录为`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d995_20261031/3b_busy_service_rate_validation`。
runner实际HOST路径使用`/dev/shm/tc-d995_3b_arrival_preparation_full_w2`，与旧点隔离。
原有启动门槛、72/80GiB服务域、3/4GiB外置域、watchdog与释放记录继续使用。
开始观察磁盘约162GiB、MemAvailable约104GiB，四GPU空闲；最终以启动回执为准。

完整结束后先释放资源、验证输入/计时/原生绑定、保存，按同一冻结V1/单轮V2
重算联合SLO、U和尾部。完整数据出炉前不称改善或达标；也不能通过改变阈值
或挑选重复获胜。若无改善且无新证据，关闭本候选而不是盲目改并发/解包参数。
后续消融共享该非目标需求实现，仅按目标机制开关区分，防止多因素混淆。

同配置两模型W0/W1/W2、G1/G2、A1–A5、S1–S13、CE/Pareto、两轮审稿文档、
最终图表及V2交付全部仍在主线。未达标前不画图；保护旧结果和用户manifest。
备份不force远端既有分歧，明确记录真实分支回执。
