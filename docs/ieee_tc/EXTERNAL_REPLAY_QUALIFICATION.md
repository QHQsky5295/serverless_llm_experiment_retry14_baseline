# 固定到达与独立回放：测量资格，不是性能结果

## 问题、历史实现与依据

旧 `ScenarioRunner._run_continuous_observed` 在服务 event loop 中按时间创建请求。
同一循环执行控制和 dispatch，阻塞可能使注入变慢；旧 `run()` 还在模型、预热、
最小实例池准备后才定义回放起点。那不是计划要求的固定部署通知加 60 秒。

同一审计发现：`_derive_trace_scale_down_floor_s` 读取整条到达序列，
`_scale_up_ready_candidate_queue` 根据预测 ready 时间预读未来请求及 adapter。
新 TC 路径不允许这两种未来信息。历史路径和原始结果保留身份，没有改数字。

对照 vLLM 0.30.0 的 [serve.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/benchmarks/serve.py)
中 `get_request` 和 `create_task(limited_request_func(...))`：到达计划与请求执行
需要分离。其[官方参数说明](https://docs.vllm.ai/en/latest/cli/bench/serve/#--max-concurrency)
也指出并发约束可能使实际执行请求率低于设定值。本项目直接复用原时间戳，
不重新生成同平均速率的 Poisson 过程；进一步把产生进程与服务资源域分开。

可证伪假设：接收端阻塞时，外部进程仍按原到达产生请求；产生、提交、接收之间
的滞后均可观察且不能从 TTFT/E2E 中扣除。若外部进程也停止产生、起点被重置、
请求丢失或时钟不一致，则资格失败。

## 现有入口的扩展

- 复用 `workload_generator.py`，增加只读 `FrozenReplayPlan` 和独立 transport。
  不创建新 adapter、trace 或平行实验 runner。
- 原 `run_all_experiments_user_scope.sh` 通过
  `FAASLORA_TC_EXTERNAL_REPLAY=1` 开启；必须是受保护 launcher、一个 scenario、
  一个 run 和完整 shared trace，不支持旧 quick/按 drain 分 phase 的隐式重置。
- 外部 publisher 与 supervisor/watchdog 共用最多 4 GiB 辅助域。被测服务仍在
  72/80 GiB、swap=2 GiB 服务域；实际 publisher PID、birth、cgroup、affinity 留证。
- 原 launcher 在模型 exec 前确定 deployment notice，正式 `t0=notice+60`；
  publisher 不等待模型 ready。接收晚或 transport 背压产生的延迟照实保留。
- timer 只把到期请求放入有限 trace 队列，发送任务独立；不等前一请求的结果。
  全部冻结 request JSON 通过私有本地 Unix socket 传输，不只发 index。
- 每请求保存计划到达、产生、提交开始、socket drain 和服务接收。成功写 socket
  不等于服务已经接收；`N_submitted` 与 `N_arrived` 分开。
- 原模型 runner 以收到的请求启动既有 run_one/admission；在线 backlog、RPS、
  handoff 候选只看已收到的请求。原文件仅作输入核对，不能作为未来需求 oracle。
- 固定源 SHA、view SHA、每请求内容 SHA、request ID 与 Linux boot/time namespace
  核对。clock helper 由原 native token 计量复用，跨机器不直接相减。
- 收发失败不退回原 timer，保留完整计划分母、已产生/提交计数。异常取消时 join
  已启动任务，不能把异常消失后补出的空结果当作合法运行。
- W0 保留原到达；W1 仅按批准的 8×500、半速时间、30 秒间隔公式建立索引视图。
  W2 必须先取得冻结 adapter map，本入口目前拒绝未经资格的 W2 变换。

## 无 GPU 实际检查与状态表

使用旧 7B seed42 trace（28,120,629 字节、4,000 请求）的前 32 请求，8 倍速率
只用于微型资格。该检查准备窗口 0.2 秒明确不是正式 60 秒，也不是独立测试 seed。
接收进程在首请求后同步暂停 1.5 秒；外部进程处于另一资源域。

| 检查 | 结果 | 解释 |
|---|---|---|
| 第一次，原空进程 witness 的 64/128 MiB 限制 | 失败并保留 | 原 trace JSON 读取持续触发 high；停止该拥有身份的检查进程，无 GPU/host OOM |
| 第二次，专用 192/256 MiB replay witness | 32/32 收到，进程释放 | 原文件读取峰值 119,590,912 bytes；high/max/OOM 均零 |
| 接收端暂停期间外部产生/提交 | `req_00001` 仍产生且提交 | 前两次接收间隔 1.50210 秒，没有让接收阻塞冻结外部 timer |
| 请求产生滞后 | 均值 1.16245 ms，最大 2.52727 ms | 单次测量资格，不是注入能力的普遍上界 |
| 接收相对计划到达的最大滞后 | 217.36497 ms | 暂停后承担的剩余等待保留，未扣除 |
| 字段、取消、W1、无未来信息、dispatcher 错误 | 确定性测试通过 | 详见对应测试；不代替实际模型性能 |

原始日志在 `paper_results/ieee_tc/safety/external_replay_attempt{1,2}.launch/`，
curated 回执和 `external_replay_summary.json` 保存 SHA。第一次失败不覆盖。
改变的是微型检查容量，不是任何系统的共同性能包络；本节用状态表，不画性能图。

## 仍未证明的部分

1. 当前接收在原 runner 的服务阶段开始；初始化/预热期间的延迟保留为连接/提交
   等待。后续须将入口接收与启动并行并计入服务资源域，明确客户端等待与服务队列，
   不能把当前阶段分解冒充完成的启动/队列归因。
2. HTTP 基线须复用同到达/内容/时间合同并独立做端点注入资格。本地 IPC 不等于
   HTTP 客户端开销；需要分别报告 transport，确认它不是排名驱动因素。
3. 实际 native 模型 cgroup/clock/GPU census、物理占用积分、控制器资源 owner、
   正式 timeout 和结果汇总资格仍未完成。`production_launch_authorized=false`。
4. 没有运行 4,000 请求模型性能比较、M1/M2、消融或 sensitivity；不宣称胜出。

后续返回 P1/P2 主线：启动期间真实服务接收、原生 worker 与资源观测、原子
admission/reservation，再做 Serverless 资格和公平对照，不扩大此微测矩阵。
