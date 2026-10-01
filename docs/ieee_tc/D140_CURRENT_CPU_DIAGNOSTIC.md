# D140：当前版本的控制器与推理进程 CPU 诊断

2026-10-01。13:36:31 启动一次，13:43 前失败退出。未重跑，无性能改善结论。

## 问题与依据

D139 从两模型完整记录恢复了秒级回复获取等待，而通道申请均值小于 0.1 ms。
优先验证当前控制器的 CPU 工作是否延迟回复处理；同时观察 native core，
避免把前端、规划进程或 GPU core 的工作混在一起。现有日志没有逐类
source_snapshot/preparation/retirement RPC 时长，旧 D123 父进程 profile
又早于多项优化，不能代替本项测量。

依据 [Python asyncio 官方说明](https://docs.python.org/3.12/library/asyncio-dev.html)
中的同步 CPU 阻塞机制，以及 [py-spy 0.4.2 官方文档](https://github.com/benfred/py-spy/blob/v0.4.2/README.md)
的 subprocess/GIL 采样功能。这里只确定可证伪问题，不预先断定 JSON、
存储图遍历或任何一个函数是主因。

## 不改变的条件

- 当前已备份代码 `78ce501fee27e2ca517423140e498fc1f12f0cac`；serving 与 D137/D138 相同。
- D137 的 7B Full 配置，仅更换三处独占输出/工作目录，并采用 D122 已验证的
  前 1,000 请求索引视图；原 4,000 请求源、工件池、到达和生成内容不重建。
- cap=2、seq=2、LoRA slots=4、CPU LoRA=24、FP16/TP1、四卡上限、
  准备时间 profile、真实远端发布对象、1,800 秒保护和论文九式均不变。
- service 72/80 GiB、swap≤2 GiB、CPU 4–23/28–47；外置回放/监控
  3/4 GiB、swap=0、CPU 2/3/26/27。磁盘 150/100 GiB 门槛保持。
- 只运行一个重型诊断；不在推理期间修改远端配置、打包、校验整池或清理。

## 新增观测与解释规则

复用 D123 的 profiler wrapper，仅加入 `--subprocesses`；保留 100 Hz、
GIL-only、threads、完整源码路径和 speedscope 输出。模型子进程继续使用
真实 vLLM Python，不递归启动 profiler。不采集 locals，不更改系统 ptrace。

逐 PID/角色报告父控制器、dedicated frontend、planner 与实际 native core。
用运行时进程/worker 证据核对角色覆盖；缺采样就报告缺失，不填零。
保留采样错误、延迟告警、启动/import、在线处理、保存结果的不同调用来源。
inclusive 栈比例可重叠，GIL 样本不包含释放 GIL 的原生执行，也不是墙钟归因。
profiler 可能扰动调度，所以本次 TTFT/TPOT 只作诊断，不进入 Full 主表或宣称加速。

结束后按原顺序：资源释放→完整诊断 population/生成/时间核验→角色热点表→
选择或否定一个候选优化→最小正确性验证→普通 4,000 请求 Full。
3B TPOT 退化、输出 hash 变化、两模型历史性能差距仍未解决；不因此转入 baseline。

## 路径与启动记录

`results/ieee_tc/p2_backend_qualification/d140_20261001/` 保存复用脚本、
新鲜配置、预检、profile 和完整失败证据。实际 PID、scope、remote invocation
与进度在 `EXECUTION_STATUS.md` 记录；尚未观测到的身份不预填。
预检 192 项来源、147 项历史保护、资源门槛通过；两端网口为 1 Gbps/full。
profile 已出现启动期进程退出/栈读取警告，将保留并核对角色覆盖，不能忽略。

## 失败状态表与解释

以下来自唯一同次运行；按 analyze-results / academic-plotting 和计划第十一节，
不完整诊断使用状态表，不生成性能排名或置信区间。

| 检查 | 本次观察 | 含义 |
|---|---|---|
| 工作量 | 原始 4,000 请求的前 1,000 索引；90 条 native-contract 完成，72 条 interrupted | 未到达请求不补造 timeout；不是完整资格 |
| GPU 占用 | 观察到 1,041.300741 GPU-s；4 份租约均释放 | 截断占用不作低成本优值，完整 GPU-s 为 N/A |
| 准备任务 | 1 个 residency epoch、其 file/GPU 准备各记录 failed | 同一传播错误链，不是 3 次独立系统失败 |
| 退出错误 | `preparation source identity/coverage contradicts its frozen objective` | 准备执行前的一致性检查失败，不是 OOM |
| 采样质量 | 162 次滞后告警，最大 46.84 秒，最终 speedscope 文件缺失 | 不能回答当前 CPU 热点，更不能当普通性能点 |
| 内存保护 | 384 个样本，服务峰值 20,096,225,280 B；主机最少可用 92,685,787,136 B | high/max/OOM/swap/保护告警均为 0 |
| 远端 | 31 个 UUID 配对、52,228,851 B，两端一致；31 个 published | 全部复用 D78；动态打包和临时归档为 0 |
| 收尾 | GPU 上下文消失、服务资源域移除、工作目录清理完成 | 13:44:50 后关闭本轮远端及空辅助域，没有推理期间远端调整 |

主进程先抛出上述 ValueError；外部回放随后失去接收端，launcher 表层记录
`protocol_or_launcher_error` / `external replay failed`。保留这两个原始字段，
不能仅凭 launcher 标签把本次归为纯外部干扰。代理尚未发送人工中止信号；
最后的 -15 为既有监督器收尾，不能写成代理主动停止成功。

失败 GPU plan 为 `38c1dab7ec3b49eabbe19edb37912c6c`，冻结 hash 为
`7a07c99348c20189300fa03efa88c014a8fe0a1351494114ab424633e70ea9b4`。
它包含三个 target（880770、904661、170491），注册 epoch 29，关闭 epoch 195。
唯一 GPU attempt 是 880770，停在 `observing`，尚无 prepare RPC 提交证据。
既有调用路径只处理“当前执行 target 的来源变化”和“非 target 来源变化”；
如果另一个选中 target 的来源不同，后者会将其作为矛盾拒绝。这是待验证分支，
**不是已经复原的现场原因**：失败瞬间 complete source snapshot 未保存，
不能由异常文本唯一判定是哪一条断言、哪一份来源触发。

该问题须区别于 D109 的非目标重载，以及 D134 的退役后新计划选源。
联网复核的 [vLLM 0.30.0 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
按整数 ID 管理 native 对象；这不保证跨 RPC 的整体准备计划原子。
它支持检查对象/文件/计划的生命周期边界，但不能代替本地内容与所有权证据。

## 下一步与尚未完成的主线

先复用 CPU owner/planner 测试，构造“同一计划多个 target，其中另一个 target
具有不同的已确认文件来源”的可重复反例；明确是合法的预先选源、后来失效，
还是实际身份损坏，再决定修正。禁止删除校验、吞掉异常或盲目重试。
目前未改任何 serving 文件、配置、容量或九个公式。

采样诊断独立保留为不合格；不得原样重复 100 Hz 全进程树采样。
后续若需要采样，先验证有界、较低开销及异常时可落盘的观测方式。
CPU 热点、3B TPOT/输出 hash 差异、7B/3B 完整性能仍开放；baseline 继续暂停。
共同 warm/Resident、M1/M2、核心消融和 S1–S13 尚未完成。
