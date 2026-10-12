# D467：固定设备身份的 CPU 前端准备边界

2026-10-12，7B 优化主线。**CPU 准备边界通过实际环境检查；尚未实现提前
准备与扩容衔接，不是新的性能结果，不代表 PrimeLoRA 已达成 G1/G2。**

## 问题、历史与本次新增证据

D465 最新完整 W0 为 3790/4000 联合达标（94.75%），未达到共同的 3800
门槛；生命周期 11984.148429 GPU-s，没有优于先前可行 D458。D466 已复用
完整日志定位到少 ready 副本时的真实容量积压；没有证据支持补一个唤醒轮询。
主线继续优化 PrimeLoRA，不修改已封存基线、SLO、输入或 IEEE 九个公式。

本次问题是：**固定真实 GPU 身份后，现有前端能否先完成纯 CPU 的准备，
而将新的 GPU 引擎创建留到实际扩容时？**这不是从旧启动时间中扣除导入耗时。

已核对：

- D337：实际 activation 平均约 47 秒，嵌套的加载/编译日志不能相加归因。
- D338：单纯删除早期导入没有消除完成相同 frontend 导入的成本，不重复该对照。
- D339：CUDA 隐藏条件下已有两个 Humming reaper；无额外线程的 fork 前提不成立，
  fork 候选已拒绝。D467 不再提出 fork，也不把这些线程当作新发现。
- D340/D341：关闭可选后台预构建的完整回放没有形成全程资源/时延优势。
  本次沿用 D465 原环境，不重新试这个开关。
- D463：live owner projection 仍在源码中，但 D465 不支持其性能收益。
  D467 没有回滚、扩展该修改或宣称它已被性能验收接受。

新证据相对 D338/D339 的区别：不是隐藏 CUDA 的导入计时，而是**真实 UUID
固定可见时**，用驱动当前/primary context、原生 platform 映射和独立 NVML
观察核查实际 pre-initialize 边界。没有测量 CPU 准备的端到端性能收益。

## 官方实现约束

1. vLLM 0.30 的
   [设备能力查询](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/platforms/cuda.py)
   经 NVML 获取信息，`get_device_capability(device_id)` 具有缓存；缓存键并不
   包含可见设备 mask。另一个 device-count 缓存则显式包含 mask。因此不能
   仅凭 torch 尚未初始化，就认定任意“隐藏 GPU 导入后再换 mask”都安全。
   本候选固定 UUID，不清空或篡改第三方缓存来掩盖设备身份变化。
2. [CoreEngineProcManager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/utils.py)
   将 core 启动与 frontend 分开；当前生产路径为原生 `spawn`。
3. [core_client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)
   对后台 warmup 和 core 启动有明确顺序。
   [多进程设计说明](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/design/multiprocessing.md)
   提醒线程依赖与 fork 风险；该说明中的部分历史策略写于 2024 年，不能替代
   对当前安装源码的核查。

四份官方资料已下载，URL、取得时间、SHA 和本机相关源码随证据归档。网页工具
当时返回 429；改为直接读取公开官方原始文件，没有依据未取得的页面作结论。
`serving-llms-vllm` 技能用于核查后端边界，其通用阈值不替代冻结评价协议。

## 实际验证

只扩展既有 `ieee_tc_preflight.py` 的诊断入口 `backend-frontend-check`，并增加
13 项拒绝/边界单元测试；没有修改请求路径、worker、分配器或 autoscaler。
复用既有 `gated_launch`、完整 D465 模型配置和 native 环境，不新建另一套 runner。

- Python native 环境保留 torch 2.13.0+cu130 / vLLM 0.30.0。
- 在 fresh interpreter 启动前固定 NVML index 0 对应 UUID，整个过程不换 mask。
- 保留原 `sitecustomize`、allocator 策略及前端类；原生 core 方法仍为 spawn。
- 观察脚本只初始化 CUDA **driver**，不 retain/create context；使用只读的
  `cuCtxGetCurrent` 和 `cuDevicePrimaryCtxGetState`，不用会触发初始化的
  `torch.cuda.current_device` 或 runtime `cudaGetDevice` 代替。
- 实际导入 runner、IEEE frontend、dedicated worker，并只构造未初始化
  `InferenceEngine` 描述符；不调用 initialize/shutdown/generate，不创建
  AsyncLLM/core，不读取权重、制品、trace 或访问远端。

| 检查点 | torch CUDA initialized | current context | primary active |
|---|---:|---:|---:|
| 原启动钩子之后 | false | 0 | 0 |
| runner 导入后 | false | 0 | 0 |
| native frontend/worker 导入后 | false | 0 | 0 |
| 未初始化描述符构造后 | false | 0 | 0 |
| 原生固定设备映射核查后 | false | 0 | 0 |

原生 platform 检查确认相同 UUID、一个可见设备及 SM86。各检查点保留
MainThread 和两个已有 Humming reaper，OS task 数为 4；**不允许据此 fork**。

独立 watchdog 共 22 个资源样本、24 次 GPU census，均无本实验 GPU context。
观测服务内存峰值 **1,488,506,880 B**，主机可用最低 110,324,412,416 B；
high/max/OOM/swap、保护告警均为 0。服务、watchdog 正常退出，服务路径移除，
两个辅助 scope 按 captured InvocationID、inode 和实际递归空进程列表关闭。

CPU 回归涉及 272 个不同测试。首轮在 LLM conda Python 中 268 项通过，4 项
系统保护测试因该解释器缺少 `signal.pidfd_send_signal` 失败；没有改生产保护
或测试断言。仅将这四项放到既定 `/usr/bin/python3` 后全部通过，实际守护启动
也使用该系统解释器。初次失败、日志和关闭回执全部保留，不称首轮全通过。

### 限制

- 五个内部检查点和采样 census 不能排除任意未观测的瞬态；第一内部检查点
  晚于原启动钩子。不存在绝对“永不创建上下文”的证明。
- 单次资源峰值包含此诊断和导入子进程，不直接乘四作为未来准备池的精确预算。
- 没有在服务中实际复用 prepared frontend，也没有验证其取消、绑定、GPU
  分配和 engine-ready 顺序。此次通过不是模型或性能资格。
- 不从历史 TTFT/GPU-s 中减掉导入时长，不把未测的 CPU 准备收益写入论文。

## 决定与唯一下一步

边界可行，进入**有界、固定设备身份的前端提前准备与 fresh core 衔接**实现，
不再重复 CPU 导入微测。下一版本须满足：

1. 只在共同部署通知后准备，全部 CPU/HOST 工作留在服务资源域，不使用未来
   请求/热点。数量由既有合法设备/实例上限约束，不建立无界后台进程池。
2. CPU-prepared 与 engine-ready、GPU-ready 分开；未创建引擎不能进入 ready
   容量、路由或 admission。UUID 不跨进程生命周期改变。
3. 真实扩容仍新建 native core，保持 spawn；不复用已持卡的旧进程冒称免费
   CPU 准备，不用 sleep-mode 代替释放。
4. 在实际排他持卡时开始 GPU 计量。若扩容触发时 CPU 尚未准备完而设备已被
   独占，等待也必须计入；不能仅把 allocation 时间戳后移来制造资源降低。
5. 现有 `PhysicalGPUAllocation.bind_process` 记录 `worker_spawn`，不应将提前
   出生的 CPU 进程伪报为此刻新生。明确记录出生、准备、绑定与 acquire，校验
   owner、UUID、取消、重复 claim、失败和实际退出，再放行 GPU 初始化。
6. D463 保留为当前父版本的一部分，并明确“没有接受其性能收益”；新候选只
   检验准备时序，不捆绑回滚、阈值、容量或后台构建开关。
7. 通过生命周期测试和最小 native 资格后，才用新执行键进行一次原 4000 W0；
   以既定联合 SLO、GPU-s 判断，不靠准备阶段的局部改进结案。

当前阶段没有跳到 3B。7B W1/W2/G2、HydraServe 独立环境资格、3B、A1–A5、
S1–S13 和补充文档/后期图表全部仍待完成。现有 k3s 不作任何改动。

产物：`paper_results/ieee_tc/p2_backend/d467_cpu_frontend/`；原始目录
`results/ieee_tc/p2_backend_qualification/d467_20261012/`。结果 SHA
`b3a6949d6287423f4dc6f3b6b3bbeac17757017e0f5f23ac0df9670544c67391`。
147 项历史保护核验通过；远端服务未操作，唯一权重/trace/工件和用户修改不覆盖。
