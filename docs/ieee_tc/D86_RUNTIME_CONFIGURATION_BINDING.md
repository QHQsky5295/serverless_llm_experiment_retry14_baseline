# D86：同一观测窗口与初始化后配置回执

日期：2026-09-27；修改基于 `e7b48fe0b3b2198a8e5808132763f9f4b88c9d07`。
本轮为 P1/P2 的 Full 进入条件校验，不是性能实验或独立审查。
没有新 GPU 运行、真实远端请求、工件生成、负载生成或 baseline 修改。

## 当前研究问题与历史证据

进入 Full 之前必须保证：需求频率与已完成生成长度使用论文中的同一个 W，
加载压力的分母对应实际共享传输容量，初始化 profile 对应实际运行配置。
否则即使各模块单独通过，也不能声称整条路径遵循同一论文定义。

D84 已结束的两模型五来源测量保留，不重新运行。D85 已从它们的原生成功
终态恢复了实际完成长度。本轮发现两项具体缺口，并修复测量/配置绑定：

1. 需求窗口、原生完成窗口和 movement 并发分别配置，没有在 Full 构造入口
   强制核对。旧自动推导与显式配置可能产生不同窗口；不能把它当成同一个 W。
2. D84 子进程 proxy 的 `model_config` 是父侧解析后发送的配置，**不是完整的
   初始化后 worker 回执**。其中 7B 没写 `max_cpu_loras`，而当时实际
   `InferenceEngine.initialize` 使用 `max(max_loras,24)`。3B 显式为32。
   原代码 `488a716` 可直接核验这条规则。D85 文档中把 D84 字典笼统称作
   “实际 child 配置”的表述应按此限定；并非新增发现7B实际用了另一容量。

## 实现与论文语义的对应

| 论文/测量要求 | 当前实现证据 |
|---|---|
| 两处 W 一致 | Full 的真实 `ScenarioRunner.__init__` 核对显式 demand W、runner W、实际 HotnessTracker W、可选 preload W 和 native initializer W；不一致拒绝，不修改任一值替它过关 |
| 加载压力分母对应执行上限 | initializer transfer limit、显式 `max_concurrent_loads` 和实际 `OwnedMovementQueue.max_concurrent` 必须同为相同正整数 |
| 不改变九个公式 | 只检查配置身份，不改 EWMA、需求统计、路由、planner 或 admission 算式；汇总记录绑定回执 |
| 计划配置与已初始化配置一致 | 父 factory 与初始化过程共享既有 CPU LoRA 容量解析；worker 初始化后回传自己的字典，父侧逐字段比较，缺回执/不一致停止并收回自己的启动进程 |
| 不读取半份回执 | worker 先写完整临时文档，再原子发布 ready 文件；地址和配置属于同一回执 |
| 不削弱 profile 身份 | `FrozenServiceProfiles`/`FrozenPreparationProfiles` 的精确校验不变；不删除 capacity、admission 或其他配置字段 |
| 尽量复用已完成观测 | 仅在完成长度初始化证据中显式解析有源码依据的历史 CPU 容量默认值，保存原记录与解析后配置；实际容量不同仍拒绝，不重标旧 D/T/O |

新增 `faaslora/runtime_configuration.py` 是共享的纯配置函数，不启动后端、网络
或另建实验框架。24 是已有 facade 规则，不是新调参值，也不是 vLLM 上游
默认值。后续按模型冻结显式容量仍可覆盖它。

“初始化后配置回执”仅指本项目 `InferenceEngine.model_cfg`；不能声称它列出
vLLM 内部全部隐式设置、内核选择或资源状态。这些仍由既有运行环境及原生观测
保存。真实 GPU 路径中的新回执尚未在本轮运行，不能把模拟后端测试写成 GPU 资格。

## 验证与即时结果表

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 真实 Full 构造入口，实际 demand/movement owner | 通过 | 一致时记录，窗口不一致时拒绝；不偷偷启动队列 |
| worker/RPC 原生事件与配置回传 | 通过 | 模拟后端初始化后的变化进入回执，而不是重用发送前字典 |
| 父侧错误容量回执 | 通过 | 拒绝实例并终止本次启动进程；不会进入可路由池 |
| 100KB 回执的原子发布 | 通过 | 发布前无 ready，发布后为完整 JSON，无半份文档 |
| 相关回归及基础 smoke | 418项通过，34.973s | 包含既有288项基础检查，不是418次推理实验 |
| OS/证据入口 | 60项通过，0.857s | 既有资源/进程保护未放宽 |
| demand/request/admission/preparation/pressure | 331项通过，7.336s | 需求、预约、准备和压力所有权回归 |
| 最终 launch/basic smoke | 349项通过，33.726s | 最后调整只写容量字段，保持其他嵌套配置对象不变；再次核验 |
| D84真实原始文件的离线长度复用 | 3B/7B均通过 | SHA和原生完成检查保留；不新增GPU运行 |

具体日志、原始来源及代码 SHA 在
`paper_results/ieee_tc/p2_backend/20260927_d86_runtime_configuration_binding.json`。
其余 demand/request/admission/preparation/pressure 检查也在该清单中。
测试用表格交付，不制作虚假的性能曲线。早期测试日志保留；最终测试重新覆盖了
原子发布改动，未覆盖旧测试日志。

## 原始资料核查

[vLLM 0.30 LoRA 配置源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py)
把 `max_cpu_loras` 定义为 CPU 可存储 LoRA 数，要求不小于 `max_loras`；未设置时
上游补为 `max_loras`。因此不能混淆上游默认值与本项目显式传入的历史24容量。
这里引用源码用于核验配置含义，不由此推断性能收益。

## 下一主线：仍不启动半配置 Full

本轮解决了窗口/分母检查和 worker 配置回执，**尚未完成生产配置冻结**。
候选 W5s 仍只是历史配置导出的开发起点，不是新共同 SLO，也不宣称最优。
接下来一起处理：

1. Full 实际主入口还执行 `_normalize_runtime_concurrency_cap`，并在真实 HTTP
   场景要求 `artifact_content_manifest_path`；D84 源配置不包含同一完整字段集。
   新初始化/校准配置必须经过与 Full 相同的解析，不能靠删除这些身份字段接入。
2. 同时确定并记录 EWMA beta、service bin、IEEE 扩缩容阈值/窗口/冷却、实际
   共享 HOST/NVMe/native HOST 预算；不拿测试 fixture 数值冒充已选配置。
3. 核对完整500集合可出现的 service/preparation 类覆盖；缺观测类不填零、
   不借邻近类。已有长度和实际 footprint 继续复用。
4. 最终候选一次组装后才测受影响的原生路径，再进入 Full 真实多次激活、
   pending、准备、请求和 GPU 生命周期验证。Full 无条件保护仍保留，不能凭
   本轮测试通过直接移除或设置绕过开关。

保持 Prime 优先、baseline 暂停。M1/M2/A1–A5/S1–S13 未开始，没有新性能排名。
不重复已完成的 D78发布、D80全池下载、D81并发交付或 D83/D84校准；需要补测
的配置差异先统一核对，避免逐元数据字段重复启动模型。
