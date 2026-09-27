# D87：正式入口与测量入口配置统一、完整静态集合的类别覆盖

## 当前结论与主线

这次没有新增 GPU 实验、远程下载、权重或 trace。复用 D84 的真实子进程测量，
核查完整静态集合，不将旧的 source-only 时间重新标记成启用准入的 Full 时间。
Serverless 与其他 baseline 继续暂停；M1/M2、A1–A5、S1–S13 仍未开始。

| 既有测量覆盖检查 | 3B | 7B |
|---|---:|---:|
| 冻结静态集合中的逻辑 adapter | 500 | 500 |
| rank 8 / rank 16 | 350 / 150 | 350 / 150 |
| 完整文件树的不同内容类 | 24 | 6 |
| 配置下需要 / 实际观测的服务类 | 40 / 40 | 20 / 20 |
| 非 GPU 准备来源类 | 96 | 24 |
| 每个准备来源类最少观测数 | 3 | 3 |
| 缺失的内容—来源组合 / 服务类 | 0 / 0 | 0 / 0 |
| 原生 GPU slot 数 / 单 slot 字节 | 8 / 228130816 | 4 / 329056256 |

以上是 **D84 原运行身份下的类别覆盖**，不是新 Full 的性能资格，也不是 500
个独立训练 adapter 的证明。24/6 是包含元数据和其他文件的完整文件树类；
独立权重 SHA 数和零权重限制继续以 ARTIFACT_CONTENT_AUDIT 为准，不混用概念。
每类的三条加载来源观测不构成三次独立运行。

## 为什么需要统一配置

实际主入口原本会应用固定生成上限、scenario overrides，并规范化并发上限；
source collector 单独组装相近配置。D85/D86 已发现旧测量缺少准入初始化，
且 7B 的旧代理回执未显式记录实际采用的 CPU LoRA 容量默认值。

本次只抽取主入口既有的组装步骤为共享 helper：

- `_prepare_scenario_runtime_model_config`：生成合同、759/256 上限、既有 override
  优先级和实际并发约束；不新增调参规则。
- 重复规范化保留最初请求的并发上限；显式 scenario 并发 override 才建立新的
  requested 值。实际有效容量算法不变。
- `prepare_admission_source_runtime`：先在旧测量身份下推导完成长度初值，再加入
  Full 的准入初始化、经 SHA 核验的内容清单路径，通过上述同一 helper 和实际
  子进程 factory 形成下一候选配置。旧 spec 不含此 opt-in 字段时行为不变。
- D86 的初始化后 worker 配置回执及严格逐字段比较保留；不删除配置身份字段
  来让历史性能 profile 强行匹配。

真实 3B/7B 输入均已离线组装成功。W=5 秒仍只是从历史观测设置恢复的开发
候选；不是已冻结的共同 SLO，也不是调优后的最佳值。并发仍为 8/2，GPU
slot 仍为 8/4，CPU LoRA 容量仍为 32/24，movement limit 仍为 3。

这与后端的性能语义有关：并发、token budget 和 KV 压力可以改变服务时延，
不能把“相近配置”视为同一测量条件。这里只用官方文档支持配置必须一致的
判断，不从文档推断 Prime 的加速量。[vLLM 官方优化说明](https://docs.vllm.ai/en/latest/configuration/optimization/)

## 覆盖检查的边界

复用 `HttpArtifactStoreClient.routing_identity`：逐一读取原池中的小型 PEFT
配置，与冻结清单核验字节 SHA；不读取或复制完整权重，不访问远端。
复用 `ServiceClassBins` 和 `FrozenPreparationProfiles.source_class` 对实际
admission/source 快照分类；预热不用于补齐代表性覆盖，缺失类原样列出。

在当前空 footprint 分桶、content prompt ≤759、declared output ≤256、实际
并发 ≤8/2 的边界内，逐项枚举可达分桶。原生 tokenizer 可产生 760 tokens，
它属于完成长度估计的原生输入桶，不是这里的 canonical content 长度；两者
不能混为一个定义。非空 footprint 分桶需要另行验证实际分配域，不自动外推。

初始化布局由原生 `sources_before` 的完整实测快照通过既有
`native_activation_layout` 核验，包括 tensor views 和独立 allocation。
只继承几何布局的证据，不继承旧副本 readiness、地址、剩余容量或 epoch。
布局仍属于原始 D84 身份，尚未冒充新准入配置的实测布局。

单元检查包含：重复组装幂等、显式 override、旧生成路径、小文件 SHA 错误、
静态内容缺失、只有预热的来源、prompt 尾桶/并发尾桶、错误 rank/来源/计数、
未完成请求及相同内容却不同 rank。CPU 检查不替代真实 Full。

## 还缺什么，下一步只做什么

1. 同时明确开发用控制上/下限、窗口/cooldown、服务 bin width、EWMA beta；
   不把单元测试 fixture 常量当作经过选择的生产设置。
2. 使用这次已经统一的候选 runtime 对受影响原生准入/时间路径测量，保留
   D84 的旧身份；无需重复 D78 发布、D80 整池交付、D81 四并发检查。
3. 用匹配配置的实测证据导出严格绑定的初始化 profile，核验完整 Full 的
   activation、准备、routing/admission、生命周期与实际释放，之后才能进入
   开发回放和 baseline。Full 无条件资格拦截此时仍保留，没有配置绕过开关。

原始证据：`results/ieee_tc/p2_backend_qualification/d87_20260927/`。
机器可读摘要、日志与源码 SHA 见
`paper_results/ieee_tc/p2_backend/20260927_d87_main_source_configuration_coverage.json`。
本次应交付上述状态表，不绘制没有新性能测量的“性能改进图”。
