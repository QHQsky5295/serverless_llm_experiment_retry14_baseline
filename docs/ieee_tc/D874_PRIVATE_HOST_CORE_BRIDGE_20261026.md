# D874：私有 CPU 准备已接入原生完成与关闭路径

2026-10-26。承接 D873 的同一个候选，不是新的优化方向。完整计划 SHA 为
`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
本轮完成 core/worker/frontend 的连接及 CPU 验证；尚无新增 GPU 性能结果。

## 本轮实现

- 新 opt-in `ieee_host_construction=private_cpu_core_commit_v1`。只有已验证的
  vLLM 0.30、dense TP/PP/DP=1、UniProc、IEEE owner、LoRA、workspace 及原生
  control-interleave 组合可启用。已有 YAML 和默认串行路径不改。
- 前端只把文件→HOST 主动准备送到 core utility；其它来源观察、需求获取、
  HOST→GPU 激活、KV/admission 仍用原来的原子事务。九公式不变。
- core 在原线程持有预算并开始一个私有 CPU 构造；native Future 保持未完成，
  原生循环可继续推理。后台线程只做构造并向已有输入队列发送本地完成消息，
  不提交 cache、不发布 ready、不执行 GPU step，也不在后台完成 utility 回包。
- 完成消息在 core FIFO 中结算：join 已结束构造，复用 D873 的新鲜计划、来源、
  cache 和物理预算检查，提交或丢弃，然后完成原生 Future。已有公平交错只在
  完整事务之间让出一次原生循环；没有新增时间阈值或另一个模型调度器。
- 同一在途命令重传共享原 Future，身份变更拒绝；其它构造按已有单 owner
  约束明确延期。等待者不能取消 Future 从而提前返还预算。
- native shutdown 在销毁 worker 前 join 私有构造，并按取消路径结算；实际
  等待仍包含在生命周期。正常 WAKEUP 保留原语义，已结算的完成消息不重复
  发布。构造或结算失败保留 owner invalidated，不伪造释放/ready 或回退结果。

在途峰值仍是 D873 的保守上界，可能与 allocator 已见字节重叠；这是保留的
容量代价，不称互斥实际占用。线程仍共享 GIL，也没有消除总 CPU 工作。
因此本实现只建立合法重叠机会，不能由结构推断净加速或 SLO 提升。

## 依据和历史排除

本轮使用 vLLM 技能核对实际版本和生命周期，不采用其通用性能阈值/示例
配置。官方 [EngineCore 0.30](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
支持返回 Future 的 utility，但回调在完成 Future 的线程执行；故不能直接让
构造线程完成含 cache 提交的整个事务。官方
[UniProcExecutor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
的单 worker 调用是同进程同步调用，私有对象只在此边界内传递，不跨 RPC
序列化到客户端。关闭先 join，再执行原 shutdown；未替换 native busy loop。

D871 当前完整日志和 D872 边界见证已足够，本轮不重做 profiling 或 HOST
组件微测。未恢复 D851–D854 整个文件 owner 线程化，未叠加 graph、GC、stat、
跨 epoch 缓存、控制阈值或远端配置变化。D747/D751/D869 仍保留。

## 实际验证及限制

复用 D873 的受限测试入口，仅增加本候选及相邻原生事务检查：最终
**232 项、102 subtests 通过，19 条已有警告**。新增 9 项 core/frontend 测试，
另扩展原配置测试。真实 reference owner、worker bridge、已安装原生 input
与 utility 方法一起验证；CPU fixture 不初始化模型或 CUDA。

覆盖推理步骤能在私有构造期间推进、无提前 ready、FIFO、core 线程完成
回包、重复命令/取消、idle wake、shutdown join/discard、构造失败、线程启动
失败、未启用时原路径以及错误配置拒绝。相邻 GPU/KV 原子路径和 metadata
回归通过。首次 231 passed/1 failed 是新测试给原生四参数函数传了第五个
参数；测试改为原生所需零参数 callable 后，最终源码全通过。首次记录保留，
没有为通过此项改生产语义，也没有重复 GPU 运行。

最终 CPU scope 峰值 720080896 B；high/max/OOM/swap 事件为零，CPU 集合
2/3/26/27，3/4GiB、swap0。无 GPU 请求、无远端/共享 k3s 操作。旧 147 项
结果与用户 manifest 的零变化以 closure 回执为准。执行原始目录为
`results/ieee_tc/p2_backend_qualification/d874_20261026`，curated 为
`paper_results/ieee_tc/p2_backend/d874_private_host_core_bridge`。

**下一步是受影响的真实原生混合推理资格，随后一次新键完整 3B W2。**
复用既有资格/runner，不增加另一次 100/1000 请求 Full 前缀或整套无关资格。
实际 allocator 并发、GPU slot/adapter、生成与取消/关闭要有原生证据；CPU
测试不能代替它们，也不是新的独立数值 LoRA 证明。新 YAML 启用前保持默认
串行；不得用新路径未触发的运行宣布候选有效。

## 当前论文证据和完整范围

当前完整结果仍为 D870：3773/4000=94.325%，U11931.565094 GPU-s；共同
预算12083.063870，距离95%差27条。没有新性能收益、无CI，N_correct仍未知，
G1/G2未完成。只做实现与验证进展，不画图、不改变评价规则。
同版本双模型W0/W1/W2及G1/G2、剩余baseline/dLoRA、A1–A5、S1–S13、独立
数值证明、两轮审稿图文及最终V2交付全部开放。HydraServe隔离适配沿用，不
接管非本项目k3s。按明确文件备份可回退检查点，最终V2同步仍待完成。
