# D872：原生 HOST 准备的分配、提交和 Future 线程边界

2026-10-26，承接 D871 的唯一依赖问题。完整 PLAN 已重读，当前主线仍为
3B W2 达到共同联合 SLO 且保持生命周期资源目标；不是恢复基线适配或绘图。
本轮新增三个确定性 CPU 反例测试，并运行相邻已有回归，32 项通过、10 个
subtests 通过；19 条已有配置/弃用警告保留。未运行模型、未改变生产实现，
不新增工件/负载，不声称性能改善。

## 1. 实际验证结果

| 所检验的假设 | 实际见证 | 结论 |
|---|---|---|
| 已有 workspace 检查相当于后台预算预留 | 两次准备检查和一次 demand 检查都在同一实测占用上通过；owner 状态不变 | 检查没有创建在途 reservation，不能让异步任务直接复用该通过结果 |
| 整个 file→HOST 方法可原样让出 core | 在已有 NativeManager fixture 中插入一次合法的其他 adapter 加载/激活，原方法拒绝提交并 poison owner | 当前前后 cache/slot 不变量依赖串行执行；直接线程化并非语义等价 |
| 返回 Future 后，完成回调会回到 core | 执行安装版本原始 `_invoke_utility_method`：工作线程完成 Future，回调就在工作线程执行 | Future 只延迟回包，不提供原生状态提交的线程切换 |

第一项使用既有最小 fixture：预算 910 B、当前 310 B，每次加载保守峰值
300 B。两项准备与一项 demand 都未开始分配时可读到相同占用，三个检查
各自通过；若据此同时启动，潜在合计 1210 B。顺序重查占用 610 B 后，
准备正确被拒绝。这是**反对直接异步套用旧检查**的受控见证，不是声称
当前生产越界，更不是在机器上制造 OOM。当前方法仍串行执行。

第二项在 file loader 回调处插入一次其他 adapter 的 CPU 注册/GPU 激活，
实际得到 `native HOST preparation changed GPU or unrelated CPU residency`；
未生成成功 HOST lease 或完成回执。没有把错误吞掉或改为重试。

第三项直接从已安装 vLLM 0.30 的 core.py AST 提取原方法，与原生输入循环
测试沿用相同方式；只注入轻量结果容器，不初始化 CUDA。Future 先注册
回调再由独立线程完成，确认回调线程与 core 调用线程不同。它不是把本地
模拟调度器误称原生推理，也不测试加载速度。

## 2. 对候选的实际裁决

拒绝以下三种捷径：原方法整体 `to_thread`；仅凭两次容量检查通过就并发
分配；在任意 Future 完成回调中注册/激活 adapter。此拒绝不等于否定
CPU 准备与推理重叠。实际需要改变的是所有权与提交协议，而非再换线程 API。

下一单一候选限定为“有预算持有者的私有 CPU 构造＋原生 core 提交”：

1. core 在新鲜物理观察下为本次构造建立真实在途峰值预算；所有主动与
   demand 加载都能看到该持有量。不能把 projected 数值当 reservation。
2. CPU 构造只产生未发布的私有对象，不写 CPU LRU、GPU slot、source
   registry 或 ready 标志。复用官方解析、映射、packing/scaling 语义。
   保留每 owner 原有单构造顺序，不引入按测试点调整的并发参数。
3. 完成信号只唤醒 core；在 core 内重查 owner、内容、plan、当前 cache 与
   物理占用后提交。若 demand 已先完成同一来源，需处理私有对象的真实
   释放及本轮已发生费用，不把它算成首次准备命中或免费工作。
4. private 对象、临时 tensor、取消、失败和线程生命周期均留在原服务资源
   域；真实释放/提交后才结算 reservation。不能先放弃预算再等后台结束，
   也不能把尚未提交的对象提前用于 confirmed tier。
5. GPU 激活、KV/headroom 和 IEEE admission 决定继续在原生串行 core 上
   进行；本候选不拆开 Eq. (8) 的观察—决策—执行事务。

这五项是该候选自身必须实现的语义，不另建性能准入阈值。实现阶段应复用
现有 owner、movement 取消结算、native utility 与 D355 交错入口，不重建
实验框架。若原生加载包含无法隔离的共享状态，则拒绝并记录具体依赖。
本轮尚未接入上述协议，生产默认仍 D869/D870，不把测试通过写成异步化完成。

## 3. 历史排除与原始资料

D851–D854 的全文件 owner 线程化已经回退；本问题只涉及原生 checkpoint
到 CPU 对象构造，不能借此恢复整 owner 方案。D355 事务间交错、D747/D751
HOST 复用、D869 union 投影继续保留；不重开 GC、整图描述器、stat、路由
projection 或阈值调参。D329 禁止仅靠 epoch/model-ID 复用物理事实的反例
不变。D863/D868 已有控制 CPU 证据足够支持边界核查，不追加同类 profiling。

本轮使用 vLLM 技能核对固定后端实际执行边界，不采用通用性能数值或参数。
在线核对了 [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
的 CPU checkpoint 构造以及 [EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的原生 Future 回包行为；实际测试绑定本机安装文件 SHA，不以文档中的
“async engine”推断整个加载事务线程安全。D871 已记录 ServerlessLoRA 的
准备/执行干扰研究背景，本轮不移植它的 policy 或性能数字。

## 4. 资源、当前主结果与完整范围

测试在 CPU2/3/26/27、3/4GiB、swap0 的独立资源域运行，CUDA 不可见。
峰值 670244864 B，high/max/OOM/swap 均 0；无 GPU 进程，资源域已退出。
原始目录 results/ieee_tc/p2_backend_qualification/d872_20261026；小型
汇总 paper_results/ieee_tc/p2_backend/d872_native_host_boundary。原日志、
三个测试、执行源码 SHA、关闭/保护记录与脚本快照保留。没有新模型文件，
无需清理未确认用途的数据；后续重型实验仍重新检查磁盘和内存门槛。

当前完整结果仍 D870：3773/4000=94.325%，U11931.565094 GPU-s，预算
12083.063870；距联合目标 27 条。N_correct 仍未知，G1/G2 未完成；本轮
不计算 CI、不改写主结果。只验证三个具体边界，不证明完整异步加载正确性
或改善幅度。下一步是该单一候选的所有权协议实现与受影响验证，而不是
继续堆叠反例测试、重复短前缀或同键完整回放。

两模型同版本 W0/W1/W2、剩余 baseline/dLoRA、A1–A5、S1–S13、数值
证明、两轮审稿文档/图表及最终 V2 交付仍开放。HydraServe 复用隔离适配，
不操作非项目 k3s。目标 active，本轮为已执行边界验证的 progress。
