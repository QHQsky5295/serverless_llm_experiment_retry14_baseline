# D873：私有 CPU 构造与原生提交的所有权实现

2026-10-26。承接 D871 原因分析与 D872 边界见证，本轮开始实现同一个候选，
不是第三轮仅做 profiling。完整 PLAN SHA 仍为
`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。

## 已实现与尚未接入的边界

在现有原生 reference owner 上实现三个阶段：

1. `begin_file_host_construction` 在原生线程复用源/预算/计划检查，并持有
   一个实测工件合同的峰值上界；每个 owner 至多一个私有构造。已有 HOST/
   staging 命中仍走原串行路径，不重复构造。
2. `NativeHostConstruction` 只在线程中调用官方 checkpoint 解析及私有
   packing/scaling/pinning，结果保存在私有任务对象中；不写 CPU LRU、GPU
   slot、source registry 或 ready。线程不返回带权重的 Future，也不让错误
   traceback 持有临时 tensor；异常文本保留。唤醒回调不得提交原生状态。
3. `finish_file_host_construction` 只允许原 owner 线程在构造结束并 join 后
   提交。重新检查当前计划、cache、来源和物理占用；空槽可注册，满缓存须
   有现有计划支持 staging，不隐式 LRU 驱逐。需求请求抢先完成时丢弃私有
   副本并记录实际发生的构造；不会将其记成免费命中或重标记另一合法路径。

取消不能提前结算在途内存。已关闭计划不发布结果，正常丢弃后重新观察真实
占用；提交不确定/原生错误保持 owner invalidated 和任务记录，不假称回滚。
旧串行准备共用原验证逻辑，GPU 激活与 Eq.(8) 的原生串行事务没有拆开。

**本轮还未把上述 API 接入 EngineCore 的 utility/Future/WAKEUP 路径。**
没有修改 runner 配置或启用异步模式；当前正式/开发回放默认仍走原串行
调用。新 API 在确定性线程测试中执行，不等于 vLLM 原生混合推理资格通过。
下一步只做同一候选的 core 完成通知、取消/关闭结算及最小受影响原生验证，
继而新版本完整回放；不重复 D872 反例或再起同类原因 profiling。

## 容量语义及明确代价

所有新分配检查都能看到在途峰值持有量，包括需求加载。持有量取已有工件
合同的 `peak_additional_tensor_bytes`，不是新的人为余量或按模型名称常数。
没有私有构造时数值为零，不增加物理清点或缓存实际状态。

构造过程中，部分 pinned 字节可能已经计入进程 allocator 观察。因此
`accounted + in_flight_peak_upper + incoming_peak` 是**保守上界**，不能称
互斥物理字节之和。没有用猜测重叠去抵扣，也没有给未来释放信用；这一保守
做法可能暂时延后 demand，必须在后续完整回放观察，不预先断言一定改善。
完成后只在同一个串行结算中解除预留，并将私有对象纳入新鲜实际观察或
先释放私有引用再观察；不把 object 引用删除等同 allocator 字节已释放。
HOST 总 RAM、Python 对象/page cache 仍由原服务 cgroup 包络约束。

## 实际验证

第一版 170 项通过后，补齐“空 cache 也检查计划身份”和“需求已使用另一
等价确认路径”的边界；最终源码运行 **172 项、68 subtests 通过**，保留
19 条已有警告。两份记录都保留，不覆盖第一版测试结果。新增 11 项分离
所有权测试；另扩展原 worker CPU fixture，验证私有回调调用原 loader 和
packing、但不做 inventory、注册或 GPU 激活。其他为相邻 reference、预算、
计划、取消、原生控制交错回归。

具体覆盖预留可见、单构造顺序、无提前 ready、当前容量变更、GPU 状态变化、
staging、需求抢先完成、取消在途任务、计划关闭/身份、路径绑定、原线程
提交、重复提交与失败。fixture 不执行 CUDA，不把它当真实 pinned allocator
并发、数值正确性或尾延迟验证。

测试使用既有 D872 受限入口的小幅参数化复用，没有新建实验框架。最终资源
峰值 674369536 B，high/max/OOM/swap 事件为零；CPU2/3/26/27、3/4GiB、
swap0、CUDA 不可见。无新 GPU 请求、无远端操作、无非项目 k3s 操作。
原始目录 `results/ieee_tc/p2_backend_qualification/d873_20261026`；小型
产物 `paper_results/ieee_tc/p2_backend/d873_private_host_ownership`。

## 历史排除与原始实现依据

核对 D871/D872 和实际修改历史后，未恢复 D851–D854 的整个文件 owner
线程化；未加入 D864 整图描述器、GC、stat/projection 或新的阈值。D747/
D751/D869 和既有 admission、movement 结算继续保留。

采用 vLLM 技能核对实际版本边界，不使用技能示例中的通用性能数字/配置。
[vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
将 CPU 对象构造与注册分离；
[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
的 dense packing/scaling 修改新对象，cache 注册是另外一步。
[EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的 Future 回包并不替代原生线程提交，故本轮不直接把整个 worker 方法线程化。
[ServerlessLoRA](https://arxiv.org/html/2505.14468v1) 的制品准备/推理争用问题
只提供研究背景，不证明本候选线程安全或必然加速，也不移植其性能数字。

## 主结果及未完成范围

当前完整 3B W2 仍为 D870：3773/4000=94.325%，U11931.565094 GPU-s，
预算12083.063870，距联合达标差27条。没有新性能结果、无 CI；独立数值
正确性仍未知，N_correct=null，G1/G2 未闭口。
两模型同版本 W0/W1/W2、剩余 baseline/dLoRA、A1–A5、S1–S13、数值证明、
两轮审稿图文和最终 V2 仍开放。最终 V2 同步不等于本轮备份分支；不 force
push、不 stage 用户 manifest，不改旧结果。当前为实现进展，目标保持 active。
