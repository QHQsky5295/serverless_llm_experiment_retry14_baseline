# D329：HOST 物理表示的依赖核查（尚未选择实现）

2026-10-08。主线仍是 7B。D328 诊断完成，当前仅进行了代码与历史核查；
没有新 GPU 回放、生产修改或性能收益结论。不得重复 D328 来等待更好结果。

## 已排除的捷径

1. D180/D181 已完成请求范围 footprint，D189 已复用静态文件描述，
   D193/D194 已精简规划传递，D202 已省去无消费者的 HOST view 描述。
   每个 view 的重复 pinning 查询也已删除，不重复这些候选。
2. 不能直接不测 pinning。`NativeSourceSnapshot._footprints` 明确把
   pinned / unpinned / mixed_pinning 编入 HOST representation；
   `confirmed_source_class`、规划准备成本、fallback/replacement 消费它。
   它不仅是诊断标签，也影响 Eq. (2) 的合法观测类别。
3. 不能按 owner epoch 或 Python model ID 直接复用整个物理快照。
   当前 owner 的 `_remember_source_object` 跟踪 model incarnation；这不等于
   对同一对象内部的 tensor replacement、storage resize、pin registration
   或外部 mutation 的不可变性证明。whole-snapshot TTL 也不是候选。

## 核对的原生事实

- [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  在原生单线程 core loop 中加载、移除、注册并激活 adapter。load-inplace
  和原生 LRU 仍可能改变对象集合；不能只监听 Prime 主动准备入口。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
  对 CPU tensor 的 `is_pinned` 查询底层 storage pointer 的 accelerator
  属性。它不是纯 Python 字段读取；这为研究重复原生查询提供依据，
  但不证明本机每次调用耗时或一定可以缓存。
- 本地 `gpu_monitor.py` 当前仍每次遍历真实模型、验证 A/B、storage
  capacity、alias、dtype 和 pinning；`residency_manager.py` 的 removal
  回调失效确认状态，staging/registration 另有生命周期。

这些依据只支持研究依赖，不支持从文献借用加速百分比。

## 下一项唯一研究及否定条件

考察：能否把不变的物理表示事实绑定到**实际受控的 storage 生命周期**，
保留每次 live membership、source incarnation、GPU confirmation、引用和
容量重检查，仅免去能够证明冗余的 native 属性查询。

首先沿现有 owner、native manager 及已装后端检查全部对象/存储变更入口，
包括原生按需加载、load-inplace、LRU、staging、取消、replacement、共享
storage 和失效。复用现有 tiny tensor fixture，不创建 adapter/trace 池。
若无法证明失效覆盖，或必须跳过现有错误检测才能获益，就拒绝该候选，
转向 D328 中已记录的查询外部等待；不添加兜底重试或降低安全检查。

不得把“同 pointer”视为跨生命周期身份，缓存不能延长驱逐后权重驻留；
不能把每 adapter 的子图独占字节误用为全局可回收容量。初始化/维护开销
必须由实际服务承担，不移入免费外置工作。若候选成立，先正确性测试，
然后普通完整 4,000 请求回放，不能由局部微测关闭 G1/G2。

此文是承接 D328 的只读审计记录，不是又一套实验计划或已合格实现。
共同 warm、Resident、数值正确性及新旧 Prime 新指标验收仍开放。
7B 后再推进 3B、基线、主比较、消融与敏感性。目标前不绘图。
