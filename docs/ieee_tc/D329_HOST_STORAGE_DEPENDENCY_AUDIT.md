# D329：HOST 物理表示的依赖核查（CPU 反例验证完成）

2026-10-08。主线仍是 7B。D328 诊断完成，本轮进行了代码核查和 tiny CPU 验证；
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

## 反例验证与裁决

复用现有 NativeManager/NativeHostFootprint 微型 fixture 和真实 CPU tensor，
6 项新增测试加已有 HOST footprint/projection 回归共 32 项通过。
环境为既有 Torch 2.8.0+cu128，CUDA 未初始化；不是 vLLM 0.30/Torch 2.13
的真实 GPU 性能或工作负载发生频率证据。

| 受控改变 | owner 身份不变时，实时 inventory 的结果 |
|---|---|
| 替换内部 tensor | 总字节 768→1152，dtype 变为混合 |
| 同一 tensor 的 storage resize | 总字节 768→1280 |
| 两个私有 tensor 改为已有共享 view | 总字节 768→512，独占回收量也改变 |
| mock pinning 查询 false→true | allocation 的 pinning 标志改变 |
| 增加嵌套 tensor 字段 | 原有非法表示检查拒绝 |
| 移除注册对象 | epoch 推进，联合/独占字节重新计算；空集为零 |

pinning 用的是受控返回值，不是真实 cudaHostRegister；前五项不证明普通
运行确实发生这种修改，只证明现有 epoch/model-ID 合同本身不足以排除它。
因此拒绝**仅依赖这两个字段**的跨调用物理表示缓存，不将它加入生产代码。
这不等于否定将来具备完整受控 storage 生命周期的设计。

原始日志在 `results/ieee_tc/p2_backend_qualification/d329_20261008/`；
curated witnesses/cleanup 在 `paper_results/ieee_tc/p2_backend/d329_host_dependencies/`。
32 项连同导入耗时 6.326936 s；实际 3/4 GiB、swap0、CPU2,3,26,27，
峰值 496173056 B，high/max/OOM/swap 全零，精确身份核对后空域已关闭。
保护清单的唯一差异仍是既有 baseline replay 脚本，不称 147 项全部未变。

## 下一项边界

考察：能否把不变的物理表示事实绑定到**实际受控的 storage 生命周期**，
保留每次 live membership、source incarnation、GPU confirmation、引用和
容量重检查，仅免去能够证明冗余的 native 属性查询。

若以后恢复该方向，须沿 owner、native manager 及已装后端检查全部变更入口，
包括原生按需加载、load-inplace、LRU、staging、取消、replacement、共享
storage 和失效。复用现有 tiny tensor fixture，不创建 adapter/trace 池。
若无法证明失效覆盖，或必须跳过现有错误检测才能获益，就拒绝该候选，
转向 D328 中已记录的查询外部等待；不添加兜底重试或降低安全检查。

不得把“同 pointer”视为跨生命周期身份，缓存不能延长驱逐后权重驻留；
不能把每 adapter 的子图独占字节误用为全局可回收容量。初始化/维护开销
必须由实际服务承担，不移入免费外置工作。若候选成立，先正确性测试，
然后普通完整 4,000 请求回放，不能由局部微测关闭 G1/G2。

当前转向复用已经取得的真实测量，而非推定旧物理事实永远有效：
D289 的加载身份事件、D301 的同 epoch scope 保留、D316 的 GPU-copy 保留
已存在，不能重复实现。下一候选核查 initialized planner 取得的完整新测量
能否同时供现有 router 使用，保持原始 epoch/时间戳和选中来源原子复核；
不能增加新 RPC、重用旧预算决定或把历史测量重标为新测量。

此文是承接 D328 的依赖验证记录，不是性能资格或新一套实验计划。
共同 warm、Resident、数值正确性及新旧 Prime 新指标验收仍开放。
7B 后再推进 3B、基线、主比较、消融与敏感性。目标前不绘图。
