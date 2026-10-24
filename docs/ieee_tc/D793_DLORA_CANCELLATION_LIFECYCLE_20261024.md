# D793：dLoRA 取消与迁移交接的必要正确性修复

2026-10-24；CPU 验证，不是新的性能结果。主线仍为完成 3B 基线后对照
PrimeLoRA，再完成同版本 Full、消融、机制和敏感性。不修改论文、负载、模型
权重、共同阈值或基线调度策略。当前 Kubernetes 属于其他项目，继续不触碰。

## 证据与有界问题

D792 完整 W2 有 4 次迁移数量断言，其缺失响应流的请求此前均已取消。
原路径先取走 SequenceGroup、释放原 KV，再移交响应流；因此不能删除断言
或捕获异常后继续，把失配掩盖成正常低性能。D791 W1 未出现同项断言，但有
598 个超时，使用同一有缺陷的取消路径，不能凭无断言认为完全不受影响。

本次结合 D775 当前 owner 取消测试与 cf421793 之前历史；旧测试仅检查响应流
终止，没有检查原生调度队列。重读固定官方 commit
75f1c439446fe194b1df8a24982ef9067841fab5，未把本地 wrapper 当成官方算法。
使用 research-lookup 的原始来源核查原则；此前默认查询服务缺认证，本次
沿用可用网页工具直接读取固定源码，不安装服务、不上传私有材料：

- [官方 Scheduler](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/core/scheduler.py)：批量取消一边遍历一边删除，同队列相邻项会被跳过；取消没有更新请求计数。
- [官方 AsyncLLMEngine](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/async_llm_engine.py)：先移除 tracker，再经可能暂停的新增请求 RPC，最后发送后端取消。
- [官方 EngineManager](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/engine_manager.py)：迁移持有 select_lock，但上述交接原先没有共用该锁。

两个原生方法 CPU 测试在修复前实际重现上述问题，均以保留缺陷行为的断言
记录；修复后测试仍保留为来源对照。没有证明这解释全部 778 次 W2 超时，
也没有把最后 3 个无原生终态的请求直接归因给这 4 次异常。

## 改动与不变项

仅在现有 source-view 工厂增加 opt-in `cancellation_lifecycle=True`：

1. 取消遍历队列快照，完整处理相邻请求；只对确有未结束序列的组减一次
   req_model_cnt。正常完成后的重复取消不再减计数，也不重复释放 KV。
2. tracker 提取、原生入队和后端取消的交接共用 manager 原有 select_lock。
   原生 GPU step 调用在锁外；取消 RPC 本身仍可能等待 actor 正在执行的工作，
   不宣称锁等待完全不含后端等待。
3. 取消按当前响应流 owner 定位，在同锁内等待后端取消确认再允许迁移。
   tracker 后续重复取消为幂等操作；不广播取消、不吞异常。

取消修复限定共同 canonical n=best_of=1 合同。默认 source-view 行为不变，
旧 vendor 与旧结果不编辑。控制操作串行化可能影响时延，必须由新完整运行
实测，不称为零开销修复。迁移求解器、迁移断言、调度/加载策略、batch、GPU
数量、1,800 秒保护均不改；没有 PrimeLoRA 策略移植或基线性能参数搜索。

## 验证与证据边界

`run-experiment` 复用现有 unittest/source-view 入口，CPU 2,3,26,27，
MemoryHigh/Max=3/4 GiB、swap=0，不启动 GPU 推理。结果：

| 验证 | 结果 |
|---|---|
| 原生缺陷重现 | 2/2，通过意味着成功重现缺陷，不是原实现正确 |
| 取消专项 | 9/9，含四队列、重复/已完成、入队并发、迁移后 owner、RPC 失败传播 |
| 既有完整 dLoRA CPU 测试模块 | 55/55；3.387 秒 |
| 迁移函数 | AST 与原路径一致；断言保留 |
| 未变的模型/加载/观测源码 | 与 D788 source-view SHA 相等，复用 D790 有界数值证据 |
| 旧投稿及用户修改 | 147 项 SHA 保持一致 |

本轮不证明 Ray/GPU 的完整性能、全部请求数值正确或所有超时均被消除。
没有增加短 GPU smoke：模型与计算路径未变，新的 CPU 交接测试已覆盖所改路径，
下一直接执行必要的完整工作负载，不重复旧资格或增加无新问题的检查。

## 受影响执行键与下一步

D791 W1、D792 W2 均保留原始失败与结果，不覆盖、不改统计数字；作为修复前
诊断证据，不用其异常低达成率论证 PrimeLoRA 优势。二者均发生超时取消，且
请求计数参与后续分配，因此 W1 即使无同项断言也不能与修复后 W2 拼成同配置。
下一冻结同一修复版本，各执行一次完整 W1/W2；仅源码交接修复改变，原输入、
DP2/TP1/PP1、migration_type=3、500 工件和所有资源/生成配置保持不变。

如果新结果仍低、但没有具体正确性/适配异常，以阶段和机制证据解释后结束
dLoRA 适配，不进行参数寻优。随后依次处理 Loquetier/HydraServe 3B 的必要
兼容性及完整场景；其余所有已批准实验仍保留。

原始记录：`results/ieee_tc/dlora_qualification/d793_20261024/`。
本次 checkpoint 使用 github-sync 的显式路径、SHA、diff 和凭据检查；
BASE 推送 origin/main，MAIN 因既有 V2 分叉使用具名 backup 分支，不能宣称
最终 V2 已同步。实际版本与远端核验以随后 backup_receipt 为准。
