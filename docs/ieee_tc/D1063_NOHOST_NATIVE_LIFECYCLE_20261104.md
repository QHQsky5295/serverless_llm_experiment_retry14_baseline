# D1063：NoHOST 原生 CPU 副本退役边界

2026-11-04，接续 `D1063_NOHOST_CONSUMER_PRICING_20261104.md`。
本页是当前增量交接；用户已修改的 `EXECUTION_STATUS.md` 保持不动。
3B A1 的性能进度仍为 **10/11**，NoHOST 尚未启用，没有新增 GPU 性能回放。

## 本轮落实的范围

在已有 `configure_host_budget` 安装点增加严格布尔复用策略。Full 缺省不变；
NoHOST 必须在空 native owner、第一次加载之前安装，之后不能切换。保持正的
张量预算、workspace、GPU/NVMe 容量；不将 HOST 预算置零或转给其他层。
native 物理快照保留真实 CPU 行和 footprint，仅附加不可独立复用的政策。

CPU-only 副本只在完整操作边界退役，保护集合包括：

- 所有 GPU-active/slot backing、CPU/GPU pin、GPU 和 HOST 引用；
- 所有尚未关闭的 plan targets，包括 pending 已为空的完成兄弟目标；
- staging lease 与未完成 settlement 的私有构造目标，线程 done 不等于已结算；
- 身份未知的 native 对象继续作为未知状态报告，不删掉来制造合格状态。

回收复用原 owner 的 remove/fence 路径。GPU replacement 先完成相同 victim、
slot 和暂存 backing 的原有检查，最外层才清理。snapshot、refresh、native
remove callback 均不主动回收。删除对象不等于 allocator 已归还物理字节，
仍由原实际分配观测计算容量；新增计数仅称退役 CPU copy 数，不称释放字节。

成功 demand/proactive commit、最后引用释放、plan close、已经 join 的 private
construction settlement 才进入对应收尾。冲突、soft defer 不进行额外驱逐。
回收后的 epoch/source snapshot 同步到返回回执及其幂等缓存；transport retry
不重复回收，也不改写第一次成功回执。Full 不增添 NoHOST wire 字段。

## 为什么采用此边界

继续使用 D1049/D1063 已核实的原生依赖，不重试“HOST 容量设零”等已否决方向。
本轮在线核对官方 vLLM v0.30.0 的
[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
和 [worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
registered 与 active cache 关联，删除 registered 会触发失活；因此保留执行必需
backing、只移除独立复用是合适的可实施消融，而不是零 HOST 内存实验。
本机仍有已资格化补丁，不声称源码与上游逐字节相同。

`ablation-planner` 既有 reviewer 做了 provisional 只读复核，重点发现并修正：
外层回收后的回执陈旧、private construction 回执缓存顺序、失败操作不应触发
回收、成功 transport retry 不能再次收尾。该同系列复核不是独立外部审稿。

## 测试暴露的历史问题

原始记录目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261104/native_lifecycle/`。

1. `native_tests1`：131 项，22 个错误（包含子测试）。新增生命周期及直接
   native-owner 测试通过；旧 `OwnedNativePreparationPlans` fixture 未提供当前
   runner 所需的 `residency_manager.local_source_references` 容器，造成错误及
   等待超时。补齐 inert 容器，不伪造文件 lease 成功，不放宽原断言。
2. `native_tests2`：134 项，9 个失败，集中在已有异常回执注入测试。真实问题是
   runner 在验证 plan/hash/lease/source 绑定之前，就将回执视为 outcome known。
   因此异常回执可能进入正常 fallback release。将确认标记移至完整绑定验证
   之后；正常成功路径和策略不变，异常路径保留未确认 ownership。
3. `native_tests3`：**135 项通过，0 失败/错误**，源 SHA 前后一致，耗时
   40.977 秒。包含使用真实文件引用的异常回执测试，不能只靠无文件 fixture
   证明 fallback 未被提前释放。内存峰值 1,258,573,824 bytes；high/max/OOM/
   kill 和 swap 均零。原始 JSON 是结果与源码身份的权威回执。

上述失败全部保留，不删除或改成成功；没有模型性能失败，也没有据此宣称收益。
测试继续使用 CUDA 不可见、CPU 2/3/26/27、memory high/max 3/4 GiB、swap 0，
沿用 D1063 的已有测试启动和源 SHA 前后核验，不重建实验框架。

## 尚不能运行 NoHOST：下一步仍在同一主线

本轮只完成 native 生命周期部分；以下工作不能省略：

1. 精确授权 staging→GPU：demand 不能偷偷复用另一请求/计划的 CPU-only 副本；
   `hold_host_source` 不能作为任意 HOST 复用入口。保留原计划/lease 的正常完成。
2. 文件 owner 同步冻结政策，禁止独立 HOST 最终驻留；保持必要在途文件和预算。
3. NoHOST 所有 GPU 替换都收集实际 NVMe/remote fallback，而非仅 staged 情况；
   按 D1063 已完成计价执行，验证同 victim/slot 和真实后继来源。
4. 文件替换零损失来源需实际 GPU lease；不能仅改旧 CPU fallback 的 tier 标签。
5. 全调用链读回一致后，再增加既定 NoHOST 配置与一个 3B W1 回放。不通过手动
   改快照字段或只启用本轮 native 参数冒充完整消融。

没有新图、远端操作或共享 k3s 修改。HydraServe 复用已完成的隔离适配，不重复
部署其他用途集群。历史结果、两项用户 dirty 文件及旧投稿保护清单继续保持。
完整剩余范围仍包括：7B A1/累积节点、受控 A2–A5、M2、S1–S13、CE/价格/Pareto、
B1/B2/C1/C2/C3、公式/创新性/两轮审稿文档、IEEE 图表、统一 canonical 来源，
以及实际 FaaSLoRA V2 分支的非强制集成。当前 Full 的负面消融证据继续保留。

备份产物位于 `paper_results/ieee_tc/ablation/d1063_nohost_native_lifecycle/`。
备份脚本检查源 SHA、两项用户 dirty、147 项旧投稿文件、staged allowlist 和
秘密信息；先提交，再推送 origin V2 与 FaaSLoRA 独立备份分支，禁止强推。
实际完成身份见原始目录 `backup.json`，不能据此称已整合分叉的实际 FaaSLoRA V2。
