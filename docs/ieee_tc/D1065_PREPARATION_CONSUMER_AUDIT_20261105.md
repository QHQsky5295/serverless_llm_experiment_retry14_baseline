# D1065：排除未触发的回退保护路径，保留空执行证据

2026-11-05。上轮 D1064 为 progress；本轮为既有日志/源码的有界分析，
新增 GPU 运行为 0，没有服务性能修改或新工作点。目标保持 active。
用户 dirty 的 EXECUTION_STATUS.md 保持原字节，本页作为增量交接。

## 用户确认的集群边界

现有 k3s 不属于本项目。复核既有
HYDRASERVE_ISOLATION_DECISION_20261012.md 及 D807/D808/D809：独立 rootless
Podman 适配和 3B W0/W1/W2 已完成，继续复用；不读取共享 kubeconfig、修改
权限、重启共享服务或重新部署。pp1 没有证明完整 pipeline/consolidation
性能收益，限制不变。不能因再次收到同一授权回到已完成的基线部署。

## 同一冻结事务的消费者核查

- ValidatedPreparationPlan 的成员查询已有 D163 快路径；initialized 执行
  已有 D741 消费视图，执行器展开一次后读取普通字典。未发现足以支持再次
  实现这两项的证据。
- 原生回退保护逐个 source_snapshot/hold 的循环确实存在，但是否构成本次
  开销必须看实际触发。D691 已说明 hold 本身不重做完整物理图；不能直接
  换成 identity-only 并声称检查等价。D514/D516 的另一个查询精简方向已有
  完整负结果，不能机械扩大到新调用点。
- 本次直接复用 D1047 的封存 main_outcome，并通过 D1064 的来源 SHA 绑定，
  逐项要求 native_fallbacks 显式存在、注册身份相符、结果/状态字段完整。
  不用缺字段默认 0，不从旧画像推断本次调用数。

| D1047 文件准备记录 | 数量 |
|---|---:|
| 唯一 plan_id | 2397 |
| completed / superseded / cancelled | 1579 / 816 / 2 |
| 原生回退保护 lease 记录 | 0 |
| 含多个原生回退保护的 plan | 0 |
| results 为空 | 1948 |
| completed 且 results 为空、注册文件目标为 0 | 1130 |
| completed 且 results 非空 | 449 |

结论：**不登记回退保护批处理或该路径视图精简候选**，它未在当前完整 Full
中触发，不能改善这项已观测开销。此结论不外推到文件替换被触发的其他场景。
1130 个无输出完成记录并不证明其正收益候选为空，也不证明先验可跳过规划；
更不能把 816 次 supersession 当作 816 次可免费删除的工作。

本轮核对 [vLLM 0.30 的消费视图](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/output.py)
和 [Python 3.12 asyncio 边界](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)。
它们支持区分不可变迭代内表示与实际状态观察，不证明本项目的加速幅度。
stable architecture 页面和另一博客地址获取失败，未作为论据。

## 产物、核验和主线

复用 D1064 file_rounds 的流式解析、来源校验、资源记录方式，仅替换统计对象。
新目录：paper_results/ieee_tc/p2_backend/d1065_preparation_consumer_audit/。
analyze.py SHA：fe398fd96ddeba1eec156a72dfb274b62c9de5e08ff5b2007cc98a85f267e0f1。
summary.json SHA：6047ed4d45dd1af54638bdd9e3374de87bd38acfda0a42c457d402c926f5682d。
主结果源 SHA：8a4bc4804fea75a291771fd529d3af76509f921a4d1fff52e2f76e79938aa46a。

分析使用 3/4 GiB、swap0、CPU2/3/26/27、CUDA 隐藏域；记录峰值 30519296 B，
high/max/OOM/swap 为 0，正常退出且 scope 已 inactive/dead。仅新增离线脚本与
资料，不需要重跑服务资格。初步只读计数首个 session 的末段输出未被保留，
重新读取同一封存日志取得计数；不是重复性能运行或筛选更好结果。

两模型外部比较和 3B A1 十一组继续复用。Full 的联合时延达成 95.900%，
但相对 ElasticOnly/ElasticLocality 的资源、尾延迟和达成率没有净优势；
不把完成消融当作贡献已验证，不填补未知 n_correct 或改为正式排名。

下一项回到**实际触发的**规划/执行边界：区分 1130 个无输出完成记录的
选择为空、容量不可行或执行时已满足，不先假定同一种原因。若现有日志不能
辨认，明确保留未知；只有可证明保留九式与实时所有权的具体候选才实施，
不再重做回退保护、D163/D741 或已否决的线程/缓存/查询候选。

完整待办不缩减：Full 净收益优化、7B 十一组 A1 与两个累积点、受控 A2–A5、
M2、S1–S13、CE/价格/Pareto、B1/B2/C1/C2/C3、公式和新颖性及两轮审稿文档、
IEEE 图表、最终同版本来源统一和实际 FaaSLoRA V2 非强推集成。
无运行中 GPU 实验，无新配置或论文修改，未删除/压缩数据。
