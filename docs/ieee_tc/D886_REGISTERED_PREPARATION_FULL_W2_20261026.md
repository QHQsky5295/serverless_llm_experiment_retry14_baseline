# D886：同次已注册准备共享物理图的完整 3B W2 检验

2026-10-26，启动前登记。只验证D885已实现的限定候选，不增加生产机制。
问题是：在保持真实内存、容量与IEEE九公式语义的情况下，减少同一次已注册
HOST→GPU准备事务的重复物理图清点，能否提高完整churn回放的联合SLO，
同时保持生命周期GPU时间预算。资格通过不预先等同为系统收益。

## 冻结条件与参照

- 直接父检查点5f197a2bfe2f757f713ef0eec46d6883e7fa737e；实际运行commit
  由启动回执记录。D885的328项检查及两项扩展边界检查直接复用，不重新
  进行组件计时、短Full、profiling或重新生成profile。
- 直接比较D883：3753/4000联合时延达标（93.825%），物理GPU时间
  11934.81358408602 GPU-s，P95 TTFT 2153.8861524 ms。D870的
  3773/4000（94.325%）、11931.565094108693 GPU-s仅作较早参照。
  不将胜过D883等同为达到目标，也不隐藏相对更早版本的退化。
- 共同预算12083.063870235463 GPU-s，联合要求至少3800/4000；共同
  D648 SLO与D657 Resident参考不变，不采用在线5000ms作为最终评价阈值。
- 与D883配置仅output、NVMe和HOST三个独占路径不同；同一3B、4000请求、
  500逻辑adapter、seed42源trace、W2 rotation100映射、min1/max3、D725
  profile、固定greedy输出、D747 descriptor及isal解压。真实远端复用
  D78/D80已发布gzip缓存，无请求中打包，无本地模拟或计时扣除。
- D885只在已注册、无staging/私有CPU构造且原生线程/方法/回收链可验证
  的同次事务复用物理图；allocator/workspace仍实时读取。其他路径按原
  语义重新清点。运行后统计真实权威journal中的
  `same_transaction_registered_preparation_v1`，不以源代码存在代替触发。
- 冻结V1和单轮V2共同生效。这是已知负载开发版本的一次完整观测，不是
  独立留出评估，无重复均值/CI。不因结果未胜出重跑同键或改阈值。
  独立数值正确性未知时N_correct保持null，不能用token数量替代证明。

## 安全、复用与发布

沿用现有run_d293入口、72/80GiB服务限制、2GiB swap、外置4GiB上限、
CPU分离与整树释放。启动前核验GPU空闲、磁盘150/100GiB和剩余增长余量，
正式测量期间不进行维护/安装/构建/全量哈希。远端仅管理本项目工件服务
和本次监控，按InvocationID收尾；不触碰共享k3s或重新发布工件。

run-experiment技能用于复用已有回放；monitor-experiment用于核验实际进度
和资源释放；github-sync用于明确文件的检查/可回退备份，不stage用户manifest。
旧147投稿项、冻结协议、资产与用户修改不覆盖。最终V2存在分歧，不force
push；本次备份ref不冒称最终分支已同步。

原始目录：results/ieee_tc/p2_backend_qualification/d886_20261026；汇总目录：
paper_results/ieee_tc/p2_backend/d886_registered_preparation_full_w2。
开始/完成与失败状态以实际回执及EXECUTION_STATUS为准。

## 收尾与下一步

运行结束先确认GPU/服务域与远端释放，再验证4000请求终态、原生生成合同、
阶段恒等式、物理生命周期、真实交付及准备回执；按冻结口径与D883比较。
若无完整系统收益，保留全部数据并关闭此候选，不沿小组件无限微调。
达标前不画图，不预先宣称G1/G2或最优。

全量两模型同版本W0/W1/W2/G1/G2、剩余baseline/dLoRA、A1–A5、S1–S13、
独立数值证明、两轮审稿文档/图表和最终V2交付全部保留，完整goal active。
