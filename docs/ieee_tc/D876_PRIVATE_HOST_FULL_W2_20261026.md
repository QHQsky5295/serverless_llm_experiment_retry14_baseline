# D876：私有 CPU 构造候选的完整 3B W2 验证

2026-10-26。D875真实原生资格通过后，返回完整回放主线，不再短Full或profiling。
状态以执行回执为准；本登记不是已完成性能结果。

- 唯一机制差异：D873/D874 `ieee_host_construction=private_cpu_core_commit_v1`。
- 配置由D870派生，仅另改output/NVMe/HOST三处独占路径；既有trace、W2 adapter
  map、prompt、target、种子42、3个实例上限、准入和控制规律全部不变。
- 使用既有完整runner的d876入口，真实远端已发布gzip/isal，不按请求打包；
  D747 descriptor、冻结工具链、60秒部署通知、全部资源安全/计量条件不变。
- 不恢复负面graph/整owner线程化，不调阈值、不改对手、不重跑同键挑赢家。
- 结果保存D876新目录，原结果保护；若同模型W2通过，也不替代W0/W1及7B。

完整请求身份、原生token/计时、CPU构造触发/结算/取消、来源与HOST/GPU预算、
真实远端交付、物理GPU释放全部保留。只按冻结V1+单轮V2分析；无CI。
N_correct未知时仍未知，不能把4000原生长度成功直接叫4000数值正确。
目标为联合≥3800/4000及U≤12083.063870235463，再比较资源与P95；独立数值
正确性与外部比较未完成前，不宣称整体最优。历史D870未达标点永久保留。

冻结配置：`configs/ieee_tc/20261026_d876_3b_max3_full_w2.yaml`。
原始目录：`results/ieee_tc/p2_backend_qualification/d876_20261026/`。
汇总目录：`paper_results/ieee_tc/p2_backend/d876_private_host_full_w2/`。
重型运行前后不压缩/扫描/构建其他任务，监控磁盘及内存；结束后复用既有
complete/native_and_lifecycle/close_delivery分析器，保留失败与不足。目标前不绘图。

## 实际尝试：到达前拒绝，不产生性能点

已备份提交`bef55873a9084a962d3e6a272d097443a04309b0`后启动。原生远端服务
健康，链路1000Mb/s/full。ScenarioRunner在加载D725服务标定时拒绝新的
`ieee_host_construction`配置身份：`service profile model/configuration/context mismatch`。
这是候选注册没有同步补齐测量初始化的遗漏，不通过删除身份字段、关闭检查
或更改旧profile标签来通过。

原计划4000请求，实际到达/提交/终态均0，无GPU allocation。U_obs=0是启动
前观察，不是低成本成功；正式U及N_correct为null、排名不合格。全套资源和
远端两服务/monitor已正常释放，失败目录永久保留。

下一步只补本候选受影响的source-matrix标定，复用D725最终有效423请求索引、
原工件/trace、`run_d298_capacity_source.sh`和`export_d701_3b_profile`；不新造
负载、不重做共同SLO或Resident参考。取得匹配新身份的真实初始化测量后再
执行完整新键W2。当前D875资格仍有效，D870仍是最新完整性能，不宣称改善。
