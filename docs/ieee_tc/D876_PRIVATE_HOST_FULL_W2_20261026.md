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
