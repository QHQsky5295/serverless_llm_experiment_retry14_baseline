# D736：3B同次已验证规划投影的完整W0评估

本轮只评价D735候选：由已有规划进程返回同次已验证的不可变路由投影，
controller保留实时身份/epoch/capture检查后发布。它减少重复图校验，
不减少native物理观察，不替代执行时容量检查，不改变IEEE九式与目标集合。
发布时点延后是明确的闭环变化，效果尚未测定，不能声称严格时序等价。

依据为D734已完成的controller采样、原始源码、D329/D516/D689等历史排除
及官方Python/vLLM执行隔离资料；详见D735设计文档。本轮不再跑相同前缀、
HOST微测或广泛后端资格；只复用针对性CPU/spawn检查，然后一次完整回放。

## 冻结执行合同

- Llama-3.2-3B、原4000请求和500逻辑adapter池、seed42源trace、W0；
  不生成新工件/负载，保留已知负载开发身份，不改称独立留出正式结果。
- 配置与D732逐字段相同，只换output、NVMe工作cache、HOST工作cache三路径；
  继续使用D725准备/服务profile与D648共同SLO，不重新定标。
- 共同提前60秒部署通知、自然控制、open-loop、fixed_length_greedy_v1，
  真实远端已发布只读gzip，线上读取/传输/竞争/物化全部计入。
- 现有vLLM0.30.0后端、CPU/内存/GPU上限、独立监控及真实释放计量不变；
  无stack或原生细粒度profiler，不修改远端配置、不改共享k3s。
- 执行前核验磁盘/内存与145个保护项；用户manifest不stage、不覆盖历史。

## 判定与后续

完整结束后核对请求/原生token/adapter/计时/远端内容及资源释放。使用冻结
V1加单轮V2、D648阈值，报告TTFT、TPOT、联合达成、P95及物理GPU-s。
当前完整参考D732为92.500%/15903.076489 GPU-s，另保留D726等所有候选。
不将短前缀D734与完整W0直接比较，不以局部CPU开销减少宣告G1/G2达标。
独立数值正确性尚未知，不能由计数/绑定合格推定n_correct=4000。

本键只执行一次；低于95%照实保留，不重跑同键挑赢家。若达标仍须审计
资源目标、版本一致性及后续W1/W2，不因W0改善关闭全部目标。7B最终同版本、
3B其余场景/基线/dLoRA、A1–A5、S1–S13、CE/Pareto、审稿文档与图表仍须完成。

## 路径与状态

- 配置：`configs/ieee_tc/20261022_d736_3b_validated_publication_full_w0.yaml`。
- 现有入口：`bash scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh d736`。
- 原始目录：`results/ieee_tc/p2_backend_qualification/d736_20261022/`。
- 小型资格/来源：`paper_results/ieee_tc/p2_backend/d736_3b_validated_publication_full_w0/prelaunch/`。

登记时未启动。CPU检查及明确文件备份完成、资源门槛通过后，由现有tmux/
scope/远端收尾流程执行；是否已启动以start_receipt和实际服务身份为准。
