# D878：修正受控准备入口后继续同一候选标定

D877在32/423条后因collector传入未注册计划编号而等待，已保留失败并释放。
修复和测试提交`1d11ba52d15838702db83491a1b9c1eb9fe5955b`已推送核验
`backup/d877-controlled-source-fix-20261026`。生产私有CPU构造、计划检查、
九公式不变；受控空cache setup明确不冒充residency planner。

本次直接复用同423条原cases、24代表工件、76 waves及D877模型配置。
只新建raw/spec命名空间，复用18+1项针对检查及D874/D875原生资格，不再
重复CPU测试、短Full、原因profiling或SLO/Resident标定。严格profile exporter
只接受本次完整实测；D877片段不混入。真实remote、isal、D747描述器和
资源包络一致，既有工件/trace不复制、不覆盖。

结果以`paper_results/ieee_tc/p2_backend/d878_private_host_profile/completion/`
为准，未完成不宣称通过。导出后直接登记完整新键3B W2，保留D870对照与
D876零到达失败。独立数值正确性仍未知，完整目标active，全部后续范围不变。
