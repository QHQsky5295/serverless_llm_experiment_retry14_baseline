# 2026-11-08：用户确认的主次指标与闭口规范已落地

最新目标附件：`2d471652-385d-404d-8aa5-d9233f0004ea/goal-objective.md`。
正式修改在D1126完整释放、分析与来源封存后完成，没有改变测量中配置。

- 主文件：`/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md`。
- 新批准快照：`docs/ieee_tc/PLAN_APPROVED_20261108_QUALITY_RESOURCE_CLOSURE.md`。
- 两文件SHA相同：`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`。
- 原完整内容逐字保留；只在顶部加入用户确认的执行修订。
- 旧快照及SHA保留：`PLAN_APPROVED_20261014_PUBLISHED_BASELINES.md`，
  `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。

修订明确G1、G2、辅助指标，执行/子结论/系统主结论三种闭口，禁止用
GPU低但尾延迟更高单独宣布系统最优；禁止拼接工作点、筛赢家或削弱对照。
规定在下一次重型实验前，先复用已完成实验的canonical结果/注册表，形成
闭口台账，再推进影响主结论的必要缺口。全部原实验和交付范围保留。

代码只更新`ieee_tc_preflight.py`的批准快照绑定和ApprovedPlanIdentity测试
中的名称/SHA。没有修改模型服务、算法公式、评价阈值、预算、安全条件或
任何历史结果。新preflight SHA：
`8f55c4577ee70bc48edb2fa82ac0438a3cc89587bf4d7b26d0617777a61adf62`。
对应测试文件SHA：`7d65a2246ddf4225887f0ee3532a5e5eb8fd8bb10d660503708371938a238bb3`。

run-u402.service，Invocation `187880dbd4a94b5cb8f131c65f93c519`：两项测试
2/2通过（0.002s，主进程0）；wrapper为1/timeout，实际域已移除。沿用既有
管理器收尾异常记录，不重跑通过的测试、不修改全局inotify配置。测试为
CPU2/3/26/27、3/4GiB、swap0、隐藏CUDA。没有新GPU性能运行。

D1126继续原计划/执行checkpoint身份，closure SHA
`5e005242b6e930fcca0ec89845dab41f6e20b6b539ed5ad72642ab9b38cf9539`。
其封存的旧preflight源码可由执行commit
`4828565398acec5cd2e98a562a05150845bca00a`定位；不能为新计划反向更新
旧source hash。执行者核验和用户确认均不冒称独立语义审稿通过。

当前待办：关键点备份；全部已完成实验组的闭口台账；随后继续既定缺口。
本文件不是整体目标完成声明。
