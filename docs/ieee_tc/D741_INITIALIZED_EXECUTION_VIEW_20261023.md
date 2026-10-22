# D741：同事务不可变执行视图候选

状态：单一生产候选已实现，55项针对性检查通过；无新GPU实验或性能接受结论。

## 可证伪假设

依据 D734 调用观察、D740 已完成的消费依赖/组件测量及历史排除，减少主
事件循环展开不消费的完整规划描述，可能改善请求处理及时性。额外的编码、
IPC字节和保存成本可能抵消这一收益；必须以完整3B W0的共同联合SLO、
阶段时间和物理GPU时间判断，不以单测通过或CPU组件加速宣布系统改善。
与D737不同，本候选不改变RPC解析时机或原生观察，D737保持撤回。

## 唯一实现差异

`planning_cpu.py` 在既有单个worker完成原选择器、完整冻结校验和目标构造
后，为初始化后的 owned epoch 生成不可变执行 sidecar。完整 payload 原样
保留，用于公开导出/审计；读取副本相互隔离，不能以局部视图重新规划。
普通字典先经过原完整校验再投影，没有靠布尔标志绕过完整校验。

执行入口只在非pre-init、非显式file-only replacement、具有source_view
的合同中使用sidecar。其他合同继续完整消费；缺少应有sidecar直接报错，
没有错误后静默回退。完整selected/objectives保留，尤其未选来源仍可供
执行时victim定价；owner/epoch/内容、物理reservation、live容量、取消、
注册/提交与释放路径不变。原计划SHA未改，不将局部视图当新完整计划。

没有改变IEEE九公式、需求/成本、profile、阈值、SLO、并发或运行参数，
没有新worker、旧状态缓存或跨epoch可变别名。pickle只处理可信本地IPC，
不扩展到远端工件。

原 `output_bytes` 仍是主payload大小，另记 `initialized_execution_bytes`
为新增sidecar大小，二者之和是这两份序列化描述的字节量，不冒称全部wire
字节。worker CPU/完成时间包含投影编码；sidecar不进入JSON事件日志。

## 验证范围

复用D739受限CPU入口：现有planning CPU测试（真实spawn、隔离、取消、
完整导出/普通重验）、D740的27项混合执行/自动替换检查，以及显式文件
replacement合同测试。新增消费视图等价、不可变、部分输入拒绝、原profile
不匹配拒绝、字节记账和事件可JSON化检查。没有重复RPC广泛资格、GPU短
回放或再测一遍D740组件微测。

唯一tests1执行55项通过，68.373秒；峰值1,711,923,200B，high/max/OOM/
swap0。145项保护文件和受测源码SHA前后未变。没有失败后重跑或增加GPU
检查。实际资源身份/释放回执随检查点保存，不能把退出码当作资源释放证明。

原始目录 `results/ieee_tc/p2_backend_qualification/d741_20261023/`；小型
资格与源码快照保存在 `paper_results/ieee_tc/p2_backend/d741_execution_view/`。
历史D739源、D737撤回源码、D738失败与用户manifest不覆盖。

下一步：检查通过后备份此可回退候选，复用D736完整3B W0配置、D725 profile、
D648共同阈值、源trace/500池、真实只读远端交付与固定生成合同，只换唯一
运行/缓存路径和源码身份；不增加新前缀。D73692.65%是直接父参照，历史
其他点一并保留。候选不能以一次局部收益关闭G1/G2或后续全部实验。

设计参照及D159/D163/D193/D737差异详见D740文档。使用system-profile与
analyze-results区分组件观察/系统主张；备份按github-sync明确路径、保护SHA、
敏感信息与测试核查。V2远端已有分歧，不force，独立backup分支核验SHA。

## 实际备份与释放

21个明确文件提交 `528bdb12c5752f1abcc6b46ca793faf1a45187ba`，推送到
`faaslora_origin/backup/d741-execution-view-20261023` 并核对远端SHA。
V2未同步，用户manifest不纳入提交，旧投稿保护文件未变。测试/备份资源域
均核对InvocationID、inode和无进程后释放；备份域峰值58,757,120B、
high/max/OOM/swap0。`raw/post_backup_release.json`保留实际回执。
此补充在下一正常检查点备份，不为记录备份自身循环提交。

下一为D742完整3B W0，尚未创建或启动；不以本次55项检查宣布性能改善。
