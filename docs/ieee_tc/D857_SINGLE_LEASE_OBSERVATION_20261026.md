# D857：同一次文件读引用共用一个实时观察

## 主线、历史和假设

恢复参照D846的3B W2为3715/4000联合通过（92.875%），仍差85请求；
U=11929.629332 GPU-s，在12083.063870预算内。D711独立数值正确性仍未知。
D854全owner异步负向结果已保留并在D855精确撤回；D856原生stat边界没有净
收益，未接入服务。本候选不复开这两条方向，不改变九公式或在线策略。

复用D843调用者记录：acquire_confirmed有71个inclusive采样，source_snapshot
41个；这些不是调用次数、纯CPU时间或可节省时延。源码则显示一项明确重复：
acquire_confirmed先验证来源，然后在同一RLock内调用acquire，后者再次遍历
容量并再次验证来源。三次扫描之间没有需要另一次快照的受管文件写入。
原始组合可追溯到da9109bd。D153/D610处理规划观察，D731处理控制轮共享，
D834处理完成绑定；它们没有删除本次读引用中的重复观察。D850/D851处理通知
和执行线程边界，也不是这一候选。不减少独立事务或规划控制轮的核验次数。

可证伪假设：在一个文件owner锁内，身份、容量和引用安装使用同一份两遍
实时核验后的观察，可以删去重复扫描且保持原语义；是否改善完整W2由下一
单轮完整回放决定，不由局部次数直接推断。没有新线程、TTL或跨调用缓存。

## 原始实现参考及适用边界

vLLM 0.30.0的WorkerLoRAManager采用已加载对象的激活路径；加载新对象时先
验证再替换，注释明确其单线程执行前提。本次只借鉴“在受控事务内让后续
动作使用刚验证的同一对象”的顺序，不声称vLLM提供相同文件容量合同，也不
借用其性能数值。[固定版本官方代码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)

Python说明DirEntry.stat可能返回缓存信息。本实现仍为每个路径执行两遍
fresh os.stat，不将目录条目缓存当新观察。
[Python官方文档](https://docs.python.org/3.12/library/os.html#os.DirEntry.stat)

## 实现和保留边界

- acquire/acquire_confirmed在既有RLock内调用同一私有观察与安装过程。
- 一次_local_file_inventory产生容量及临时stat映射；首遍记录必须通过
  第二遍fresh stat核验后才公开。身份、根inode和容量从该观察投影。
- 入口owner/epoch仍严格核对；已预留包络内的块变化仍允许刷新epoch，
  不据此放宽物理预算或拒绝不变内容。错误内容/adapter/引用重绑定仍拒绝。
- 第二遍发现变化时不安装引用；已确认来源撤回一次，epoch增加一次。
  普通acquire也在扫描失败时撤回已确认来源，比原先先报扫描错更及时。
- 引用安装和LRU更新仍在同一锁内；幂等重复调用仍做新观察但不再触摸LRU。
- source_snapshot、demand_witness、独立规划观察、发布SHA、extent settlement、
  引用释放及native HOST/GPU执行侧检查均保留。锁外不暴露可跳过检查的参数。
- 非协作外部写入不受RLock约束；两遍检查仍不是原子文件系统快照，未作此保证。
- 无新工件池、无新trace、无请求打包、无共享k3s操作。

## 最小验证结果

先在未改生产的父版运行两项新增测试，保留原始失败：单次已确认引用实际
遍历三次，未通过“一次”断言；第二遍内容变化拒绝测试已通过。随后只改
上述生产路径，复用现有文件归属、发布、物理包络和native HOST原子读测试。

| 证据 | 父版 | D857 |
|---|---:|---:|
| 新增测试 | 2项，1项预期失败 | 纳入后续86项 |
| 已确认读引用的inventory调用 | 3 | 1 |
| 新版普通/已确认读引用的单文件fresh stat | 未由失败断言输出 | 均2 |
| 针对性完整检查 | 不重复旧资格 | 86/86通过 |
| CPU辅助域峰值 | 619479040 B | 676397056 B |
| high/max/OOM/swap | 0 | 0 |

完整文件容量回执与发布身份相等、幂等LRU、内容变化撤回、包络增长/缩减/
超限、错误owner/epoch/content、链接/特殊文件、引用释放、first-load和reuse
真实竞争线程原子读均在86项内通过。实际CUDA未初始化，GPU未运行，147项
历史保护内容和用户manifest未变。没有通过删断言消除失败。

最初driver中尚未执行的HOST测试类名误写为NativeHostPreparation，实际类名
为OwnedNativeHostMovement。after_resume只纠正这一个入口；原driver和执行
快照都保留。首次候选测试即86项通过，无失败后调参或GPU重跑。

## 改动与后续

| 文件 | 类型/作用 |
|---|---|
| faaslora/memory/residency_manager.py | 单次锁内观察投影和引用安装；无生产计时插桩 |
| tests/test_ieee_tc_file_observation.py | 两项新增次数/变化保护断言 |
| raw/d857_20261026/check.py、after_resume.py | 受限CPU验证、原始日志与SHA |

CPU验证只支持语义保留及重复工作减少，不证明TTFT/SLO或模型数值正确性。
下一登记唯一完整3B W2运行，沿用D846配置、真实远端、原始负载、D648阈值与
D657预算；不追加短GPU前缀或另一轮广泛资格。达到3800/4000联合通过与否
由数据决定；n_correct仍null、ranking仍false，直至独立数值证据闭合。
后续完整同版两模型、baseline/dLoRA、A1–A5、S1–S13、文档/图表/最终V2
仍未完成。代码检查点备份不是性能接受或最终V2同步。
