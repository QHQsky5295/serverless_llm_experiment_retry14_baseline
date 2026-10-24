# D820：权威共享压力接入；尚未形成新的性能结果

2026-10-25。本轮继续D818→D819的同一候选，不另开优化方向。HydraServe已完成
隔离适配及既有比较；用户确认共享k3s不属于本项目，不读取其凭据或改动服务。
真实远端只读发布缓存继续复用，本轮未启动GPU回放或远端任务。

## 科学问题和历史约束

D817完整3B W2联合SLO89.90%，U11283.832798GPU-s；预算内但未达95%。
D818对503条可绑定Remote请求观察到start/finish压力确认包络均值分别为
197.259/223.215ms，内部RPC等待是包含项，不能相加或事后扣除来造SLO。
D595并行RPC在D598没有整体收益，D628移SHA锁在D629没有支持，均不重做。
本候选改变权威状态的传递架构，而非并行发送同一组消息、删除压力项或使用
陈旧缓存。性能是否受益只由之后的新执行键完整回放决定。

## 实现及九公式语义

- `resource_coordination.ieee_file_pressure_transport=same_host_memfd_flock_v1`
  显式选择新通道；缺省仍是旧`native_rpc_v1`，不是失败后回退。
- 配置属于文件域控制器，不属于模型执行配置；放在`model`下明确拒绝。
  不给FrozenServiceProfiles增加身份豁免，不改既有profile/生成/内核参数。
  首次53项检查时配置仍放model；发现身份归属问题后改为coordination，首版
  源快照和通过记录完整保留。该版本未跑模型，未尝试绕过profile一致性。
- 实际物理文件owner是唯一writer。每core仅订阅一次；新core的订阅确认后，
  controller发布其当前全部活动区间，才允许该副本进入准备/服务路径。
- 每次IO开始前发布区间；实际操作及取消后的writer join完成后才发布结束。
  每次发布有generation/revision，不伪称各core已执行start/finish RPC。
- 缩容进入retiring后等待已有区间，不承接新任务；retired reader不能返回
  虚构的零压力。未知IO终态保留active，拒绝正常close，等待明确owner恢复。
- 同native线程的压力读锁覆盖当前scheduler/KV/length观测到同步worker
  admission/load返回，防止观测与使用之间插入新文件传输。旧通道仍依赖
  原同线程串行事件。active计数、transfer limit、物理预算与IEEE公式不变。
- 主动准备与按需加载的回执保留transport/generation/revision。domain退出
  要求所有成员退休、实际IO结清；普通shutdown、无pool及profile清理均接入。
  关闭发布者不是GPU释放证明，仍使用既有真实进程/cgroup/设备证据。

## 同步边界与依赖环核查

使用`serving-llms-vllm`核查原生边界；不采用技能里的通用500ms/利用率目标或
参数扫描。官方[vLLM0.30 UniProcExecutor源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
的默认blocking collective_rpc在调用线程执行worker方法。本桥继续要求精确
UniProc类型；不扩展到TP多进程，也不把异步model execution视为同步完成。

锁内worker路径沿当前gpu_monitor/native reference owner进行本地文件读取、
tensor检查和本机加载，不await controller或等待controller发布。这是当前
调用图的核查结果，不是所有后端都无死锁的证明。controller的publish用已有
executor线程等待排他锁，事件循环仍可推进，完成/重复取消都join真实任务。
[Python flock接口](https://docs.python.org/3.12/library/fcntl.html)提供跨进程锁；
D819已经通过真实spawn/独立fd/owner退出等12项组件测试，本轮不重复。

## 本轮验证和边界

`tests/test_ieee_tc_shared_pressure_integration.py`新增13项，包括双core共同
区间、迟到副本、退休与新IO隔离、重复取消、未决操作、结束发布失败、通道
绑定、真正runner executor取消join及两条native桥的锁区间/异常解锁。
同core测试使用假的推理对象但真实IPC/锁，不称实际CUDA执行证明。
另42项受影响旧路径回归合计55项通过（0.325s），保留所有输出；日志中的
构造失败警告属于既有拒绝测试，并非被隐藏的实验失败。

两次受限CPU检查峰值538697728/539525120B；high/max/OOM/swap均0。第一轮
53项0.436s记录不覆盖。机器没有GPU计算作业；未改工件、负载、SLO、后端
版本、正式模型配置或旧结果。旧147文件及用户修改在备份时再次逐SHA核验。
执行/源快照位于`paper_results/ieee_tc/p2_backend/d820_shared_pressure_integration/`。

## 下一主线

1. 保存本检查点并核验远端备份；不提前启用新正式配置。
2. 复用现有native资格入口，只补实际进程订阅、当前观测与加载这条改变的路径。
   不重采无变化的整套profile、不新增重复Full前缀。source-profile入口允许
   spec显式声明`file_pressure_transport`，保留原native model identity。
3. 新键完整W2复用D817输入与冻结模型/控制配置，仅启用本候选并使用新路径。
   验证发布/订阅/实际准入计数与所有IO区间，不能将publication当native ACK；
   D594旧逐core RPC分解不直接用于新通道。继续按原SLO、真实GPU-s验收。
4. 完整结果决定接受/撤销，并返回同版W0/W1/W2主线；不承诺改动必胜。

D813 W1 93.975%、D817 W2 89.90%未达标，D711独立数值身份仍未知。
同最终版本7B/3B主比较与Full、A1–A5、S1–S13、成本/CE/Pareto、两轮审稿
文档、最终图表和V2交付仍未完成。整体目标active，当前不画图。
