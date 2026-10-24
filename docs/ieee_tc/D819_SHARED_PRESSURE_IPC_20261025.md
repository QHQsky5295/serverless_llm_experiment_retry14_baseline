# D819：权威共享压力的有界跨进程验证

2026-10-25。接续D818唯一候选；本轮只实现并验证IPC组件，**未接入正式
runner/native scheduler，未进行GPU性能回放，不构成性能改善或G1/G2达标**。
完整PLAN、执行规则及历史否定证据已核对。HydraServe已有隔离适配和观测继续
复用，不接入或改动用户明确说明的非本项目k3s集群，不重新询问集群归属。

## 动机与区别

D818的503条精确关联Remote请求中，压力start/finish包络均值分别197.259/
223.215ms；包含的RPC等待均值166.043/166.839ms。它们是条件观测，不是可直接
从TTFT扣除的因果效应。D595并行发送相同RPC仍等待全部core，D598整体退化，
故不重做该候选。本次研究的是取消逐传输的core消息投影：由同物理文件域的
唯一writer发布真实区间，由native准入点直接读取权威状态，而非缓存旧快照。

`serving-llms-vllm`用于核对原生后端边界。其通用性能目标和调参建议不替代
本项目冻结协议；不改prefix、batch、量化、模型配置或任何IEEE公式。

## 组件的精确合同

新增`faaslora/scheduling/shared_transfer_state.py`，不是另一套runner或调度器。

1. 唯一文件域writer持有匿名Linux memfd。每次仅发布当前active IDs与成员
   投影，不复制adapter、输入或请求结果，不新增磁盘工件池。
2. reader通过`/proc/<writer>/fd/<fd>`独立只读open，不用dup继承同一open
   description；核对boot/time namespace、inode、generation和pidfd身份。
3. writer的完整修订持exclusive flock；reader的完整“观测→使用”区间持shared
   flock，支持同native线程嵌套读取而不提前解锁。使用pread/pwrite与内核锁，
   不假定多字段Python赋值、裸共享内存或本地thread lock跨进程原子。
4. publication回执明确`native_acknowledgement=false`，不伪装每个core已读。
   attached成员必须看见全部真实区间；retiring成员保留其既有区间，retired
   成员不能成为新准入的压力来源。成员状态本身仍由既有domain生命周期负责。
5. `publish_async`在工作线程等待锁，避免阻塞控制协程；重复取消必须join
   已发publication，再传播取消。取消不自动清空压力，也不等于IO完成。
6. 有未决区间不能close；owner退出、错误identity、损坏/关闭publication和
   无有效订阅均报错，绝不回退为零压力。memfd内存仍属于实际服务资源域。

这一组件**不自行证明IO已经join，也不拥有GPU/KV预留**。后续必须由现有
`SharedFileTransferDomain`在真实IO开始前publish，真实writer/清理join后
publish finish；未知终态继续保留。不能由组件测试冒称整个生命周期已通过。

## 资格结果与失败保存

首次12项测试均在初始化阶段失败：已冻结Conda Python3.12.12不导出
`os.memfd_create`、`os.MFD_CLOEXEC`或`os.pidfd_open`。没有GPU或真实IO开始。
完整失败log、源SHA和初版源码tar保留，不删除或改写为通过。

核查实际host libc与已安装头文件后，memfd改为明确的libc ABI调用；
`MFD_CLOEXEC=1`是Linux头文件常量而非人工调参。pidfd复用项目原有
`metrics_collector._open_pidfd`（已限定Linux x86-64），不新增轮询或磁盘兜底。

随后在实际native环境中，12项通过，2.062s：

- 两个spawn进程无逐事件RPC读取同一真实区间；30次交错修订读取完整；
- 一个真实reader持锁时，writer的非阻塞exclusive获取确实被内核拒绝；
  内层snapshot退出后外层锁仍保持；真实publication等到释放后完成；
- publication等待期间重复取消仍等待实际提交，提交后区间保留待清理；
- IO期间成员加入、退休投影、未知区间关闭拒绝；
- writer测试进程真实退出后，即使旧memfd仍可读，也不返回陈旧压力；
- reader只读权限/线程归属、错误inode/generation/clock、损坏数据、非法
  投影及显式关闭均被检测。

这不是同时运行完整vLLM，也不是传播开销microbenchmark；不据2.062s推算系统
收益。组件测试手工提供合法区间，没有证明production调用次序。该限制明确保留。
失败/成功aux峰值50,663,424/118,521,856B，high/max/OOM/swap均0，均已退出。
本轮未改现有production调用路径、远端服务、工件、负载或生成合同，不画图。

## 下一项同一候选的集成（不是新增优化方向）

1. 复用既有SharedFileTransferDomain及其成员锁、取消join和不确定状态，
   改为实际publication，不复制第二份生命周期实现；attach仅建立并确认reader。
2. native observation增加明确transport provenance；在同core的实际
   `prepare_host`及`admit_and_demand_load`观测/同步提交整个区间保持reader
   transaction，禁止只读一次后无锁延迟使用。其他Eq.(8)/(9)计算不变。
3. 记录publication前后时间/实际revision；旧逐ownerstart/finish回执不能冒充
   新reader确认。旧RPC模式和历史数据身份保持可回退，显式新执行身份。
4. 补真实domain与native bridge的受影响测试，复用未改路径资格；仅必要原生
   验证后新键完整W2，再依据完整目标比较。不要重跑D818分析或无变化的本组件
   资格，也不增加无问题指向的短回放。上线前核实锁持有区间不存在跨进程回调
   依赖环；不能用CPU测试替代这个检查。

跨物理计算节点不共享此memfd；只说明当前同物理文件域内的实现，不把它写成
多计算节点scale-out验证。性能可能改善也可能因锁竞争退化，完整结果决定去留。

## 原始来源（设计依据，不代替测量）

- [vLLM0.30共享广播源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/distributed/device_communicators/shm_broadcast.py)：
  支持研究本机共享通信路径；其ring-buffer/读写flag假设不直接移植为本项目锁协议。
- [Python fcntl官方文档](https://docs.python.org/3.12/library/fcntl.html)：
  shared/exclusive flock接口。当前候选用实际跨进程测试验证open-description行为。
- [Python mmap官方文档](https://docs.python.org/3.12/library/mmap.html)：
  明确MAP_SHARED与私有映射的区别；本候选没有使用裸mmap实现无锁协议。
- [Python os官方文档](https://docs.python.org/3.12/library/os.html#os.memfd_create)：
  memfd接口与平台依赖；当前Conda构建并无此Python导出，不能只凭文档假定可用。
  实际libc声明与常量核查`/usr/include/x86_64-linux-gnu/bits/mman-shared.h`
  和`/usr/include/linux/memfd.h`，失败时不切换其他语义。

完整范围不变：D813 W1联合93.975%、D817 W2联合89.90%未达标；D711数值
正确性仍未知。同最终版本两模型主比较/Full、A1–A5、S1–S13、成本/CE/Pareto、
两轮审稿文档及最终图文/V2均未完成。该组件通过不关闭任何上述实验。
