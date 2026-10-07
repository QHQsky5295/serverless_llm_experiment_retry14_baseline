# D307：原生 prefill 粒度的单一开发候选

2026-10-07，7B 优化阶段。前置证据与完整条件组见 D306_PREFILL_OVERLAP.md。
本文件登记验证，不预告性能胜出；不推进3B、外部baseline或正式主比较。
完整计划 SHA 为0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7；
冻结V1 SHA 为5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22。

## 可证伪假设与不变项

D303 的129/139条TPOT临时超阈值请求与peer-prefill包络重叠，但不是因果证明。
将 `max_num_batched_tokens` 从1024减到512，保持 chunked prefill 开启，
检验较小单轮 prefill 是否缓解短 decode 尾部。512是一个预登记二分预算，
不是自动求出的最优值，不进行点位扫描，也不把既有chunking包装成新机制。

保持D303容量8、slots8、rank16、上下文1024、FP16、TP1、显存比例0.92、
prompt759、固定输出上限256及全部IEEE公式/机制。保留同一控制律及配置，
包括队列2/1、活跃比例0.5/0.25、TTFT upper5000/lower2339.924886357039ms、
窗口45s、cooldown38s、hotness5s、movement3、bin27.43356142744625ms和EWMA0.5。
不同时改NUMA、加载并发、soft target、扩缩容、路由或安全屏障。

重新测量受后端配置影响的service/preparation初始化；由此产生的初始估计
变化如实记录。这是一个后端配置候选及其一致性标定，不称为严格固定全部
中间估计值的单因素算法消融。现有D170临时评价尺子和V1正式定义不变。

## 复用的统计与不得复用的统计

历史D170的512个唯一已完成请求输出长度继续作为开发初始化数据，不重复
warm回放。现有初始化器原先将长度与整个性能配置绑定；新增显式
`native_fixed_output_length_transfer_v1`，只允许声明的iteration token预算
变更，要求其余模型/输入/生成配置、后端、运行环境和trace身份相同。

理由：在相同fixed_length_greedy_v1、忽略EOS、无stop、相同目标和合法上下文
下，成功完成的原生输出长度由共同生成合同限定。批处理粒度会影响耗时和
可能的浮点输出差异，不改变这些已经观测并校验的完成长度。该规则不适用
自然停止、不完整请求或其他未声明配置变化。不用未来请求的target作在线
已完成长度；原始source身份保留，目标initializer生成新的model/profile ID。

仍逐样本验证原生终态、actual==target、prompt、去重和分组均值。明确记录
source/target配置与变更，旧严格绑定入口不放宽。**service/preparation耗时
加载器完全不变**，禁止用这个长度复用规则将D298旧耗时冒充512-budget数据。
数值正确性没有因此获得资格，N_correct与正式G1/G2仍不能填已达标。

此处复用了D298收集器、代表adapter/request、31波次和索引，未生成任何
新权重、完整trace、未来热点或另一整池工件。预先发布的远端压缩对象复用。

## 验证顺序与停止规则

1. CPU测试：旧初始化规则保持、显式长度迁移可追溯、自然停止/额外变更/
   错误预算/错误原生完成拒绝；现有source导出和精确profile绑定测试。
2. 新源配置与D298严格比较，唯一YAML性能变化为1024→512；实际child与
   source合同相等。D303控制配置完整保留。检查现有500池rank/内容身份。
3. 资源保护与备份后，现有收集器运行8 warmup＋240 measured；五来源、
   三轮、两次八lane排列，不改变D298样本选择。UUID/字节/无请求打包、
   原生计数、物理释放和资源事件均须通过；重新导出准确绑定的初始化耗时。
4. 复用现有100请求smoke及4000 W0完整回放入口。100条只判合同/安全，
   不能从前缀延迟宣布全负载胜出。每次保持独立cold本地cache和原始arrival。
5. 与D303全4000比较临时联合达成、TTFT/TPOT均值/P95、生命周期GPU-s、
   等待分解和输入/输出身份。保留退化，单轮不出CI、不称正式最优。
6. 一个候选，不扫更多budget；至多在出现实质新证据时追加一次最小验证。
   否则接受/撤销/明确权衡后返回主线，不陷入局部分组或重复标定。

## 历史与主要来源

历史代码783fa42引入D170已完成长度复用；175b37a只区分启动loader与服务
性能身份。D298验证cap8而保留1024预算，已明确TPOT可能退化；D300独立处理
scale-in参考，D301/D303修复confirmed-state作用域。本次不重复这些优化。

- [vLLM0.30优化文档](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)：iteration budget的ITL/TTFT权衡。
- [原生scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)：按剩余预算限制处理token；已与实际安装代码核对。
- [原生SamplingParams](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/sampling_params.py)：ignore_eos、max_tokens与stop语义，结合本项目原生完成验收支持长度复用。
- [Sarathi-Serve，OSDI24](https://www.usenix.org/conference/osdi24/presentation/agrawal)：prefill/decode干扰与分块思路，不外推其A100加速数字。

使用analyze-results组织证据，vLLM技能辅助定位后端路径；技能通用阈值和
硬件建议不替代本项目V1。无图形、无论文源文件修改；新数据独立目录保存。

## 启动前准备完成；尚未测量性能

58项CPU测试通过（39项长度/组装/导出，19项精确耗时profile绑定）。新增
长度迁移规则的4项测试覆盖显式迁移、旧入口拒绝、自然停止、额外配置和
错误原生完成；两个耗时加载器仍拒绝budget变化。实际源配置/child组装、
500个静态adapter及248个固定case均通过核查。9项嵌套来源SHA一致。

新配置SHA56704b6189eef8c878f66125d3867adafc680cc2386b3e1c616fce139c0436ec；
source spec SHA0a1044fb7233a8784fd0a336e2ee7de01c55d7801043990c1708fc75242df258。
复用脚本：`prepare_d298_capacity --prefill-candidate/--verify-prefill-launch`、
`run_d298_capacity_source.sh d307`、`export_d298_profiles.py d307`。前两项
准备已完成，不重新执行或覆盖；最后两项待真实源测量与资源释放后顺序执行。
原source collector和profile estimator未改，runner记录实际启动commit。

D115历史9.19GB大日志无损压缩为260.19MB，两次解压SHA一致后仅删除原始
未压缩副本；历史失败身份、请求文件和curated证据保留。释放约8.93GB，
可用169499455488B，超过150GiB门槛。离线压缩出现memory.high回收事件，
如实保存在单独存储回执；没有OOM/swap或并行性能运行。systemd的inotify
监控告警也保留，实际cgroup限制可直接读回；下一GPU启动仍须通过原有外置
watchdog附着/资源检查，不将准备记录当作永久准入许可。

五个CPU验证资源域峰值均低于0.60GB，无内存事件；GPU进程为空。
147项历史保护仍只有同一已知baseline脚本差异，服务runner和用户dirty
manifest均未变。当前没有新的GPU测量或remote操作，不宣称7B性能达标。
