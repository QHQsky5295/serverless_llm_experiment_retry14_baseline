# D724：按模型包络扩展 CUDA Graph 覆盖的有界资格

## 问题、历史与唯一候选

D723 的当前 3B 前100诊断发现：100个含多token分配的迭代为90–764 tokens，
execute_model平均主机wall60.418ms，FULL/PIECEWISE入口均未触发；原捕获
尺寸仅[1,2,4,8,16]。这支持捕获覆盖不足的假设，不证明全部耗时都是GPU
发射开销，也不预告完整性能改善。D719完整joint91.60%、U13508.300779，
数值正确性仍未知，不能关闭3B目标。

本轮核对了生产入口：实际引擎在scripts/run_all_experiments.py构造，
faaslora/serving/vllm_wrapper.py不是本轮执行入口。历史该构造路径没有
compilation_config透传；现有默认由原生vLLM决定。D529 LoRA元数据优化
完整回放退化、D687/D688 HOST元数据退化等拒绝结论不变，不恢复这些改动。

唯一候选保持并发8、token预算4096、max_model_len1024、FP16/TP1、LoRA
容量、所有控制机制和九公式。只显式指定原生compilation_config：

```yaml
cudagraph_capture_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
max_cudagraph_capture_size: 1024
```

选择规则为：保留原decode尺寸，以2的幂扩展至
min(max_model_len,max_num_batched_tokens)=1024。几何网格将覆盖范围的
捕获数量控制为对数级，非精确匹配时padding小于2倍；这是有待验证的
启动/空间/计算折中，不声称最优，不根据764反向选点，也不扫描多个上限。
它只覆盖至一个模型上下文规模；超过1024的合法聚合batch仍走原生非图
路径，不截断输入、不降token预算、不改变batch形成。

## 原始来源与实机边界

- [vLLM v0.30.0配置实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/vllm.py)：默认上限由max_num_seqs×decode_query_len×2和平台上限决定；本配置得到16。显式尺寸仍受调度token上限约束。
- [原生图管理器](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/cudagraph_utils.py)：混合批与decode分别捕获，按token/LoRA和兼容性选择；PIECEWISE无需固定请求数。
- [官方设计说明](https://docs.vllm.ai/en/v0.30.0/design/cuda_graphs/)：编译与图模式正交，不把NONE等同于全模型未编译；说明基于较早设计，实际已安装源码/日志为本次判据。

抓取的官方版本不假定与已安装文件逐字一致；资格回执保存实机文件SHA。
新增捕获会增加启动时间、显存和padding计算，全部须实测。D724不是论文
新算法贡献；如采用，需披露后端配置，对同后端参照保持公开适配边界。

## 实现和资格步骤

1. 现有引擎构造函数仅在显式非空mapping时deep-copy透传原生配置，原生
   校验失败直接终止；无该字段的历史路径不变，禁止失败后去掉配置重试。
2. 原始模型配置和运行摘要保留字段；现有FrozenServiceProfiles完整身份
   自动包含它，因此D701旧profile不能用于新Full。不得排除该字段偷渡复用。
3. CPU测试验证默认不变、隔离拷贝、原生错误传播与profile身份不同。
4. 复用现有native_execution_metadata资格入口，仅原3B前16条、原工件、
   原生固定输出；只启动一个GPU。已有详细观察器含同步/readback，故时延
   不作性能证据。检查实际PIECEWISE/FULL覆盖、请求/slot/lease映射、所有
   请求及allocation释放；独立数值正确性不因元数据通过被升级。
5. 此资格使用本地既有工件隔离后端验证，不启动远端，不伪称真实远程主实验。
   通过后才复用D701完整采样规格取得新profile，再做有界Full和完整回放。
   失败时保留原证据，最多两次有新增错误依据的最小资格，不兜底换eager。

本轮不启用新Full、不改SLO、不重建输入、不画图。run-experiment沿用既有
gated-launch/tmux，monitor-experiment核对真实进程；生产安全门槛不变。
服务72/80GiB、辅助3/4GiB、CPU亲和性、swap、150/100GiB磁盘与独立watchdog
均保留。旧投稿144项和用户manifest保护；共享k3s不操作。

CPU资格46项通过（19条既有pytest/Pydantic警告），峰值630599680字节，
high/max/OOM/swap全0；144旧投稿和用户manifest不变。管理脚本CLI复核发现
初稿将既有参数误写为--trace；初稿尚未执行，保留原文件，实际使用校正的
launch_checked.py及--replay-trace。不改引擎或资格门槛，不是GPU失败重试。
GPU资格尚待执行，不能宣布效果改善。

## 首次原生资格：启动环境错误，0条请求

预启动50397f0已备份到backup/d724-graph-coverage-20261022。原生新图触发
编译时ninja不在PATH，初始化失败，0/16请求；已有合格环境中ninja文件
存在。完整runner原本设置venv/bin PATH、CUDA_HOME、FLASHINFER_NVCC、
MAX_JOBS=2及VLLM_CACHE_ROOT，本轮管理入口漏继承它们。这不是算法失败
或性能结果。失败原始JSON、worker日志及物理释放记录全部保留。

gated-launch服务返回2、watchdog返回0，service路径消失且原生GPU上下文
已确认释放；资格对象自身没有完整physical_allocation，不能补造它的逻辑
释放时间或将失败启动计成0资源。第二次资格只恢复上述既有环境，配置/16
请求/安全门槛完全相同，另存native_execution2/native_launch2，不覆写失败。

第二次仍在初始化阶段、0请求退出：实际外层仍采用默认180秒，且漏绑定
D534共同编译合同；D641已有同类入口差异，D642已有完整修正。此次历史
核查不足导致重复的启动规范遗漏，应记为管理错误，不归因于图候选。
两次GPU均由外置census确认释放、无OOM。第二次11秒/0.27GiB仅为初始化
中的图内存profiling捕获，尚非完整启动或真实请求重放证据。

第三attempt只恢复D534/D642既有合同SHA328b2025…和主入口600秒保护，
不继续加超时、不改变候选。不把未开始请求的两次失败当作有效候选性能
比较；仍只允许该候选的有界资格与后续验证。先以既有工具链8项CPU测试
及实际helper绑定核验原生编译环境，另存native_execution3/native_launch3。
若完整既定环境下仍失败，停止当前候选的自动启动，依据实际阶段另行诊断。

## 完成结果：覆盖资格通过，尚无性能结论

第三attempt在既有工具链和600秒保护合同下完成。全部16请求的原生token、
adapter/lease映射和释放检查通过；2,736迭代中2,704个非空迭代、21,728事件
通过原有严格journal validator。没有更换配置或回退eager。

| 观察项 | 实测值 |
|---|---:|
| FULL图迭代 | 2,688；每次1个实际/填充token row |
| PIECEWISE图迭代 | 16；实际179–760 rows，填充256–1024 rows |
| PIECEWISE累计实际/填充rows | 8,984 / 12,288 |
| 引擎启动 | 99.485116秒 |
| 本资格物理GPU生命周期 | 164.534216 GPU-s |
| 最终实际graph pool | 0.17 GiB（原生日志舍入值） |
| 服务内存峰值 | 5,738,520,576字节 |
| high/max/OOM/swap | 均0 |

这证明本次顺序前缀的多token执行已走PIECEWISE，支持继续验证覆盖候选。
它不证明混合Full负载的服务质量改善：详细observer含同步/readback，不能
拿本次时间与D723直接作性能对照；启动缓存状态不同，也不能把99.485秒
解释为因果启动改善。原生映射不是kernel算术或独立数值正确性，n_correct
保持null，valid_performance_runs=0，ranking_eligible=false。

初次离线汇总误用event而非kind字段，保留analyze.py及失败scope身份；
另存analyze2.py并按原validator格式修正，未重跑GPU、未改变数据。最终
分析峰值84,353,024字节，无high/max/OOM/swap；144旧投稿与用户manifest
核对不变。详细来源、三次资源记录及空scope关闭见completion/目录。

下一项只复用D701的24代表/76波次/423既有请求采样规格，测量新编译身份
匹配的profile。旧测量不能混用，也不能把model identity排除编译配置。
之后有界Full验证和完整W0；当前完整3B仍D719 joint91.60%、U13508.300779，
数值资格未知，未达到冻结G1。后续矩阵与两轮审稿问题仍全部保留。
