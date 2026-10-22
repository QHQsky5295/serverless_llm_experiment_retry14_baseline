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
