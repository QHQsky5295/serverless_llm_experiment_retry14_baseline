# D512：当前 PrimeLoRA 的原生迭代与控制开销诊断

2026-10-14；D510 七系统 W0 比较和 D511 全量离线分解已完成。
本项是一次原始前1000请求诊断，不是优化候选或普通性能点。

## 问题、历史与判别

D476 比 ElasticLocality 多885.732932 GPU-s，差额主要在业务到达窗口；
平均 E2E 差583.489529ms，其中原生解码窗差467.140437ms。
这些是位置分解，不证明 GPU 计算或状态管理造成差额。现有 D351/D456
观察发生在 D457、D468–D474 改动之前，不能当作当前各操作的耗时比例。
D329 已否定跨调用身份缓存，D352 原子 GPU-hit 复用、D355 控制交错、
D474 HOST 执行身份投影均已生效，不重复已完成的优化。

待检验：当前状态查询/加载 utility 是否仍占用原生 core 线程并与解码
间隔重叠。若控制占用不足或主要时间落在 step，则否定其为主要解释，
转向迭代组成；不机械追加 snapshot 微调，不任意提高容量或改 cooldown。

2026-10-14 再次核查官方材料：

- [vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：输入处理、utility 与模型推进的调用边界。
- [UniProcExecutor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)：本进程 worker 调用与异步返回边界。
- [vLLM 性能分析](https://vllm.ai/blog/2024-09-05-perf-update)：CPU 调度开销需要与 GPU 执行区分；其硬件/版本收益不外推到本机。

## 唯一执行合同

- 原 D476 配置，仅结果/NVMe/HOST 三条路径改变。生产代码和主 runner/
  preflight 相对 D476 执行提交不变；不改九个公式、profile、容量或 SLO。
- 复用 D293 runner、D328 wrapper 和 D351 CoreHostTiming；只新增精确
  campaign 标识。`host_spans_v1`，不启用 frames 或数值同步观察器。
- 原4000源的前1000、原顺序/到达/adapter/prompt/target，60逻辑ID、
  122510目标token，view SHA `0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981`。
- 真实远端已发布只读gzip，不请求打包、不重新生成输入/权重/交付池。
- 共同60秒准备、实际CPU/GPU生命周期、外置回放/监控、72/80GiB服务、
  3/4GiB辅助限制和原磁盘/内存门槛不变。同一时刻只运行这一重型任务。
- 计时器必须处在真实服务域、nonformal、diagnostic_prefix_v1 下；每进程
  200000 span/64MiB 上限不变。保存完整关闭、截断和观察器自身开销。

## 分析与退出标准

复用 `analyze_core_host_timing` 与既有请求窗口投影。核查原生输入/数量/
adapter绑定、时钟、所有进程日志关闭、GPU真实释放和远端传输来源。
先记录1000条完整覆盖、缺失、失败和截断，再统计各 utility 的主机时间、
线程CPU时间、迭代组成及其与请求窗口的关系。若按SLO分组，使用当前
D394冻结阈值，不沿用D351旧暂定阈值。

嵌套 span 不直接相加；input idle 不全称阻塞；step 不是纯 kernel 时间；
异步GPU可与控制工作重叠。条件相关性不当因果增益，观察器开销不事后扣除。
不得外推1000到4000、替代D476主结果或宣称已完成G1/G2。
独立数值正确性未知保持未知；原生数量和身份绑定不冒充数值证明。

完成后以本次证据选择一项优化，最小验证后普通完整回放；不得停留于
无新假设的重复诊断。当前只准备本项，尚未取得GPU诊断结果。
7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13、成本/文档/最终图表
全部保留。接近目标前不画图；旧投稿和用户修改零覆盖。
