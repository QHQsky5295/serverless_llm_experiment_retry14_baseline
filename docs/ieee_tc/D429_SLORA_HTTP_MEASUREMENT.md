# D429：S-LoRA 同口径 HTTP 计量组件

2026-10-11。CPU 组件通过；未启动 GPU、远端服务或新的性能回放。
延续 D425–D428；不将本记录称为完整 S-LoRA 复现或性能资格。

## 保留原机制，分清测量边界

历史 runner 先顺序准备完整静态 500 池，再启动原生 worker 并加载 HOST 权重。
新接口保留这一语义：准备在共同部署通知之后、服务资源域之内；
业务仍按通知后 60 秒到达，不平移到 ready。禁止把原生全池准备替换成
PrimeLoRA 的请求驱动缓存策略，也不强制每个请求重新下载。

复用现有 HTTP publisher、SSE framing、round-robin 和物理生命周期 reducer。
新 schema 为 `ieee_tc_slora_http_replay_v1` /
`ieee_tc_slora_native_v1`，不是改名成 vLLM。顺序仍按请求 ordinal，
只等待该请求所选 worker；不按 ready 数重新选路，不隐藏重试。

`NativeSloraRequestHandler` 向原生 `/generate_stream` 提交一次：
canonical prompt、原 adapter 路径、greedy、ignore-EOS、原 target。
原生 SSE 整数 token ID 和 RPC 生产时刻用于计量；接收 chunk/HTTP EOF
只作独立诊断。完整帧、目标数量、原生 prompt IDs、adapter index、实际
进程 PID/出生身份与已核验 ready receipt 必须对应。取消关闭本次连接；
真实原生 GPU 取消路径尚待资格，不把 CPU fixture 当作已经验证。

独立的 full-pool helper 按静态排序串行获取恰好 500 个 ID，不读未来热点。
请求引用其部署准备回执，`artifact_transferred_bytes=0` 明确表示该请求
没有再次传输；`preparation_transferred_bytes` 是共享准备引用，**不能跨请求
求和**。总量以后从 resolver journal/唯一 transfer ID 计算一次。
真实启动等待继续保留在 TTFT/E2E，CPU-only 准备不虚构 GPU 时间。

主仓库终态记录器只接受两组明确匹配：
vLLM schema + vLLM token IDs，或 S-LoRA schema + native SSE token IDs；
交叉混用拒绝。物理占用并集与真实释放规则不变，原始未知正确性字段不补写。

## 历史与官方来源

本轮重读 D425–D428、原生 API/RPC/HTTP manager/采样及旧 runner。
[官方固定 RPC](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/model_rpc.py)
确认全池初始化及 CPU token-vector 观测边界。
[官方固定 API](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/api_server.py)
的 HTTPS、Git HEAD、本地字节 SHA 均为
`504a1128cd8c6868a7d4c4ed21127b93f3cc7db0fc009082677bf1c1de97c485`。
原有十文件兼容补丁保留；没有声称所有 RPC 文件与上游字节相同。
嵌套默认 diff SHA 仍为
`7a0f48f60c225789014ad960ec3ab1cf6388611b7650a4f656f1208c981c2d60`。

## 验证及失败保留

- 新桥接 12 项 CPU 检查通过，含真实 loopback HTTP、7-byte UTF-8/SSE
  分片、EOF 尾部、single-token、错误/截断/取消、不重试及错误身份拒绝。
- 全套 baseline 173 项通过，8.349s；主仓库物理/请求生命周期 236 项通过，
  4.536s。GPU 隐藏，未构造模型；合成 GPU events 不作为真实 lease 证据。
- 首次组合 164 项有两项失败，日志保留。其一为旧测试要求错误文本包含
  remote artifact，恢复原语义文本。其二为同长度改写的 stat 检测 fixture；
  当时没有前后 metadata，历史原因未证实。测试显式改变 mtime 后检查原有
  stat-identity guard，生产 resolver 未改；这不是每次命中重新做内容哈希。
- 取消测试的服务端连接重置日志是有意关闭连接的结果，保留原日志。
- 五个独占辅助 scope 顺序执行，4 GiB/zero-swap/CPU2,3,26,27；
  全部核对身份和空进程后关闭，high/max/OOM/swap 零；
  最大已记录 peak 500,289,536 B。147 个旧结果保护项零变化，三个用户文件
  和原生历史补丁未改。最新 GPU census 为空，磁盘 163,375,816,704 B。

Raw：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d429_20261011/`。
日志、源 SHA 和保护记录保存于同目录 manifest；小型证据包另存主仓库，
不复制工件、trace 或历史大日志。

## 下一步与未完成范围

此组件**尚未接入实际 S-LoRA service CLI**。下一步是在既有共同部署入口中，
连接原生 Python3.9 bootstrap、实际 GPU worker ready/进程证据、所有子进程限制
与退出核验；outer pre-ready HTTP 继续在现有 Python3.12 环境，不让 Python3.9
执行 asyncio.timeout。共享 full-pool materialization 完成后，每个原生 worker
仍执行其原有 HOST 加载，使用 D428 显式表示适配，并实测总资源峰值。

实际限制/释放和受影响 100 请求合格后，补缺失 500-ID、1,000 请求及一次完整 W0。
不重复已封存 PrimeLoRA/ServerlessLLM/vLLM W0，不跳 3B、不改阈值或公式。
其余 7B baseline、W1/W2/G2、3B、消融/机制/敏感性、文档/最终图表仍 OPEN。
本轮无环境安装、remote 操作/缓存重发布、权重/负载生成、清理或论文修改。
