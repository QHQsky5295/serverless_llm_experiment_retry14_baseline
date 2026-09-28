# D100：3B Full4000 W0 第七次完整回放——失败诊断

日期：2026-09-28。执行代码 `6815beea8453ba56894e206a0ac4ad914ce7b3cf`。
这是开发期资格诊断，不是正式 M1/M2 结果。沿用 D88/D89 配置、既有 4000
请求/500-adapter 输入、60 秒部署通知、真实远端已发布交付协议和从计划到达
计起的 1800 秒期限；本轮没有修改这些条件。

## 已完成的终态和资源核验

| 项目 | 观测值 | 判断 |
|---|---:|---|
| 计划/提交/终态请求 | 4000 / 4000 / 4000 | 回放执行完毕 |
| 成功且原生生成合同匹配 | 2445，61.125% | 未达到全部成功要求 |
| TimeoutError | 1450 | 失败，不改写为单纯 SLO 违约 |
| 其他返回失败 | 105 | 全为父侧 native RPC 所有权未决，新生成未提交 |
| 物理 GPU 租约/已释放 | 85 / 85 | 无未释放租约，GPU 计算进程已清空 |
| 生命周期 GPU 占用 | 22449.901938 GPU-s | 完整失败运行的资源消耗，不进入达标排名 |
| 资源监控样本 | 6117 | high/max/OOM/OOM-kill/swap 全零，无保护告警 |
| 服务内存峰值 | 30640865280 bytes | 主机最低可用内存 88298586112 bytes |
| 远端 UUID 配对 | 132 / 132 | 工件身份对应；部分接收与发布状态分别保留 |
| 本地内容验证并发布/未发布 | 129 / 3 | 未发布不得计为内容正确 |
| 客户端已接收字节 | 305066543 bytes | 包含未发布获取，不能删除其消耗 |
| 服务端 socket 已写字节 | 306357477 bytes | 不冒充客户端实际完整接收或纯网卡流量 |
| 已验证发布的逻辑字节 | 5006562092 bytes | 仅对验证成功记录求和 |
| 请求中打包/临时归档 | 0 / 0 | 复用共同只读交付缓存，无重建工件池 |

四个互斥 GPU-s 窗口为：准备 48.363564、到达 15695.832134、drain
6559.413611、终态至释放 146.292629；未舍入值相加等于总量。
launcher pass=true 表示执行和清理通过，不表示 Full、共同 SLO 或数值 LoRA
资格通过。现有工件数值可区分性的限制仍在，不能由 token 数量一致替代。

## 副本恢复与原因边界

最早创建的四个 runtime 完成 2124 条请求；它们的类别是 1 个 initial、3 个
natural_scaleout，不能统称四个 initial。后续 61 个有成功终态的替代 runtime
完成 321 条。83 次 quarantine 均记录 released，85 个物理租约独立确认已释放。
这证明恢复后确实又提供了服务，但频繁隔离和补建仍存在，不是稳定性已解决。

本轮 residency 302 completed、303 superseded、6 cancelled；file preparation
388/303/5；GPU preparation 34/303/4。没有原先导致整个回放退出的 failed
planning epoch。303 次 supersession 中，302 次发生于 native_registration，
1 次发生于 native_gpu_source，后者确实触发 D100 的已确认 source 消失处理。
不能单凭跨版本完成率就把总体改善归因于该修改。

source 观测 requests=38780、collections=20006、joined=18774、RPC invocations=2594、
stale rejections=3694、membership rejections=181；collection 不等于 native RPC。
数据尚不足以仅凭总数确定长等待的具体锁、资源或取消链条。

## 全部请求提取后的阶段诊断

既有流式提取完成，原始 5.23 GB 结果没有整体载入内存；峰值 RSS 81408 KiB，
耗时 959.99 秒。全部 4000 请求与提交/终态集合一致；2445 条成功的原生 token
数量、prompt SHA、token-ID SHA 与终态账本逐条对应。105 条返回失败均为
`native RPC ownership unresolved; new generation withheld`，生成提交状态均为
`not_submitted`。这不证明同一 runtime 的其他在途操作已结束，也不授权提前释放。

下表仅是 **2445 个成功请求的条件诊断**，不是完整工作负载成绩；分位数用 Type-1。

| 阶段/指标 | 平均值（秒） | P95（秒） |
|---|---:|---:|
| 用户 TTFT | 828.854468 | 1788.325484 |
| 用户 E2E | 835.362005 | 1794.589798 |
| dispatch/admission 总等待 | 812.434826 | 1786.452232 |
| 其中 dispatch window 等待 | 790.440197 | 1781.057062 |
| 其中 runtime slot 等待 | 19.840378 | 56.068888 |
| 其中计划到达释放迟到 | 2.154251 | 6.603851 |
| service TTFT | 16.419641 | 93.995089 |
| 其中 native vLLM TTFT | 0.388971 | 1.330039 |
| 其中 native 之前的 service shell | 16.030670 | 93.504038 |
| LoRA I/O 观测 span | 14.596914 | 91.903766 |
| parent RPC overhead | 4.060543 | 11.572927 |
| TPOT | 0.025736 | 0.055027 |

前三项等待均值相加对应 dispatch 总等待；service shell 加 native TTFT 对应
service TTFT。其余 span 可能嵌套或重叠，不再累加。两个额外 parent delay 字段
在请求级投影中不存在，curated 样本数为 0/值为 null，不用旧摘要填充。

首个 timeout 观测于业务开始后 2888.775891 秒；首个 ownership-unresolved
返回失败在 3739.963262 秒。这说明不能把最早排队的起因直接归于稍后这些返回
错误；恢复/取消链条可能放大问题，因果关系仍须检查。

待验证的单一主假设是：同步文件状态构造/规划占据控制线程，使完成处理及新请求
推进变慢，再触发排队和取消。先离线剖析原方法及保存的状态，再决定是否修改。
依据 [Python asyncio 对阻塞代码的说明](https://docs.python.org/3.13/library/asyncio-dev.html)
和 [vLLM 官方 CPU profiling 方法](https://docs.vllm.ai/en/latest/contributing/profiling/)，
不能从上述跨度独自断言具体函数是根因，也不先把工作盲目移到线程池。

## 未发布获取与统计限制

`translate_lora_0114`、`writing_lora_0119` 收到并验证归档，但没有完成内容
验证/发布；`code_lora_0099` 客户端只记录接收 1048576 bytes，服务端记录 socket
写入 2339510 bytes。两端 UUID 和工件身份一致，后者没有完整接收证明。
目前不预先把三条归因为网络故障或正常取消。

初次收集脚本因沿用所有两端字节相等的假设拒绝输出，已保留该错误；修订后的
离线统计对成功发布继续严格要求相等和完整验证，对未发布显式保留两端字节。
这不是修改实验事实或放松成功门槛。preliminary JSON 的 kind 中误留 `d97`
字样，其路径、数据和 D100 日志对应；原输出保留，最终 curated 身份使用 D100。

## 来源与当前工作

原始目录 `results/ieee_tc/p2_backend_qualification/d100_20260928/`。
`full_attempt7_preliminary.json` 已完成上述核验。5,229,448,950-byte 完整结果
使用既有 D96 bounded jq streaming projection，在独立 4 GiB CPU 资源域中
提取全部请求，禁止整体 json.load；已完成，原始数据不变。

完整 curated：`paper_results/ieee_tc/p2_backend/20260928_d100_3b_full_w0_attempt7.json`，
SHA `ff70e419f638d94d9581c72b2ddc372c279a02d8cac97c307934095aaf3ea4a6`。
失败细表：同目录 `20260928_d100_3b_full_w0_failure_breakdown.json`。
147 项投稿保护清单零变化；全部原始日志及失败分析尝试保留。

推理退出后才核对身份并停止两项远端服务、监测服务和本地空辅助组；没有停止
无关进程。完整提取、失败分类和条件时延诊断已完成，下一步核验来源并备份，
随后针对最早的等待原因做一个因果优化。不重复发布缓存、全池下载或 profile。
7B、warm SLO/Resident、基线主比较、A1–A5/S1–S13 仍待执行。
