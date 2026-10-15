# D555：ServerlessLLM 固定原生工作点的 W2 高频轮换

2026-10-15，启动前登记。本文件创建时尚无 D555 性能结果；实际状态以本轮
回执和 EXECUTION_STATUS 为准。遵循完整 PLAN、冻结指标 V1／单轮 V2，
推进既定 7B 主比较，不增加 baseline 性能调参或重复相同执行键。

## 科学问题与共同输入

在 arrival、prompt 和目标输出不变时，将既定热点轮换由每500请求改为每100
请求，测量实际联合 SLO、尾 TTFT、真实生命周期 GPU 时间及远端获取。
PrimeLoRA D554 的 W2 联合达成89.55%，尚不能宣布 G1/G2 完成；D555 为同
负载提供 ServerlessLLM 观测，不能用 W1 不同到达条件直接代替。

- 复用 D553 的 W2 映射，不生成新权重、完整 trace 或工件池。4000请求、
  458224目标输出、497个实际 ID；500-ID 池中498零权重、2非零权重的既有
  限制保留，497个 ID 不等于497个独立训练模型。
- source SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`；
  map SHA `48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`；
  view SHA `5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7`。
- 保留 source42/development41、`formal_run=false`，不改写为独立留出证据。
  `fixed_length_greedy_v1`、`role_lines_v1`、原生 token/adapter 绑定；独立
  数值正确性继续为未知，不由 HTTP 成功或目标长度推定通过。
- 共同提前60秒部署通知；不平移业务起点。1800秒保护、D394共同 SLO、
  95%可行性和0.75参考预算定义不变。W2参考尚未完成，不填造预算。

## 固定原生配置与复用证据

保留 D414/D537 的原生 vLLM0.10.2、native FP16 store、RR、已披露的必要
ready轮询和 request-identity 修正。min1/max4/target4/keepalive10、maxseqs4、
maxLoRAs4、CPULoRAs4、tokenbudget1024、GPUfraction0.72、非 eager 不变。
实际四GPU store持有计入物理资源，逻辑缩容不冒称物理释放。

D539 已核查历史队列、16份 worker 启动日志和原生扩缩容源码；没有新证据
支持继续修改 baseline 策略。模型图命中缓存仍有初始化开销，用户排队不等于
checkpoint加载时间；不把较差效果无条件归为原论文机制缺陷。
见 [D539 核查](D539_SERVERLESSLLM_QUEUE_REVIEW.md) 及
[固定版本官方 router](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)。

本轮输入核查已完成，无 GPU：

- 相对 D537 baseline checkpoint，49个接入函数/类和33个测量函数/类 AST
  未变；仅公共 population、plan loader 的 D553 W2 接入变化。
- app/backend/router/scheduler 的原始、测量后及共享源码 SHA 与 D537 一致。
  runner 除新目录、独立身份与 canonical source 路径外逐字一致。
- 复用 D537 的114项原生/分析资格与 D553 的386项 W2 检查，不重复运行。
  D554后续4项 wrapper 检查的源码绑定也核实；不将重叠检查相加为新测试数。
- 本次6项输入检查通过：完整共同视图、服务/回放同源，以及缺map、错SHA、
  错profile、截短W2均拒绝。实际497 IDs/458224目标核对通过。
- 原 checkpoint、loader overlay、磁盘编译缓存复用；只在启动时安装本轮
  有独立备份/回执的同一 loader-only overlay，实际释放后恢复。
- 原生编译缓存实际目录内容记录，不清全局缓存、不额外预热 GPU/HOST/KV，
  不声称所有底层 kernel 初态完全相同；启动/capture照实计量。

资格辅助域实际递归为空、已停止且路径消失；峰值404832256字节，
high/max/OOM/kill/swap均为0。这是接入核查，不是新的性能实验。

## 运行与收尾边界

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d555_20261015`。
复用 D537 的运行、远端管理、释放、分析和明确路径备份脚本，仅改变 campaign
身份和显式 W2 输入。BASE checkpoint `aff60bb210c6d1fb9ce94534c5ef37521976bc0a`。

每轮空的本地 LoRA 下载缓存；远端复用 D78/D80 通知前已发布只读gzip。
真实读取、传输、竞争、本地物化计入时延；请求中不打包、不事后减时间。
不改无关 k3s，HydraServe 的隔离适配已完成，不重做基础资格。

服务72/80GiB、swap2GiB，辅助3/4GiB、swap0及CPU分离不变。
预计新增32GiB：完整500对象物化约13GB，余量覆盖临时文件、运行和日志；
复用原生 checkpoint、环境和编译缓存，无新增模型转换。启动前重新检查
150/100GiB磁盘门槛和内存剩余增长量；运行期间不扫描、压缩、清理或备份。

仅运行一次。实际终态/物理释放 → 精确关闭本轮辅助域 → 恢复本轮loader →
按实际 Invocation 停远端并保留证据 → `full4000_w2`严格分析 → 全请求
输入/原生绑定/时间/GPU积分/双端传输核验 → 封存和推送。
失败或未达95%均保留；异常结果按证据核查必要适配，不为胜负重复或筛选。

本轮后仍有其余7B W2基线与参考、同候选W0兼容/G2、3B全比较、dLoRA B6、
A1–A5、S1–S13、成本/CE与两轮审稿文档、最终学术图表。当前不画图，不将
已有W0/W1观测宣称为跨场景最优或全部目标完成。
