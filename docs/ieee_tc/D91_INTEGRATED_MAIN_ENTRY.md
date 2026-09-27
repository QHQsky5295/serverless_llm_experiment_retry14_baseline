# D91 — 完整回放入口整合，不是性能实验

## 研究问题与边界

D90 两模型各 100 请求已贯通真实准备、请求和资源释放，但该诊断驱动未提供
开放回放标签，全局接纳被限制为单副本容量。它不等于正式入口，不能据其排队
值归因整个 Full。下一项仍是既有 4,000 请求的 Full 开发回放；不重做已完成
的 D78/D80/D81 远端发布/覆盖或 D88/D89 实测初始化，不恢复 baseline。

本次不是一次新的控制参数选择：D88 parent configuration、D89 两类实测
profile、IEEE 九式、W/β/δ、tier/admission 策略、500 工件和原始 trace 保持。
只消除主入口与已验证组件之间的不一致。所有 CPU 检查均不冒充真实推理、
资源护栏资格、数值 adapter 区分性、SLO 或 G1/G2 领先证据。

## 第一性原则依据

请求的发生时间和服务接纳是不同边界。vLLM 文档也指出并发限制会使实际执行
速率低于设定到达率；0.30.0 源码在限流包装中单独保留客户端排队，不能把
它隐去或用后端提交时刻重置用户计时。
[官方 benchmark 文档](https://docs.vllm.ai/en/latest/cli/bench/serve/#--max-concurrency)、
[固定版本源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/benchmarks/serve.py)。

因此主入口以已核验的 ExternalReplayIngress 为到达合同，而不是靠“Azure”
等数据来源标签推断。服务端继续以实际副本数×单副本容量做背压；不取消物理
容量限制，不强加未来到达，也不改变 IEEE routing/admission。没有外置
回放器时旧行为不变。此设计不声称已证明 D90 所有等待都由并发上限造成。

已冻结输入的职责是提供请求映射和工件内容身份，不是再次生成数据。在完整
入口中发现 legacy setup 仍可能做模型结构探测、support-file 补齐、payload
扫描/复制/生成，以及无条件原始数据集初始化。IEEE 路径现在只读内容清单和
经同一清单 SHA 校验的微小 PEFT config；真实 payload 仍由请求/合法准备
通过已发布远端获取。未知 ID、错误 metadata、缺失远端或动态打包路径明确
拒绝，不能回退本地。非 IEEE 工件路径不变。

## 实现与检查

| 项目 | 检查内容 | 不代表什么 |
|---|---|---|
| 冻结工件装配 | 本地无 weight 也能只读装配；拒绝未知/重复 ID、config SHA 错误；禁止原始模型探测、修补、下载 | 不等于本次实际 HTTP 获取已发生 |
| 冻结负载复用 | 有 shared trace 时不构造 WorkloadDataset；仍由原 strict trace loader 校验；原始生成分支保留 source validation | 不重命名为新独立 trace |
| 接纳容量 | 外置回放对象决定开放到达；1/4 副本容量按同一个实际模型 cap 推导 | 不等于吞吐提升已测得 |
| Full 启动合同 | pending 初始 owner、未初始化 vLLM descriptor、TP1 独立子进程、实测 profile 精确 runtime identity、实际 movement/HOST/admission binding | 不等于模型已启动或最终数值正确 |
| 回放/生命周期绑定 | 已启动 ingress、完整 source count、60 秒共同准备通知、同请求/时钟的 PhysicalGPUDeployment；不注入模拟等待 | 不等于外置护栏在这次 CPU fixture 中真实运行 |
| 原有隔离 | IEEE 分支仍不调用 legacy warmup/priority；不能用 `qualified=true` 配置越过缺失 owner | 不替代正式共同指标资格 |

`_require_ieee_full_qualification` 从无条件拒绝替换为上述可执行前置条件。
其回执显式 `formal_comparison_qualified=false`；实际 native 进程、准备所有权、
释放和源身份继续由原运行时逐事件核验。没有删除这些运行时检查。

## 主入口装配表

复用 D89 manifest 指向的原 D88 parent，不把已解析的 child 当作 parent。
实际调用 `_main_async_impl`，在 `preload` 入口执行真实合同检查后主动停止，
不启动模型。测试仅在临时目录组装配置；真实输入不变。

| 模型 | 原请求数 | 原 adapter 数 | 单副本 cap | 四副本 cap | child 与已测合同 |
|---|---:|---:|---:|---:|---|
| 3B | 4000 | 500 | 8 | 32 | 完全相等 |
| 7B | 4000 | 500 | 2 | 8 | 完全相等 |

CPU 装配的 ingress 状态和通知是明确标记的 fixture；跳过真实 GPU 空闲检查，
mock 禁止模型初始化、spawn、下载和原始数据集访问。不能把这些 fixture 日志
混入请求时延、资源积分或真实传输次数。

最初 4 个反例测试出现 2 failure/3 error（有一项包含两个子用例），原日志
保留。对应修改后通过；首次综合回归 807 项仅一项失败，是旧单元测试精确匹配
已被替换的无条件错误文案。该用例仍检查拒绝和 legacy stack 未启动，改为
核对新的缺失 owner 原因，不放宽生产行为。最终回归和 final-source 装配的
精确数量、运行时间、源码/输入/日志 SHA 见配套 curated JSON。

## 下一项与未完成项

这不是 Full 已完成。实际 4,000 请求启动前尚需检查并完成：

1. 主入口每请求的有限保护从**计划到达**计时，而非接纳后重新计时；开发
   合同采用既定 1,800 秒，正式阈值仍未冻结。区分请求 deadline 与整轮中断，
   并保留原生取消/所有权闭合，不能将整轮取消伪装成所有请求 timeout。
2. 主入口异常时保留 `_interrupted_replay_evidence`、remote UUID、机制状态，
   不仅依靠 normal result 写出；物理 ledger 仍在外层 finally 完成部分观测。
3. 运行专属 HOST/NVMe 工作区的释放与清理回执，不能误删历史或关闭未证明
   已释放的 native 引用。当前实际集成尚未运行。
4. 使用原 scope、native 环境和远端服务，在新 run key 中先做 3B 完整开发
   回放；清理、校验、表/图和解释后再做 7B。配置文件是准备产物，不是已运行。

共同 warm SLO/Resident 参考、baseline、M1/M2、消融和敏感性仍未完成。既有
多数零权重工件限制仍保留，不新增权重、不以 token 数检查代替数值区分性。
