# D318：同一候选的完整 W0 验证登记

2026-10-07，执行前登记。上一任务 D317 已完成一次 100 请求验证并封存，
不是仍在运行，也不再重做。上一备份 `ef7b6488e5e40510692dd41f97e97378ede7765b`。

本轮只检验 D316 的事件确认 GPU-copy 测量复用在完整 4000 请求上是否减少
请求路径等待，并保留 TTFT、TPOT、真实 GPU 占用和全部失败。D317 的短测
仅给安全/触发证据，不能代替本轮，不外推短测 GPU-s。

## 执行合同

- 7B，既有 4000 请求/500 adapters 的 W0，源 trace seed42，开发身份；
  真正执行完整 trace，既有 runner 明确清除 diagnostic prefix。
- D315 的全部 model/workload/control/profile 配置不变；1024-token budget、
  cap8/slot8/rank16、TP1、.92、固定输出协议、60 秒部署通知、共同资源包络。
- 同一 D298 已测 40 个 service 类和 24 个 preparation 类，不新测 profiles。
- 与 D317 完全相同的 serving 实现：SHA
  `262b8f58ca12b129d0292f3e6da94ed3f5732171f959a2d0280c89e8cda46ef6`。
  不修改 IEEE 九式、生成、超时、扩缩容阈值、冷却时间或评价 SLO。
- 原工件和真实 HTTP 远端，复用 D78/D80 一次性只读压缩交付缓存；不重建
  数据，不在请求内打包，实际读取/传输/本地准备照常计入。
- 仅更换 D318 输出与 cold cache 身份；不覆盖旧结果或旧 cache。

配置：`configs/ieee_tc/20261007_d318_7b_full_w0.yaml`，SHA
`2b2a2d408d399c8cb3798ea8628e1c816f230a9b6145c4d430e5c3182d8c1959`。
已有入口：`bash scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh d318`。
原始结果：`results/ieee_tc/p2_backend_qualification/d318_20261007/7b_full_w0/`。
结果目录已存在则拒绝启动，不能用同一个键重复试跑。

## 先验核验与解释边界

既有 prelaunch verifier 核对 D315 配置、实际 native child、全部 profile
绑定、D317 原始来源与五个最终汇总文件、同一服务代码及保护清单。新增
完整入口/路径隔离测试，连同原路由和基础 smoke 共 357 项通过（23.280 秒）。
这些是 CPU/入口资格，不是新增 GPU 性能数据；实际资源准入须临启动再查。

完成后沿用已有完整 analyzer，以 D315 完整 W0 为主要参考，逐项核对输入、
生成合同、dispatch tier、原生 token、source lease、GPU allocation/release、
真实远端 ID/字节与保护清单。全部失败和输出差异保留，不能只分析成功子集。
新分支触发从逐请求 `source_collection.identity_frontier_retained_gpu_scopes`
读取；不访问没有导出的最终 route counter，不把累计计数当准确 RPC 节省。

关注完整等待分解（routing 不重复相加）、P95/P99 TTFT、平均/P95 TPOT、
完整 U、时延条件命中和副作用。D170 仍只是临时参考，不填 N_correct，不宣称
正式 SLO/G1/G2 合格；单次开发回放不给 CI 或统计优越性结论。

一次执行后先释放本机资源；核验 remote InvocationID，再停止远端监控。
它已有的 EXIT trap 会按身份关闭两制品服务：检查最终 inactive/PID0 即可，
不要要求已停止服务仍保留 InvocationID，不重启服务补日志。

接下来的动作是完成本候选的全程证据并接受/撤销，而不是继续短测、扫描参数、
重复数值/profile 或提前转入 3B/基线。完整目标仍开放，暂不绘图。
