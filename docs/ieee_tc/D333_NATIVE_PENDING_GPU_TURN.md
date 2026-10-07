# D333：GPU 命中路径的一次原生登记与保护

2026-10-08。7B 单一开发候选；476 项相关 CPU 回归通过，GPU 验证待执行。
不是性能结论，不是数值正确性或共同 SLO 资格。

## 历史证据与问题

D332 的完整 4000 请求平均 admission 为 596.710 ms，其中入口迟到
218.301 ms、dispatch 窗口 33.803 ms、runtime 选择/保护 344.607 ms。
这些不是纯容量排队；routing 是 runtime 内的嵌套区间。D328 已观察到
原生查询外部的等待，但不能把查询均值直接当作本候选的预期收益。

D184 已合并 controller→frontend 的 GPU 登记/保护，但 frontend 仍先向
core 登记，等回复后再向同一 core 保护 GPU 来源。D330 已发布规划实测信息，
不重复这些改动。HOST 的 pending-load 集合在登记 ACK 后、保护前更新，
被 `ReplicaRoutingSnapshot.routing_key` 消费；不能假定它是无关日志并删除。

可证伪假设：仅在已经选中 GPU 来源时，把上述两个原生操作放在同一次
core utility 中顺序执行，减少一次 frontend/core 往返，可能改善完整回放
的请求等待。其余控制器阻塞、模型执行和输入迟到仍可能主导结果。

## 官方依据与适用范围

- [vLLM 0.30 core client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)
  的 utility 请求各自发送消息并等待对应 future。
- [同版 core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  在输入处理循环分派 utility，和模型 step 共用执行循环。
- [UniProc executor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
  的阻塞 collective 调用在当前线程执行 worker 方法。

这是本地版本限定集成，不冒称官方稳定公共 API；不引用官方博客的加速
比例作为本系统结果。严格限制为已合格的 TP=PP=1、原生 UniProc owner。

## 唯一改动和不变量

1. 实际 native tokenizer 先生成原有 pending descriptor。登记与选中 GPU
   副本保护在同一 core turn 顺序执行，无中间 frontend 往返或 scheduling step。
2. 命令必须是精确 GPU source/owner/epoch/ID/path/content；登记 ACK 合法后
   才访问 GPU owner。后者仍执行现有来源比较与原子引用，无 miss 后隐式加载。
3. 这不是全事务回滚、物理 KV 预留或全副本原子快照。失败/丢失回复可能
   留有 pending 或 GPU 引用，保留既有未知状态、quarantine 和 retirement。
4. 已知来源冲突仍返回两份凭据，交给完整 router 重选。不能原地换成 HOST
   或本地加载。HOST/file/remote、其 pending-load 顺序和准入式完全不变。
5. 不改九式、D/T/O 类、队列排序、profile、autoscaler、并发、SLO、输入或
   真实远端交付。按请求保存 `single_native_core_turn_v1`，仅为触发证据。

## 验证与后续

复用 D184 测试文件及原生 scheduler/frontend fixture，验证真实方法的
wire arity、登记顺序、单次 utility、错误命令、异常 ACK、已知冲突、丢失
回复、取消时登记任务收敛以及正常撤销。控制器/真实 worker transport 的
既有生命周期回归一起执行，不把 fixture 成功写成 GPU 数值或性能合格。

随后复用 guarded runner 进行现有 7B W0 前 100 条短测；通过后原 4000
请求普通完整回放，对照 D332，保留失败和所有副作用。最多两轮有新证据的
最小验证，不重复 D328 profiling 或已完成数值块。性能目标前不画图。

计划 SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`；
V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
共同 warm、Resident、新旧 Prime 和数值资格仍开放；7B 后再 3B、基线、
比较、消融和敏感性。不重建已经批准并完成的 D78/D80 只读交付缓存。

## 本轮 CPU 与启动前结果

第一组 113 项、扩展组 476 项通过（重叠，不相加）；扩展组 unittest
执行 4.193 秒。实际原生 core/frontend 方法的 synthetic owner、控制器与
真实 socket worker 组合已覆盖；不是 GPU 输出/性能测试。3/4 GiB、swap0
和 CPU2,3,26,27 均读回，events/swap0，结束后精确核对空资源域并停止。
最初非 UUID scope 名被所有权检查拒绝，测试尚未开始；保留该启动失败记录，
改用既有 UUID 命名协议，未放宽安全检查。

D334 复用 D331 joint-loader 启动前核验：相对 D332 仅三个独占路径不同，
实际 child model 不变，160 service / 24 preparation 类均由生产 loader
读取。磁盘当时 161151070208 B，主机可用 112535502848 B；真正启动前仍
重新检查。保护清单只有已知 baseline replay 差异，用户 manifest 不改。

证据在 `paper_results/ieee_tc/p2_backend/d333_native_pending_turn/`，原始
目录为 `results/ieee_tc/p2_backend_qualification/d333_20261008/`。
下一步 D334 的 100 请求前缀，然后 D335 普通完整 4000 W0；不重复已完成的
D332 分析。两者都没有新 profile、详细采样器或局部模拟远程等待。

## D334 真实短测完成；下一项必须是完整回放

实际执行代码 `74587830983bd28fd4791ed8ffeec9192262c56d`，已推送并独立
核验远端 V2。D334 于 05:44:46+08 启动，既有 W0 前 100 个请求全部成功，
原生 token 数量和十项 request/adapter/owner/slot/content/clock 绑定均为
100/100，时间重算误差零。45 个 GPU 来源请求记录新 core-turn 凭据；18 HOST、
8 NVMe、29 Remote 未进入该路径。无兜底加载，来源分支未改写。

| 同输入前缀描述性指标 | D331 attempt2 | D334 |
|---|---:|---:|
| 平均 TTFT（ms） | 900.317 | 804.848 |
| P95 TTFT（ms） | 2290.801 | 2012.726 |
| 平均 admission（ms） | 268.511 | 204.575 |
| 平均 service TTFT（ms） | 631.805 | 600.273 |
| 平均 TPOT（ms） | 27.775 | 28.152 |
| P95 TPOT（ms） | 39.398 | 36.459 |
| 完整短测生命周期 GPU-s | 711.198790 | 618.357456 |

平均/P95 TTFT 观测下降约 10.60%/12.14%，短测 U 下降约 13.05%，但平均
TPOT 增加约 1.36%。每个条件只有一次短测，无 CI，不由此选择性能赢家或
外推完整 W0 资源。4 个请求输出 token SHA 与上一短测不同，全部保留；
传输和绑定一致不能替代数值资格，`N_correct` 仍未知。D170 旧参考下的
timing-only 为 99/100，不是正式联合 SLO。

29 次远端获取共 50650865 线上字节，两端 ID/bytes/SHA 一致，请求中打包
和临时归档为零；本次网络仍为 1000/full，不冒称实际吞吐等于 1 Gbps。
263 次资源采样：服务峰值 19400253440 B，主机最低可用 93944946688 B，
high/max/OOM/swap 为零。U 四个互斥窗口合计 618.357455818099 GPU-s，
没有未释放 GPU lease。

先确认实际 GPU 释放，再按本次 InvocationID 停止两个远端服务及监控，
PID0、端口消失、日志下载 SHA 一致；本机 replay 与离线分析域均确认空组
并停止。离线复用原 smoke analyzer，新增严格触发校验；15 项分析器测试
通过，资源峰值 91394048 B，events/swap0。没有图或新 profiling。

原始：`results/ieee_tc/p2_backend_qualification/d334_20261008/`；curated：
`paper_results/ieee_tc/p2_backend/d334_native_pending_validation/`。
17 文件闭环证据已校验归档，运行/原始测量不覆盖。下一项为 D335 原 4000
W0 普通回放，对照已封存 D332；配置已登记，不再短测、重复剖析或改参数。
启动前重新核对完整计划和安全门槛：磁盘刚高于 150 GiB，不为启动放宽门槛。
