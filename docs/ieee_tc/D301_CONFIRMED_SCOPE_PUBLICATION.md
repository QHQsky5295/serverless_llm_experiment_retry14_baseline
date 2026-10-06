# D301：准入等待分解与确认状态发布修正

## 主线与已复用证据

7B 优化仍未完成；3B、外部 baseline、正式比较、消融和敏感性继续等待。
本次不画图，不重跑 warm/source/numerical 参考，不修改指标协议。
离线复用 D299、D300 各 4,000 条完整请求，不新增性能运行。
分析入口为现有 `analyze_d297_full.py --admission-wait-output`，产物在
`paper_results/ieee_tc/p2_backend/d301_wait/validated1/`，逐输入和输出 SHA
见该目录 manifest；无覆盖旧结果。

| 同一请求上的平均区间（ms） | D299 | D300 |
|---|---:|---:|
| 计划到达→服务收到请求 | 177.064780 | 196.073016 |
| 服务收到→入口放行 | 28.006447 | 33.577356 |
| 入口放行→选中来源保护完毕 | 481.270979 | 526.683647 |
| 准入前合计 | 686.342207 | 756.334019 |
| 其中路由观测/选择（嵌套，不能再相加） | 201.760235 | 214.665988 |

第三项包含副本选择、容量等待和来源保护，不是纯队列时间。
入口使用 external open-loop 的实际 runtime 容量，而非 YAML 中的全局
concurrency=2。不能据此盲目放大并发或把所有等待归因于扩缩容。

D300 的最终路由视图中，3,178 条带 authoritative read 的请求平均路由
267.622156 ms；822 条无此读取的请求平均 9.927908 ms。D299 对应为
3,198/802 条、249.409147/11.758713 ms。这是系统决定的条件人群，包含
不同请求、状态和重试，**不是因果差值、独立重复或准确 RPC 总数**。
每条请求的三段和经过 1 ms 恒等式检查；P95 不相加。

## 历史、论文与外部实现核对

- D200 已排除“所有查询等待都是 inventory CPU”的归因，原生开始前等待
  最大；D214/D215 保留选中来源复核；D289 (`2b54c69`) 增加确认身份事件。
- 本次发现 D289 的 `_ieee_publish_identity_frontier` 在每次发布身份时
  无条件清除该副本所有 scoped cache。生产调用顺序是先由 collector 放入
  刚读取的权威 scope，再发布身份；所以同一新观察立即清掉了自己。
- 旧 collector-only 测试没有执行后面的 publication，因而漏掉这个组合。
  新测试按生产顺序重现，两项在未修正实现上失败，日志原样保留。
- IEEE 论文要求同一 received view、选择后 reservation/source recheck；
  不要求销毁仍属于同一确认身份的已测 footprint。本次不改变九式、服务
  分桶、Q 排序、需求、planner、admission、扩缩容或生成合同。
- [vLLM 0.30 官方 LoRA worker manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
  区分已加载对象复用、加载新对象与容量驱逐，并说明实际修改位于单线程
  core。它并未实现 Prime 的 epoch 协议，不能冒称直接证明该修正正确；
  借鉴的是让失效跟随实际身份变化，而不是每次读取都失效。
- [Ray 官方 autoscaling 文档](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html)
  区分 ongoing soft target 与硬入口上限。本项目不移植其推荐比例，也不
  因存在排队就修改 IEEE 的控制式；本次先处理可重现的重复状态读取。

## 唯一候选与不变量

发布 identity 时只保留与该观察的 owner、clock、epoch、完整 source identity
一致的 `NativeSourceSnapshot`。比较只忽略观察捕获时间和 footprint 投影；
不猜测字节，不跨 epoch 延长 footprint 有效性。不同身份或畸形条目仍丢弃。
其他副本不受影响；旧 epoch 不能覆盖新 frontier，同 epoch 身份冲突仍报错。

选中后的 GPU/HOST/file 来源验证、原子引用、冲突重选和取消释放路径完全
不变；仍保留原先只用于候选视图的已确认观察语义，不声称所有副本同时锁定。
本次不是 TTL、LastKnown 新工作点、另一套 admission，也不是调参扫描。
后续 LastKnown 消融和状态传播评价仍须按完整计划单独完成。

## 测试与资源记录

未修正代码：2 个针对性测试失败，证明 scope 发布组合缺陷。
修正后：369 项状态/footprint/引用/生命周期测试通过；加入等待分解校验
后最终 371 项通过。已有 pytest 配置/弃用警告保留，不影响通过状态。
新增测试还覆盖身份变化、旧事件、其他副本隔离、同 epoch 矛盾、缺失或
非法测量及分解不一致；这些是逻辑资格，不是性能或数值正确性资格。

所有离线任务限制为 CPU 2,3,26,27、MemoryHigh/Max=3/4 GiB、swap0。
初始检查/故障复现/回归/正式离线分析/最终回归峰值分别为
999145472/615145472/680083456/750960640/680128512 B；high/max/OOM 均零。
已核验对应 InvocationID 和空 cgroup.procs 后仅停止这些自有 scope。
原始日志位于 `results/ieee_tc/p2_backend_qualification/d301_20261007/`。

## 下一项验证（D302，尚未运行）

使用既有 protected launcher 的 `d302`，7B 前 100 条原请求、同 500 池、
60 秒通知、真实远端已发布 gzip 工件、独占新输出与冷本地缓存。
模型、D298 profiles、cap8/slot8/rank16/.92、全部 D300 controls 保持不变；
不再提高缩容阈值。源 trace 不重新生成，远端不重新打包或复制整池。

修正后的代码先备份，再运行既有 100 请求 smoke；核查 100 个原生生成
合同、实际释放、来源保护/重试、远端日志、route evidence 和资源事件。
100 请求只能验证机制和早期行为，不能替代 Full4000 或外推完整 GPU-s。
通过后用同配置做完整 W0，比较 D300 的物理 U、TTFT、TPOT、等待分解与
临时 timing 上界，保留所有退化。未胜出不自动回滚正确的状态修正；先
检查它是否触发以及额外管理成本，再按主线判断。

最终共同 warm/Resident、数值资格、G1/G2 和正式 CI 均仍开放。
