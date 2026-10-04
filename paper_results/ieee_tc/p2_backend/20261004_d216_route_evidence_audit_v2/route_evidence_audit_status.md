# D215/D216 路由证据审计修正

日期：2026-10-04。该审计只读取已完成的 100-request 结果，不是新的性能
运行，也不改变原始 JSON。

## 发现

原始 D216 运行级协调状态报告 collections=106，但请求证据中只有 83 个
不同的 collection_id 标签；D215 分别为 107 和 87。D215 有 7 个标签、
D216 有 9 个标签出现相互冲突的每波计数。原因是旧路径在异步 native RPC
完成后才从全局计数器取 ID；并发的另一采集可能已经推进了计数器。因此标签
不能被解释为唯一采集波次，快照行数、标签去重后的行数和累计字段之和也不能
当作 RPC 总数。精确 RPC 总数记为未知。

代码已作最小证据修正：采集波在创建时固定自己的 ID，并在所有等待者返回时
复用该 ID；没有改变路由、缓存、admission、九个 IEEE 公式或运行配置。新的
并发回归测试验证两个不同 scope 得到不同 ID、同一 scope 的 joined waiter
共享一个 ID。历史运行身份不追改。

## 按冻结 Type-1 规则重算

| run | 请求 | 观察到的 source snapshot | 观察标签 | mean control (ms) | Type-1 P95 (ms) |
|---|---:|---:|---:|---:|---:|
| D215 | 100 | 304 | 87 | 236.004 | 945.799 |
| D216 retry7 | 100 | 337 | 83 | 238.515 | 690.333 |

D216 相对 D215 的均值方向为 -1.064%（lower-is-better，未改善）；Type-1
P95 方向为 +27.011%。该单次 bounded block 没有 CI，也没有数值 adapter
正确性、共同 warm SLO 或 G1/G2 资格。

按运行内报告的 fallback 条件分组只是诊断，不是因果对照：

| run/group | 请求 | mean control (ms) | Type-1 P95 (ms) |
|---|---:|---:|---:|
| D215, reported fallback | 69 | 295.495 | 980.875 |
| D215, no reported fallback | 31 | 103.588 | 354.161 |
| D216, reported fallback | 67 | 306.126 | 813.694 |
| D216, no reported fallback | 33 | 101.243 | 366.122 |

D216 的高开销人口仍由 fallback 请求主导，且 D216 在 4-replica source
snapshot 上有 76/100 个请求，而 D215 为 65/100；D216 的
scaleup_affected 也为 51/100，D215 为 39/100。这说明 bounded run 的
动态 membership/scale-out 人口不同，不能把较低的尾部直接归因于 scope-local
复用。下一候选必须先说明如何在不使用未来需求的情况下减少 fallback 或
控制采集波次，同时记录 membership 与 scale-out 人口；不得盲目重跑。

## 结论

D216 的机制路径和生命周期审计仍通过，但候选继续撤销，不授权 4,000 请求、
3B 或 baseline。原始 D215/D216 curation 保留；本目录的 JSON/CSV 是修正后
分析产物。没有生成性能图：当前问题是资格/证据有效性，状态表比连续排名图
更诚实。

源文件：

- d215/route_evidence_audit.json
- d216/route_evidence_audit.json
- d215/control_request_groups.csv
- d216/control_request_groups.csv
- scripts/analyze_control_path_overhead.py 的 route-evidence 模式
