# D318：完整 W0 结果与资源—尾延迟取舍

2026-10-07。执行前登记见 D318_COPY_REUSE_FULL_W0.md。只运行了一次既有
7B/4000 请求 W0，未重新生成工件或负载；D315 为预登记完整参考。
候选执行版本 efa58b19dfc70df8daa8f9b94b4d78cd9ae531a2，服务实现与 D317
短测一致。结果不覆盖旧实验，尚无正式留出重复或置信区间。

## 实际观测

| 指标（低优） | D315 参考 | D318 候选 | 候选相对变化 |
|---|---:|---:|---:|
| 平均 TTFT，ms | 1304.383 | 1292.091 | −0.942% |
| P95 TTFT，ms | 2934.366 | 3031.176 | **+3.299%** |
| P99 TTFT，ms | 4610.766 | 4459.486 | −3.281% |
| 平均 TPOT，ms | 32.558 | 33.461 | **+2.771%** |
| P95 TPOT，ms | 62.152 | 64.226 | **+3.337%** |
| 平均 E2E，s | 4.896654 | 4.955792 | **+1.208%** |
| 物理生命周期 GPU-s | 15865.617487 | 15526.999223 | −2.134% |
| GPU-s/offered request | 3.966404 | 3.881750 | −2.134% |

本表变化为 `(candidate-reference)/reference`；机器 CSV 保留既定
`lower_better_change_percent=(reference-candidate)/reference`，二者符号
定义明确，不修改旧统计。per-offered 不能改称 per-correct。

在 D170 **临时**时延参考下，联合时间条件命中为 3818/4000，D315 为
3820/4000；相差 −0.05 个百分点。这不是冻结的正式联合 SLO。
N_correct、正式 G1/G2 资格仍未知，不因两者超过临时 95% 就宣告达标。

## 机制与代价的解释边界

- 平均路由观测耗时从 177.688 ms 到 141.098 ms；平均到达至 admission
  从 681.229 ms 到 652.184 ms。路由嵌套于后者，不能再加一次。
- D318 的均值分解为到达释放滞后 194.247 ms、dispatch-window 等待
  31.890 ms、runtime-slot 路径 426.047 ms；该路径包含路由与状态保护，
  不可全部命名为纯容量排队。分解重算误差为零。
- 原生事件确认复用分支已触发：15473 个保留视图中累计计数的观测最大值
  为 24216。不是总 RPC 节省数，也不是单次调用的因果延迟减少。
- 在线控制轨迹发生变化：D318 有 5 次 scale-up、2 次成功 retirement；
  D315 为 3 次 scale-up、没有 retirement。配置、阈值、profile 没变。
  因而资源减少伴随不同并发/再启动轨迹，不能把所有变化归为路由开销本身。
- 远端均为 132 次交付、131480060 线上字节，无动态打包；D318 的每次
  ID、内容 SHA、字节和远端时钟均与原始服务日志一致。未减去真实传输时间。

结论：支持“减少了一部分重复状态读取的观测开销”，但**不支持整体性能
同时改善**。保留 D316 的已测试正确实现及本取舍点，接受其机制可运行/有触发
的结果；不把它晋升为已证明优于 D315 的最优 Full，不冻结最终工作点，
不隐藏 P95 TTFT、TPOT 和 E2E 退化。该局部假设的短测—全回放链已完成，
不再反复短测或扫描阈值寻找胜出。

## 完整性、安全与来源

4000/4000 正常终态及原生 token 数匹配；十项 request/adapter/owner/lease/
slot/content/clock 绑定检查均为 4000，通过实际 dispatch tier 完整检查。
TTFT/TPOT 和阶段分解重算误差为零。同输入对照保留全部 240 个输出 token
hash 差异；绑定一致不等于证明数值正确，不能据此填 N_correct。

所有物理 lease 闭合，服务/watchdog/replay 返回码均为 0，实际 GPU context
释放且服务 cgroup 删除。4039 个资源采样，峰值 20559921152 B，主机最低
可用 93445820416 B，memory high/max/OOM 和服务 swap 均为 0。

19:43:05+08 验证远端三服务的原 InvocationID 后先停监控，其既有 EXIT trap
关闭两个制品服务；最后均 inactive、PID0、success。没有在测量中重启服务。
原日志复制后与远端 SHA 一致。本地 aux 实际 procs 空且 populated0 后停止。

离线只扩展已有 full/context analyzer 的明确 D318 身份，并复用 D317 的
计数核验函数；没有改变服务代码。359 项 CPU 检查通过（23.322 秒）。
分析 scope 3/4 GiB、0 swap，峰值 1154301952 B；资源事件 0，实际空组后停止。

原始证据位于 `results/ieee_tc/p2_backend_qualification/d318_20261007/`，
完成回执 `completion_receipt.json`；新汇总位于
`paper_results/ieee_tc/p2_backend/d318_full_w0/{validated1,context1}/`。
所有数据都带源 SHA；保护清单保留唯一已知 baseline replay 差异的 raw pass=false，
不冒称无差异。用户生成 manifest 未改动或暂存；不生成图。

## 返回主线

下一步仍是 7B 目标级优化，不转 3B 或外部 baseline。先复用 D315/D318
完整日志，核对新增 retirement/activation、准备和 decode 的时间交叠，
结合 D304–D315 已有失败/退化历史，定位剩余等待与尾部损失。关联不冒称因果；
下一次服务修改前必须核查官方论文/实现并登记单一可证伪假设。不能因为本次
出现两次缩容，就直接硬编码改变 cooldown 或放宽 SLO。

D294/D295 的三块数值重复诊断已经完成，不为未知 Full 资格无止境重开该小实验。
完整数值资格、最终共同 warm/Resident、旧 Prime 新指标对照仍开放，不能从
当前 7B 开发点推断全局目标已达到。整个目标保持 active。
