# D126：7B Full 接纳名额与生成阶段的占用

2026-09-30。复用 D125 已保存的小型请求投影、正常 outcome 和资源日志；
没有新 GPU 回放、没有修改在线策略、没有重新投影 11.51 GB 原始 JSON。
依据冻结指标 V1；此项是失败运行的原因诊断，不是 M1/M2 达标比较。
全体 4,000 个终态保留：3,456 个原生生成合同成功、524 个 TimeoutError、
20 个 RuntimeError。数值 adapter 身份与共同 warm SLO 仍待验证。

## 1. 成功请求阶段表

每行 n=3,456，单位秒；阶段 P95 不可相加。平均并发贡献的分母为整个
5,763.144352 秒观察窗口，仅分子取成功请求的对应阶段。

| 阶段 | 均值 | P95 | 平均并发贡献 | 峰值 |
|---|---:|---:|---:|---:|
| 计划到达→全局接纳 | 877.394043 | 1781.356688 | 526.149204 | 1154 |
| 全局接纳→所选源准入 | 2.761062 | 8.727256 | 1.655733 | 8 |
| 所选源准入→原生 dispatch | 2.473673 | 10.154980 | 1.483394 | 7 |
| 原生 dispatch→末 token | 3.876349 | 9.189747 | 2.324540 | 8 |
| 末 token→控制器完成 | 1.629483 | 7.791105 | 0.977157 | 8 |
| 控制器完成→外层终态 | 0.697782 | 2.364712 | 0.418441 | 8 |
| 全局接纳→外层终态（后五项上包络） | 11.438349 | 26.477113 | 6.859265 | 8 |

外层终态位于实际 gate release 之后，所以最后一行是占用上包络，而非已经
单独测量的精确名额释放区间。控制器完成和外层终态都不冒称客户端收到响应。
请求占用不是物理 GPU-s，也不是 GPU kernel busy time。

观察窗口内 5,687 个资源样本的持卡 GPU utilization 样本均值为 35.853719%；
5,574 个样本有前置积压；重建的成功请求 gate 上包络在 4,454 个样本为 8。
这些是离散描述，不能据此断定 GPU 持续空闲、已饱和或所有失败工作均未提交。

## 2. 积压与失败恢复的先后

以业务开始为零点：首个失败终态 3,366.880645 秒；对应异常收集边界
3,366.969737 秒；首个 quarantine 4,490.755947 秒。

| 时段 | 控制样本 | 平均 active | 平均 queue | queue>0 | queue>0 且 active<capacity |
|---|---:|---:|---:|---:|---:|
| 首个失败之前 | 1160 | 6.638793 | 575.069828 | 1121 | 696 |
| 首个失败后、quarantine 前 | 375 | 6.701333 | 1222.189333 | 375 | 253 |
| quarantine 后 | 476 | 3.724790 | 489.502101 | 475 | 114 |

首个失败之前 1,135/1,160 个控制样本的 ready capacity 为 8；quarantine 后
133 个样本的 ready capacity 为 0，不等于此时物理 GPU 已释放。
active 仅为当时可路由 slots 中绑定的请求，不是全局 gate 或原生运行请求数。
不能用它减去上表成功子集的 native occupancy 推导所有非生成工作的份额。

## 3. 观察、解释与下一步

1. 积压先于请求失败与后续隔离恢复。因此，后续恢复不能单独解释早期排队。
2. 成功请求在准备与完成收尾中分别占用服务机会；原生生成仅是其中一段。
   不能用原生每请求约 3.88 秒直接推算整个系统服务率。
3. 当前 runner 的全局 gate 在 `_exec_request` 及原生 pending/reference 的
   确认释放期间持续持有；释放确认不能跳过。提高 cap 或 timeout 没有因果依据。
4. D124 计时修正已由 D125 的全部成功请求验证，但复制候选不足以使 Full
   合格。D126 不重测或重新接受这一候选，不以单次成功比例变化证明因果收益。
5. 下一项只检查一个有证据的控制路径 CPU 瓶颈：结合 D123 已有调用栈，定位
   RPC 解码/原生状态验证与准备计算是否阻塞请求推进。先验证同输入、同校验的
   组件效应，再决定是否进入一个新版本的普通 Full，不盲跑相同配置。

本次源码核查已确认 `_send_rpc_on_channel` 在主事件循环同步解析完整 JSON，
而原生状态在不同路径还需验证 storage union、共享边、epoch 和物理预算。
旧 D123 的 13.44% RPC、11.25% footprint 持 GIL 采样比例仅定位候选，不是
D125 wall-time 占比。不能直接用这两个比例预测当前收益。

[Python asyncio 官方说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
解释同步 CPU 工作会延迟同一事件循环中的其他任务；
[vLLM 的性能分析](https://vllm.ai/blog/2024-09-05-perf-update)与
[v0.30.0 序列化实现](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/serial_utils.py)
提供 CPU 路径隔离和高效消息编码的设计参照。它们不证明本机的具体主因；
本次尚未实施线程、进程、codec 或 snapshot cache 变更。

## 4. 复用、验证与资源

未修改 `scripts/analyze_control_path_overhead.py`，使用其已有
`--native-timeline --allow-failed` 接口。四个 CSV/JSON 保留所有 offered ID；
失败的默认零阶段为空，不造零等待。原始日志、旧图与失败证据均不覆盖。

分析耗时 2.60 秒、峰值 RSS 259,048 KiB、exit 0；实际资源域 3/4 GiB、
swap 0、CPU 2,3,26,27，high/max/OOM 等事件为零。
唯一分析 invocation `604c6eaf67584cbb83270b5c16c35003` 于 20:54:38
确认无进程后关闭。没有 GPU/远端服务或后台分析遗留。

结果：`paper_results/ieee_tc/p2_backend/20260930_d126_control_occupancy/`。
summary SHA256：`e366042d59a8f55be335792641aac636e568bbed9a4f2300467c441cf320f21f`。
脚本/回执：`results/ieee_tc/p2_backend_qualification/d126_20260930/`。
完整来源与 D125 状态见 [D125](D125_FULL_W0_FULL1.md)。
此项按计划 §11 使用诊断表，不画正式优劣排名或单次 CI。

主线仍为 Prime Full → warm/Resident → Serverless → vLLM → S-LoRA →
dLoRA 3B → Loquetier → HydraServe → M1/M2 → A1–A5/S1–S13；基线仍暂停。
一次性交付缓存已完成，不能因本轮授权重申而重建。
