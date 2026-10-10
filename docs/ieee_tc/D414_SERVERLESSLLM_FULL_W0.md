# D414：ServerlessLLM 修正测量入口后的完整 W0

2026-10-11，启动前登记；04:46:28 启动，05:54 已完整结束并释放 GPU。
以下启动合同保留；最终结果与证据边界见文末，不重启本轮。
本轮只处理 D413 已经 CPU 验证的入口连接生命周期，不改变被测系统策略。
基线执行版本 `35dda117faecee5375c146a2c23b16fc2562d82a` 已推送并核验。
主仓库执行版本 `2860b53740bfea5fc8afae7e2139a13e2855a50f` 已推送并核验。

## 研究问题与保留证据

D411 完整回放的三个入口失败使其不能进入 G1/G2 排名。它仍保留全部
4,000 请求和资源记录，不用 3,997 个成功请求代替完整工作负载。
D413 真实 HTTP 事件顺序测试支持“额外入口复用已关闭连接”的故障机制，
但没有历史 socket-ID，不能宣称唯一证明三次历史失败的原因。

本次使用 `single_request_loopback_no_retry_v1`：额外入口向原生 API 的
每次 POST 使用新连接，无自动重试，连接开销照常计入。不是 ServerlessLLM
调度、缓存或扩缩容的优化，不改变公共 SLO 或迟到请求分母。
8 项入口测试及 92 项计量/启动/调度测试已在 D413 通过，不重复短 GPU 检查。

## 冻结执行合同

- 原 7B W0 的 4,000 请求；source42/development41，非独立留出正式运行。
  源 trace SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。
- 500-ID 原工件集合，实际回放出现 132 IDs，目标输出合计 458,224 tokens；
  `fixed_length_greedy_v1`、`role_lines_v1`，不生成新 trace 或权重。
- 共同部署通知提前 60 秒，原 open-loop 到达窗口 3,963.908903 秒，
  资格保护 1,800 秒；D394 共同 7B TTFT/TPOT 阈值不变。
- 与 D411 相同原生 vLLM 0.10.2、FP16 store、RR、已授权 polling 和
  request-identity queue 修正；min1/max4/target4/keep10、maxseqs4、
  maxLoRAs4、CPULoRAs4、tokenbudget1024、GPUfraction0.72、非 eager。
- 冷的本轮 LoRA 缓存；保留原生编译磁盘缓存并记录身份，不等于 GPU/HOST/KV
  已预热。编译缓存 3,858 文件/104,734,877 bytes，聚合 SHA
  `5a75a00e9e341e6ee53e5afb2195c4a193a010c6d3446adcd7c4dfc10ca50791`。
  所有实际启动、capture 和物理持卡均计入。
- 用户再次批准的远端一次性只读交付缓存已在 D78/D80 完成；复用，不重复发布。
  请求中没有打包；真实读取、传输、竞争、本地物化保留，不事后减延迟。
- 服务 72/80 GiB、swap2 GiB；辅助 4 GiB、零 swap；既定 CPU 范围。
  推理机启动至少 150 GiB 空间、102 GiB 可用内存，增长预估 32 GiB；
  运行停止门槛和独立监控不变。一项重型实验，业务期无远端管理或大扫描。
- 单次新执行身份，保留成功/失败，不挑选重复、不出运行间 CI。
  原始未知数值正确性字段不追改；复用 D406 全池功能和 D409 有界参考证据。

## 来源和安全准备

原始结果新目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d414_20261011`。
只复用 D411 启动脚本并修改本轮路径。配置字节与 D411 相同。
新 source view 与 D411 只差路径、helper SHA、source-adapter SHA；原生
测量后 app/backend/router/scheduler 文件完全一致。
helper SHA `2a9ebad1f75c337fe196cca8907c9a6e1001b0911aa54b877821f1727b187e3a`；
source-adapter SHA `7726f321d632beca03399faf0e30ce1288f13f8785f0ccf80c7d02327e32f36e`。

D411 已结束下载对象经来源/内容、不同 inode、进程及引用核验后仅回收
`/tmp/tc-d411-7br/remote_artifacts/objects`，共 1,320 文件。
观察释放 3,305,664,512 bytes，余量恢复至 161,526,784,000 bytes。
原工件、唯一权重、所有原始记录和远端不可变缓存均保留。三个无法读取的
同用户进程项已在清理回执披露。回执在 D411 raw 的 cache_cleanup_before/after.json。
不为了重新核验已通过的字节检查而重新下载。

既有 loader-only 安装器（只迁移官方 store loader 版本位置，不引入其他策略）：
`/home/qhq/relayserve_serverless_llm/scripts/relayserve_v4_3_apply_serverlessllm_vllm_store_overlay.py`，
SHA `da6a14276f92186457c44c215d45d1ca9d18f7bbea9cdec6665737743f9c5978`。
用本轮 `overlay7b_repaired/backups` 和 `install.json`，不能使用旧轮恢复回执。
实际 GPU 释放后按本轮 receipt restore；不在运行中改变环境。

## 结束与后续

启动时本轮在 `tc-d414-7b` 运行；现已结束，不重启。实际入口已记录新连接规则；
远端启动前链路 1 Gbps/full，缓存协议不变。部署通知和业务起点分别为
1172460.705107463、1172520.705107463（本机 monotonic 秒）。
实际服务与远端 InvocationID 已写入 EXECUTION_STATUS 和原始回执。
最初资源样本没有 high/max/OOM/swap 或保护中止，后续以持续监控为准。

完整终态 → 实际 GPU 释放 → 恢复本轮 loader → 按 InvocationID 停止本轮
远端服务 → 保存实际 worker、resolver、远端传输/资源证据 → 原 analyzer
核验全部 4,000 请求、原生时间、内容和物理 GPU 生命周期 → 保存和备份。
出现失败显式走失败分析，不绕过成功断言、不隐去请求。

PrimeLoRA D371/D397 继续封存。完成本轮后返回 7B vLLM 比较，不继续
ServerlessLLM 的局部参数搜索，不跳到 3B。有效 Resident 参考、W1/W2/G2、
其他基线、3B、A1–A5/S1–S13 和论文图表仍待完成。

## 终态结果：完整单轮观测，不是全目标完成

原始 4,000 请求全部到达、提交并返回，无失败或重试；实际原生输出
458,224 tokens，与逐请求目标完全一致。现有分析器核验了全部请求的
source/prompt/adapter/content 绑定和原生时间，E2E、TPOT 与原生重算最大误差
均为 0 ms。沿用共同 D394 阈值，不使用运行时 CE 或另设 SLO。

| 指标 | 本轮观测 |
|---|---:|
| 请求数／通过原生合同核验数／失败数 | 4,000／4,000／0 |
| TTFT 达标数 | 3,929／4,000（98.225%） |
| TPOT 达标数 | 3,996／4,000（99.900%） |
| 联合达标数 | 3,925／4,000（98.125%） |
| 物理生命周期 GPU-s | 16,089.762743524276 |
| GPU-s／通过合同核验请求 | 4.022440685881069 |
| 平均／Type-1 P95 TTFT | 973.686780／1,052.246361 ms |
| 平均／Type-1 P95 TPOT | 26.736734／32.929275 ms |
| 平均 E2E | 4,000.733946 ms |
| 平均提交后 dispatch wait／service TTFT | 533.644773／437.193547 ms |

物理四卡原生 store 池按真实持卡计量，四个物理 lease 均关闭；5 个实际服务
epoch 不代表 5 张 GPU。互斥窗口 GPU-s 为：业务前 208.223009501、到达窗口
15,855.635612488、drain 3.567444200、收尾 22.336677336，总和与积分一致。
不把这一固定原生 store 部署的持卡方式泛化为所有 ServerlessLLM 部署。

实际 132 次已发布 gzip HTTP 交付，131,480,060 线上 bytes，3,300,789,780
已核验物化 bytes，3,868 次请求复用本轮缓存，按请求打包次数为 0。
所有 transfer 与请求、收据、远端日志、内容 SHA 对应；读取、传输、竞争、
物化与加载均保留在真实时间线，不事后减去任何累计 span。

观察解释：本轮完整原生合同和共同联合 SLO 门槛通过。前 500 请求的联合
达标为 446/500，后续七段为 485/494/500/500/500/500/500（各段 500 请求）；
这是包含启动影响的事后描述，不删除启动段，也不把它们当独立重复。
本轮入口零失败不等于唯一证明 D411 三次失败的历史因果；D411 原始失败
及 D413 连接生命周期证据继续独立保留。

正确性范围：原始 `n_correct=null`、独立数值正确性与旧 qualification flags
不追改。复用 D406 全 500-ID 功能覆盖和 D409 有界 HF/PEFT 参考；这不等于
逐请求独立数值证明。7B 工件主要为零／重复权重的局限继续披露。
跨系统排名仍须逐字段 join 输入、生成、初态、计量和资源合同；不因本轮
`observed_feasible=true` 自动宣布完整 M1/M2、W1/W2/G2 或跨场景最优。

## 收尾、审计与备份依据

- 外层 service 返回 0，`native_gpu_context_release_confirmed=true`、
  `service_path_removed=true`；新鲜 GPU 进程检查为空。
- 本轮 loader 用自己的 receipt/backups 恢复，状态 RESTORED；不是旧轮回执。
- 按本轮 InvocationID 停止两个远端 artifact 服务及 monitor，确认 inactive、
  MainPID=0、端口不存在。空的本轮辅助域也已停止；没有终止无关进程。
- 实际 worker/raylet、scheduler、resolver 收据与日志保存至本轮独有目录。
  远端传输日志 SHA `bb861518a2593609d0a438e5e01a3017765550bb42dab053f6e10d301029b71f`；
  远端监控 SHA `9e677bde660c022fa6eeee820b7b39870d4ae6d003f8e728083bf9eeffc8078e`，
  拷贝后与远端一致。
- 服务内存峰值 57,570,185,216 bytes，high/max/OOM/oom_kill/swap 均为 0；
  主机最小可用 55,076,528,128 bytes。旧结果和用户修改 147 项前后零变化，
  `protected_after.json` SHA `6b4deabad0cf41b4f928ab8d0069edd8ae445128fb75786a9739f4e9baf11a70`。
- 复用现有 `audit_d402_serverlessllm_path.py --population full4000` 的严格成功
  路径；4 GiB／零 swap／辅助 CPU 范围内离线审计通过，23 项既有 CPU 测试
  通过（0.095 秒）。没有修改运行代码或统计规则。
- 新产物目录 `paper_results/ieee_tc/baseline_audit/d414_7b_full_w0/`：
  `audit.json` SHA `714a2fdb01b2d5a0e2e6f3a9b0faf7cb23b29632efbf5b343822b7dfe4b7dd74`；
  `evidence_manifest.json` 绑定 237 个保留文件、399,017,519 bytes，SHA
  `8912ab5259d7b10e990ab3b6233b789ce4432c0af2c1544498b7587b700904fd`。
  此清单生成时未回收本轮下载对象；后续若清理，另存回执，不追改清单。
- 冻结 V1／单轮 V2／D394 SHA 与启动合同不变。数据仅一轮描述性观测，
  不生成运行间均值、CI 或显著性结论，不为更好结果重复同键。

下一步：备份此证据节点，然后恢复 **7B vLLM** 主线，依据 D412 的复用审计
修正必要计量，不继续 ServerlessLLM 局部调参或重复短 gate。新 GPU 任务前
重新检查 150 GiB 启动门槛；只能回收已经核验、无引用的可重建下载对象。
PrimeLoRA D371/D397 保持封存；完整比较、Resident 参考、W1/W2/G2、3B、
消融、敏感性、CE 补充及最终图表仍未完成。
