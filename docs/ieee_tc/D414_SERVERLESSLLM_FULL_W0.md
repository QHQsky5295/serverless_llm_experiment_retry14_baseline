# D414：ServerlessLLM 修正测量入口后的完整 W0

2026-10-11，启动前登记。当前为准备状态，尚无新性能结论。
本轮只处理 D413 已经 CPU 验证的入口连接生命周期，不改变被测系统策略。
基线执行版本 `35dda117faecee5375c146a2c23b16fc2562d82a` 已推送并核验。

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

完整终态 → 实际 GPU 释放 → 恢复本轮 loader → 按 InvocationID 停止本轮
远端服务 → 保存实际 worker、resolver、远端传输/资源证据 → 原 analyzer
核验全部 4,000 请求、原生时间、内容和物理 GPU 生命周期 → 保存和备份。
出现失败显式走失败分析，不绕过成功断言、不隐去请求。

PrimeLoRA D371/D397 继续封存。完成本轮后返回 7B vLLM 比较，不继续
ServerlessLLM 的局部参数搜索，不跳到 3B。有效 Resident 参考、W1/W2/G2、
其他基线、3B、A1–A5/S1–S13 和论文图表仍待完成。
