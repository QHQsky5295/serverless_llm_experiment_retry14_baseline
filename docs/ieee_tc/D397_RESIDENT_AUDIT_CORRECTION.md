# D397：Resident 参考计量复核与 D395/D396 更正

日期：2026-10-10。D395/D396 原始文件和旧提交不改写。

## 结论

两轮均完成 4,000 请求，原生输出长度匹配，远端均记录 132 次已发布 gzip
交付、131,480,060 线上字节，无请求时打包。它们是完整服务诊断，**不是合格
U_ref 或主比较结果**。D395 旧文档将代数分解一致、`tpot_observed` 和资源
receipt 的布尔字段解释为完整计量验收，结论过强，在此明确撤回。

| 保留的原始观测 | D395 | D396 |
|---|---:|---:|
| 成功/目标输出匹配 | 4000/4000 | 4000/4000 |
| 旧计时平均 TTFT（ms） | 1882.464 | 429.171 |
| 旧计时 Type-1 P95 TTFT（ms） | 3148.772 | 1720.128 |
| 旧计时平均 TPOT（ms） | 26.533 | 26.418 |
| receipt 记载的 GPU-s | 16146.827096 | 16152.820090 |

以上不是同协议合格性能点，不与 Prime 排名，不计算两轮平均，不用于 75% 预算。

## 可定位的原因

1. `scripts/replay_openai_trace.py:1383` 在首个非空 HTTP 字节块到达时设置
   TTFT，之后才解析 SSE。它可能包含 keepalive/metadata，不必是原生输出 token。
   `:1653` 在无服务端时间戳时用响应结束减首字节估 TPOT，把完成通知尾部纳入。
   原生 token 数量正确并不能证明首末 token 时间正确；0 ms 的代数分解误差也不能证明。
   未保存逐事件接收时间，不能仅离线相减恢复正确边界。
2. `DynamicLoRALoader` 和 `RemoteArtifactFetcher` 在回放进程中执行下载、
   解包与 adapter 注册。`run_vllm_fair_experiment.sh:1468` 将该进程置于辅助域，
   而计划 §2.2 要求这些服务活动归入服务包络。只把 GPU workers 隔离仍不足。
3. `scripts/ieee_tc/resident_protocol_receipt.py:init/finalize` 是服务外包络
   记录，不是共同物理 owner/lease 账本。finalize 在清点前取时间，凭显存阈值
   写入 `release_after_worker_exit=true`，没有用该字段验证进程身份和真实租约。
   终态单位退出与显存空闲是有用收尾证据，但不能由标签替代完整分配/释放合同。
4. 原 wrapper 等待全部副本 ready 后才启动回放器。D395 回放启动晚于业务起点
   128.312 s，D396 提前 14.510 s。虽然原计划时间仍在统计中，注入与排队位置
   不同，不能将两轮大幅 TTFT 差异直接归因于系统策略。
5. D395 的编译阶段约 29.54 s，D396 约 0.27 s；引擎初始化约 150.76/9.91 s。
   日志使用历史 D214 编译缓存，说明初始缓存状态需要明确冻结。官方说明编译
   缓存可减少后续启动工作；这支持启动差异的一项解释，不证明全部时延差来自缓存。
   参考：[vLLM 编译缓存说明](https://docs.vllm.ai/en/latest/design/torch_compile/)。
6. 两轮监控包含主机内存观测，但不能替代计划要求的独立 cgroup/PSI/磁盘保护
   证据。未观察到错误不能推成完整监控合同已经通过。

D395 原文把 legacy GPU-s 与 receipt 差异全部归因于 300 s 尾部也不准确：
四卡 300 s 只解释 1,200 GPU-s，不解释全部差额。旧值分别保留，不倒推释放时间。

## 新规则下的下一步

- 取消第三次重复；遵循单轮修订，不能选择较快的 D396 充当最终参考。
- 复用共同 launcher 的 owner/lease、独立回放和监控机制，保留普通 vLLM 的
  路由/cache 语义；服务工作归服务域，不引入 Prime 策略。
- 新增原生 SSE 事件计时解析器及无 GPU 测试作为准备；它尚未接入正式 runner，
  不声称计时问题已经端到端解决。多 token 事件是交付边界，不是 kernel 逐 token 时间。
- 修正后先短资格验证，再一个完整预登记参考；不得因数值好坏选择是否接纳。
- 主线先封存已有 7B Prime W0 观测，恢复 7B baseline 审计，Serverless 优先。

## 原始证据索引

两轮位于 baseline 仓库 `results/ieee_tc_resident_protocol/` 下各自完整 run tag。

- D395 replay SHA：`91f02ccb276b94125ca06c8da7923a9d741022baca06bd04b32ba32e75808a5d`
- D396 replay SHA：`36795c65b9d097aae660b87d27b72765cd174da27d657d0b3839c45a8f6590f0`
- D396 lifecycle SHA：`abe77e150c45634efd535a30169e9e62754cf4669d2b8f7c1425df558e3c21e1`
- D396 remote journal SHA：`2b38452c67b4c1f5ba72cd4210ea8bd4df88289f451c246998c652c8724704ad`

D396 执行已返回 0，workers 已退出，四卡各 15 MiB。远端仅在核验所属 PID/命令
后停止，端口 18081 关闭。资源释放不是参考计量合格的替代条件。
