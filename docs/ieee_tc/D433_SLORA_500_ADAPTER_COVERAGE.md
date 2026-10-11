# D433：S-LoRA 7B 全工件池功能覆盖（预登记）

2026-10-11。接续已封存 D432 的100请求实际路径资格，不重复该运行。
本次唯一问题：原生服务是否能正确绑定并执行全部500个既有逻辑 adapter ID。
这不是完整 W0、资源工作点选择或与 PrimeLoRA 的性能排名。

## 输入与不变条件

- 复用现有 `adapter_coverage_v1`：源 trace 前500条 arrival/prompt/target，
  仅将 adapter 映射为冻结池排序后的500个已有ID；保留原/派生身份。
- 500请求、500唯一ID、68,925目标tokens；view SHA
  `286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d`。
  不生成新负载、权重或工件池；500逻辑ID不代表500种独立权重。
- 完全沿用 D432 的 `slora_7b_resident_native_v1.json` 和 D430 source view：
  四个TP1副本、RR、顺序初始化、swap、原生调度和D428等价HOST表示。
  不改原生kernel/paging/批处理，不搜索参数，不以短验证SLO低为由重复运行。
- 使用共同60秒部署通知、1800秒资格保护、fixed_length_greedy_v1，
  原生token/后端时刻及D394共同SLO。业务不移至系统ready之后。
- 全静态500池在通知后获取并准备；复用D78/D80已发布只读gzip，不重新发布。
  真实远端读取、传输、竞争、本地准备全部保留；请求路径打包应为0。
- 原生Triton磁盘缓存另行封存；GPU/HOST adapter不预加载。
  服务72/80GiB、swap2GiB；辅助4GiB/零swap，单重型任务与持续安全监控。

## 唯一输出与验收

raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d433_20261011`；
私有物化：`/tmp/tc-d433-7bs`；预测新增峰值20GiB，启动仍需≥150GiB。
复用D432启动、远端管理、输入封存和收尾脚本，仅改campaign路径与覆盖视图。
curated：`paper_results/ieee_tc/baseline_audit/d433_7b_slora_coverage/`。

验收500/500 offered/submitted/terminal/native绑定及完整原生数量/计时，
请求adapter集合等于冻结池，四声明副本实际服务，500唯一部署传输/内容/receipt/
服务器记录关联，真实GPU分配至退出/释放闭合。全池准备不能替代全ID发请求。
准备传输只计一次，不因500条请求引用而重复计算。保留未知原始数值资格字段。
覆盖视图改变了adapter映射，不能直接套用D385原100请求的独立数值参考。
沿用D431严格 `slora_resident_rr_v1` + `adapter_coverage_v1` 分支；
D432离线summary修正已通过37项测试且未改变运行路径，直接复用。

结束顺序：实际GPU释放→按本轮身份停止远端→保存原始证据→严格校验→备份。
必要时仅回收有保留源和引用核查的私有下载副本，保护历史结果及用户修改。
随后原1000请求兼容验证→一次完整4000 W0。已封存PrimeLoRA、ServerlessLLM、
vLLM不重跑；不跳3B。W1/W2/G2、其余基线、消融与敏感性仍待完成。
本登记时尚未启动GPU或远端服务，不能声称覆盖已通过。

21:55:32实际运行已启动，tmux tc-d433-7b；尚无终态结果。
执行main627aebff3fdaf67822946e52eacb7c9291f2eb30、baseline
15077ab4aeeac9532b1cb8d9f18567d76883163c，均已推送并核对远端SHA。
37项CPU回归通过1.636s；500-ID输入、源码及147旧结果保护项核查通过。
远端实际协商1000Mb/s/full；复用已发布缓存，无重新生成或发布。
当前持续观察同一次运行，不在业务期间改配置、测试、哈希扫描或清理。

## 终态与严格审计（2026-10-11）

22:07:37前已结束并释放GPU，不重启。500 offered/submitted/terminal/native/source/
adapter合同全部通过，恰好500实际ID覆盖冻结池；四个声明副本均实际服务。
68,925原生tokens，零失败，E2E/TPOT/native重算最大误差均0ms。
500次部署准备传输共420,740,410线上字节、12,985,984,450验证物化字节，
请求时打包0；请求仅引用已准备工件，不能把这些引用当成额外500次传输。

| 覆盖验证的描述性观测，不是W0排名 | 数值 |
|---|---:|
| 平均 / P95 TTFT | 62,511.640 / 279,292.720 ms |
| 平均 / P95 TPOT | 38.253 / 53.249 ms |
| 平均 E2E | 67,703.881 ms |
| TTFT / TPOT / 联合SLO | 268/500、494/500、266/500 |
| 物理GPU生命周期 | 1,743.873277 GPU-s |
| GPU-s / binding-valid请求 | 3.487747 |
| 服务内存峰值 | 62,410,342,400 B |
| high / max / OOM / swap | 全0 |

联合SLO53.2%，不满足95%；功能覆盖通过不意味着性能可行。
平均router queue54.180s、service TTFT8.290s，保留启动/排队代价；不能将全部
时延归为远端传输，也不从此视图推断完整W0或跨系统优越性。
GPU积分四互斥窗口闭合；全部持卡区间有真实退出和释放证据。
未增加独立数值参考，原始n_correct=null/数值资格未知保持原状。

复用既有严格analyzer，无新统计/运行时补丁；37项CPU回归于运行前通过1.636s。
audit SHA `7cc9f8e220e0b5cf761cd5c416e6d3126d22f42cb77f46494af10341feedf3f7`；
evidence_manifest SHA `cb03ba5ddf53241e427712c285197bdcb3f89def15c2480f45e6872a4b20003c`，
576文件/36,970,413B，全部来源SHA已复核。
GPU释放/服务域移除/退出码0；本轮三个远端服务按InvocationID停止、端口无监听。
147旧结果保护项零变化，三个用户修改文件及既有S-LoRA十文件补丁未变。

下一步先备份、仅按既定引用检查回收本轮可重建下载副本并重新检查资源，
再原1000请求兼容验证和一次完整4000 W0。不要重复100/500资格寻找更好结果。

结果已备份：main77e02e01e7f18540ef38d0831ee8912aa88e9edb、baseline
4ab3d2c5f6fbe336f864bf4862bcf052e570867b，远端SHA均验证。
仅回收已核验可重建objects副本5000文件，释放13,004,427,264B；可用空间回到
163,267,284,992B。全部576项原证据和147旧结果保护项再次通过，原池与远端缓存保留。
清理阶段哈希核查触发辅助域memory.high回收，发生在GPU运行结束之后；无max/OOM/swap。
依据最新用户要求，不为该baseline开展性能优化或参数搜索；后续沿用相同配置。
