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
