# 当前交接 D1081：完整W1已闭合，有改善但净收益未解决

2026-11-06更新：D1081不再运行，勿重启。完整4000/4000成功且原生token合同
匹配，真实GPU释放通过。共同联合SLO3864/4000=96.60%，物理U6127.701722，
P95 TTFT1043.464586ms。旧Full为95.90%、6182.715971、1114.091001ms；
本轮有改善，但仍不及ElasticOnly/ElasticLocality旧消融观察，不宣告Full净
收益或跨场景完成。n_correct未知和开发身份保留，不成为重开资产支线的理由。

现有D703/D299/D1064分析已全部完成，completion/manifest与overhead_comparison
已写出；2151规划事件、1228稀疏计划、915confirmed-GPU no-op。CPU总量
506.734→295.636s，但runtime-slot等待130.026→130.889ms基本未改善，
下一步应从现有请求/准备/native事件归因剩余等待，不继续无依据精简表示。
详见D1081_FULL_W1_OUTCOME_20261106.md（含全部限制、失败和剩余范围）。

launcher/watchdog均已退出，实际服务域消失；两个分析域已空并停止。
174既有3B unit收集后按原身份停止，7B/k3s未动。工作缓存正常释放，
144旧投稿hash和两项用户dirty SHA不变。无新GPU运行或图表。
当前待本结果检查点备份；不要把原实现备份误称结果已推送。
以下为保留的启动与监测历史，不代表当前仍在等待。

先读目标、完整 PrimeLoRA-PLAN.md、V1和单轮V2。用户dirty的
EXECUTION_STATUS.md不编辑、不暂存；本文件是增量状态。

当前主线是3B A1 Full净收益。基线已完成（HydraServe独立Podman，无需共享
k3s），3B同配置W0/W1/W2 D1048和十一变体D1063已完成。不能重开适配或
把D1077/1079的load_only称Full；240秒诊断不是60秒主协议。

本轮D1081实现固定需求窗口下的精确稀疏选项表示：保留全源图/预算/替换
对象，只不物化从未进入求解器的零需求记录。不改九公式、路由、admission、
成本/SLO、模型配置或数据。详见 D1081_POSITIVE_DEMAND_OPTIONS_20261106.md。

四项新测试通过，接入后21项相关测试通过。CPU组件有收益但不是端到端证明。
两次组件启动/回执错误和一批memory.high中止均记录，未隐藏失败或放宽资源。
没有新图，没有改论文或旧结果；174仅启动既有3B只读交付服务并验证health。

下一项：按该文登记，复用D1000入口和D1047配置，仅一次完整4000请求W1，
60秒、真实remote、fresh_d747_v1、无profiling/prefix。开始后在本文件追加
实际PID/资源域；当前本段仅登记，不能据此声称已开始或完成。结束后先释放
资源，再复用D703/D299统计共同D648阈值、Type-1指标和物理GPU区间并集；
同时核查新选项计数。失败/负结果保留，不重复同键、不延长准备窗口求胜。

尚待Full净收益、M2预算、7B A1/累积、受控A2–A5、S1–S13、成本/Pareto、
两轮审稿文档、同版本正式图表与实际V2集成。总体目标ACTIVE，不标完成。

## 实际启动与首批请求（2026-11-06，进行中）

代码检查点 `8cdc8f815aa35952c8329bcba402f092db7be5ad` 已推送并以
ls-remote核验 origin/retry14_continuous_queue_v2 及
faaslora_origin/backup/d1081-positive-demand-options-20261106。实际FaaSLoRA
V2分歧不强推；本次是检查点备份，不是最终集成。

- tmux `tc-d1081-full`，launcher PID1098795，watchdog PID1098895。
- service `primelora-tc-svc-9ab99d56db504943b9b7baa9e7e6812f.scope`，
  InvocationID `0d74f02f99f846f981296ed6dc0e937b`。
- auxiliary `primelora-tc-aux-40d459147e0642bf8271a666e4d42d3b.scope`。
- raw `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1081_20261106/d1081_3b_positive_demand_options_w1/`。
- allow_exec=true；部署3440015.632025311，业务3440075.632025311，实际60秒。
- W1 count4000，source SHA4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d，
  view SHA8a7fa829c4902a761efcf58c765580cbdb5e0f367e3c2245ea43417409e9494f。
- 首批33条terminal均success=true，0失败。这是进度，不是最终正确性或SLO。
- watchdog sample79：服务峰值12947873792 B，high/max/OOM/swap=0，
  host available104805871616 B，disk164395614208 B，无warning/abort。
- 远端3B unit InvocationID33d601c5b42042aba771131058a8e3b6；沿用D80缓存，
  health已确认。测量期间不得重启、重新打包、压缩或额外下载。

继续监测这一完整键，不另启候选，不修改代码/配置，不重跑已完成baseline。
结束后资源释放、物理计量、共同阈值和机制回执分析，再决定下一主线。
此启动交接是代码检查点之后新增的本地文档，尚未另推；完成后纳入下一备份。

## 同一运行监测：2026-11-06，VERIFIED_WAIT

上一轮属于PROGRESS（实现/验证/备份/启动）；本轮为已核实的等待，不计为
新性能实验。重新读取目标、完整计划与增量状态，核对真实PID和相同service
InvocationID。launcher1098795、watchdog1098895持续存活，tmux与scope active。
完成数由173增至443；443个唯一request均成功、native_contract_matched=true，
无error_type。尚未完成4000请求，不估算最终SLO或资源排名。

watchdog sample329：service peak17.1013GiB，high/max/OOM/kill/swap全0，
host available90.5770GiB，disk151.4688GiB，无warning/abort/foreign compute。
本轮只读监测和本交接追加；没有新GPU运行、代码/配置修改、清理压缩、远端
重启或图表。已经核对现有D703/D299及D1066分析入口，完整结束后复用它们
恢复共同D648指标、物理区间并集和新候选计数。继续等待同一D1081，不重启。

## 同一运行监测续记：2026-11-06 10:53 +08，VERIFIED_WAIT

当前仍是同一D1081完整W1，非新实验或新候选。已再次完整读取目标、1567行
源计划、冻结V1/单轮V2，并核对本文件及用户dirty状态文件的历史/最新入口；
后者D998交接已过时，不据此重开baseline、HydraServe或共享k3s适配。
本轮监测完成数443→901→1147→1336→1415；最新1415个唯一request全部
success=true、native_contract_matched=true，无error_type。相同service
InvocationID仍active，launcher1098795和watchdog1098895仍存活。

watchdog sample834：service peak18.2799GiB，high/max/OOM/kill/swap均0，
host available90.5055GiB，disk150.7375GiB，无warning/abort/foreign compute。
这些是部分执行/安全观察，不是完整SLO、性能排名或数值正确性结论。

已只读核查D1066闭环收集脚本和D703现有接口：结束后复用共同SLO重算、
profile绑定、阶段分解与真实远端交付/物理释放审计。运行使用原D1047 YAML
加已记录的输出目录override，统计时应显式绑定实际目录，不改执行配置。
新positive-demand计数来自owned worker回执；不把CPU组件收益换算成TTFT，
不重复累计公开/私有事件。尚未执行离线大日志统计或远端下载。

本轮无代码/配置变更、额外回放、维护压缩、远端重启或图表。继续等待这一键
完整结束；下一步仍是释放→完整统计→机制净收益判定。全部剩余主线保持，
总体目标ACTIVE，不因正常等待标blocked，也不重复同键或提前宣布获胜。
