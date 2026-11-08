# D1131 运行状态

18:24 +08：实验及离线证据闭环完成，见
`D1131_7B_ROUTING_HANDOFF_W1_RESULT_20261108.md`。4000/4000 请求成功，
原生合同4000，联合达标3900（97.50%），GPU-s7427.711741；launch.pass=true，
service/watchdog/replay returncode均0，service domain removed，GPU context released，
远端inactive，外置辅助scope已单独停止。远端journal 132行已从当前clock对应
文件复制并核验 SHA。

已完成只读分析：D1113 reconciliation、D299 evidence review、D455 stages、
累积 comparison 和 handoff timing audit。RoutingHandoff 相对 RoutingOnly
本轮 +3.09% GPU-s、SLO -0.125pp、P95 TTFT +12.48%；Full 相对 CapacityOnly
减少4.61% GPU-s、SLO +0.30pp、P95基本持平。20 activation（1 initial/19 natural），
326明确物理加载、134次首次dispatch前完成；430次invocation字段缺失保留unknown，
不把timing audit标为完整通过。该节点作为完整观测封存，不宣称handoff单项净收益
或全局最优；下一步回到受控A2/A4或S2/S3，不重跑同键/Full/基线/3B。

17:43:31 +08：同一panePID458696仍live/Ss+，elapsed11:08；1010/4000唯一
请求完成，1010成功、1010原生合同匹配。sample654：4GPU，high/max/OOM/
kill/swap均0，无abort/foreign/escaped；available105641836544 B，
disk162370404352 B，尚无launch终态。前轮及本轮均为VERIFIEDWAIT；本轮
目标重新读取，PLAN SHA与前轮全文一致，V1/V2全文重新读取。仅准备未执行
物理核查脚本，不改服务、不测试、不维护、不推送。继续同一次运行。

17:39:40 +08：同一panePID458696仍live/Ss+，elapsed7:17，未产生launch终态。
543/4000唯一请求完成，543成功、543原生合同匹配。sample427：2GPU，
high/max/OOM/kill/swap均0，无abort/foreign/escaped；available111060439040 B，
disk162678329344 B。本轮完整PLAN/目标重读，前轮PROGRESS，本轮VERIFIEDWAIT
及未执行分析入口准备；无服务修改、测试、重分析、维护或推送。继续同句柄。

17:35:14 +08：同一个运行仍 active/running，panePID458696，InvocationID未变。
已有114个唯一请求完成，114成功，114原生生成数量/绑定合同匹配；不是完整
结果，也不是数值正确性证明。17:34:42 sample133：4个GPU，high/max/OOM/
kill/swap均0，无abort/foreign/escaped，available106826170368 B，
disk163332853760 B。launch.json尚不存在。继续同句柄等待，不重复启动。

## 17:32:24 +08：已启动唯一运行，尚无终态

- tmux：`tc-d1131-routing-handoff-7b-w1`，pane `%14`，PID `458696`。
- 原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1131_20261108/7b_routing_handoff_w1`。
- service：`primelora-tc-svc-535639dcb6ac4933ab0b326e6909ac82.scope`，
  InvocationID `a58b01d45a2c46e39e5dce09821470d9`，已读回 active/running。
- aux：`primelora-tc-aux-29d1944ae8c54a90869bc6ec363019fe.scope`。
- notice `3637596.038786396`，replay t0 `3637656.038786396`，60秒间隔。
- trace SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
  W1 view SHA `103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3`。
- remote clock：`remote-process-monotonic:d93a842a41ed4fa2b922ed2187aab75c`；
  prepublished_gzip_v1 / artifact_timing_v2 健康检查通过。服务启动后首次连接
  曾未就绪，原 runner 有限健康轮询随后成功；发生在共同部署通知之前，不是
  请求失败，也没有额外启动实验。
- 启动资源：MemAvailable121774170112 B、disk163865210880 B、GPU compute为空，
  两端1000 Mbps，远端先前inactive/MainPID0。
- sample34：high/max/OOM/kill/swap均0；available108818104320 B，
  disk163863793664 B，无abort/foreign compute/escaped owned。

继续监测同一个执行句柄；不重启、不做代码修改/测试/重分析/压缩/推送。
尚未完成全部请求，不输出新运行最终指标。

## 正常终态后的既有分析入口

实际GPU释放、服务域移除、远端专属服务停止后保存同clock journal；不可只看
tmux退出码就假定清理完成。然后复用：

1. `paper_results/ieee_tc/main_comparison/d1113_contract_reconciliation/collect.py`
   的 `--prime-only --raw RAW --run-id D1131 --output BASE/result1
   --scenario tc_ieee_routing_handoff`。
2. `scripts/ieee_tc/check_d299_full_context.py --single-run-source SOURCE
   --single-run-scenario tc_ieee_routing_handoff --warm-reference
   paper_results/ieee_tc/p2_backend/20261010_d394_7b_warm_reference.json
   --review-output BASE/result1/evidence_review.json`。
3. `scripts/ieee_tc/analyze_d455_stages.py --output BASE/stages
   --retained-control-source SOURCE --retained-control-launch RAW/launch.json
   --retained-control-model 7b --retained-control-scenario tc_ieee_routing_handoff`。

其中 BASE 为 `paper_results/ieee_tc/ablation/d1131_7b_routing_handoff_w1`，
SOURCE 必须取本次成功导出原件。D1130已准备而未执行的compare.py可在本次新
目录中更换身份/路径后复用；不能对失败D1130绕过成功校验。直接复用封存
D1129/D1128及同一个D1115 Full，不重分析或重跑已封存对照。实际handoff使用
仍须复用既有物理事件核查，计划数不等于及时准备或收益。

17:39 +08 只做离线入口准备：已在本次新目录保存复用的 `compare.py`，尚未
导入、测试或执行，没有读取本轮大结果或改动服务代码。沿用全部原检查和
统计，只更换D1131身份/配置SHA，并从原D1130登记读取不可变RoutingOnly来源：
原登记SHA `87386dac45d3b20016e369f22d57d51e757bb4a593f01d2787c1b9a577cddb2a`。
这避免假设D1131简化登记含有旧登记的所有字段。报告另行记录CPU收尾修复及
D1130失败不参与排名，不默认为源码完全相同。性能结束前不执行此入口。

实际handoff核查优先复用
`paper_results/ieee_tc/ablation/d1126_7b_delayed_handoff_w1/audit_delayed_timing.py`
的原生receipt/共享任务归属逻辑（SHA
`cf1b8db88bbf89f07167acbbeae87bb8edcfc477f209821442e0b0d623892722`）。
其旧PLAN身份需在新副本中更新；RoutingHandoff没有Delayed专属“开始不得早于
ready”的约束，不能照搬为验收门槛。未知invocation不猜为false，实际加载按
owner/clock/adapter/interval去重；复用已封存Full物理核查，不重复分析Full。
该物理核查入口尚未复制、修改或执行，留待完整结果及真实释放后。

后续准备更新（17:42 +08）：上述入口现已在本次新目录保存为
`audit_handoff_timing.py`，仅改本轮身份、当前PLAN SHA、输出snapshot名称和
来源说明；原生receipt及共享任务归属逻辑复用D1126。旧Delayed专属检查仍
只对其原variant适用，不应用到RoutingHandoff。未导入、测试或执行；未
重新读取Full大结果。比较与物理核查均须本轮完整导出/释放后按序执行。

## 启动前记录

17:31:50 +08：登记备份完成，commit/remote HEAD 均为
`340a3b8633fef37bbcad102d5aae7fdfba922f0c`。7个明确文件已推送；diff/secrets
检查、配置 SHA 和三项用户 dirty SHA 通过。main/wrapper 均0，3.419秒。
无新 serving 修改，不重复备份或测试；下一动作仅为即时资源核对和唯一启动。

2026-11-08 17:31 +08：已登记，尚未启动。

- 目的：补全7B W1累积消融 RoutingHandoff，修复的仅为 D1130 CPU helper 收尾。
- 修复源码已备份 `7ce2f815d697218733ee799c8017f27335a089de`；没有新 serving 修改。
- 新配置 SHA `04e36f3e720e3eb53e262da3a77cc42336819f3273bf20d03df9cfb67be11fbb`。
- 与失败运行的配置解析比较只改变三个私有路径，已通过；路径均不存在。
- 已完成 CPU 测试复用，不重复资格、Full、RoutingOnly、外部基线或3B。
- 新记录备份后，重读即时资源并仅启动一次；tmux 保留退出状态。
- D1130失败与恢复记录封存，不会被新运行覆盖或按数值优劣择优。
- 当前全目标继续；下一项为受控时序、位置/信息、争用及预算敏感性。

登记及完整启动命令见 [登记文档](D1131_7B_ROUTING_HANDOFF_W1_REGISTRATION_20261108.md)。
