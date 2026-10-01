# 旧、新 PrimeLoRA 性能差距与阶段完成条件

2026-10-01；落实用户新增要求：核对旧、新结果，解释差异；3B 未达到性能目标
不能因 4,000 请求结束而结项，也不能提前转入完整基线矩阵。
这是执行顺序与证据要求的补充，不改变冻结指标 V1、IEEE 九式或 D137 在途配置。

## 1. 当前决定

- **3B 性能优化仍为 OPEN。** D118 的完整生成、资源释放和计量检查已完成；
  性能目标、数值 adapter 身份及共同 SLO 资格没有完成。
- D118 是最近完成的 3B Full，但运行代码为 `48f808ab`，不是目前 `233ccc29`。
  后续针对 7B 验证的共享路径改动不能自动算作 3B 的性能结果。
- D137 的 7B Full 已按原条件完成和审计，4,000条原生契约成功、0失败，
  但平均/P95 TTFT为603.644819/1,141.327139 s，性能仍未达标。
  未重启或并行启动3B，运行中未改服务/远端配置。完成证据备份后优先3B回验，
  不因 7B 或 3B 仅“无报错”便恢复完整基线阶段。
- 已完成的 D118、D135、D136 证据和备份不重做；一次性交付缓存不重建。
- 本文优先于旧 D118 文末及旧执行记录中可能被误读为“3B 已完成，直接转
  reference/基线”的下一步文字。共同 warm/Resident 标定仍是资格依赖，
  不是绕过 Prime 性能问题的理由。

## 2. 先列原始数字，不混作受控优劣比较

以下三列都是完整 4,000 请求的单运行描述。旧结果按原表定义；D118 使用
冻结 V1 的请求时间线与 Type-1 分位数。运行合同不同，不能给出因果改善率、
配对置信区间或“新版本已胜出”排名。

| 3B 指标 | 旧投稿 local-sim | 旧真实远程 | IEEE D118 |
|---|---:|---:|---:|
| 原报告完成 / 原生契约成功数 | 4,000（原报告） | 4,000（原报告） | 4,000（原生契约） |
| 平均 TTFT（s） | 0.881314 | 1.087226 | 122.146518 |
| P95 TTFT（s） | 2.213230 | 3.744184 | 268.605684 |
| 平均 dispatch/admission 等待（s） | 0.214870 | 0.310180 | 114.177866 |
| 平均 service TTFT（s） | 0.666443 | 0.777046 | 7.968652 |
| 平均 E2E（s；完成边界有差异） | 2.942699 | 3.286235 | 127.878409 |
| 平均 TPOT（ms） | 26.588500 | 34.922200 | 29.955123 |
| P95 TPOT（ms） | 78.762608 | 112.475407 | 64.782991 |

描述性幅度：D118 平均 TTFT 数值约为旧真实远程的 112.35 倍，P95 约 71.74 倍；
平均 TPOT 数值反而低约 14.22%。这些只是跨合同数字之比，**不是控制其他变量
后的退化/优化估计**。不能只取 TPOT 较好一项掩盖巨大的请求等待。

7B 也保留历史参照：旧投稿平均/P95 TTFT 为 0.563951/1.015488 s，旧真实远程
为 0.657170/1.661288 s。新 D135 是 3,999 成功、1 失败，成功子集平均/P95
为 534.819472/1,016.659398 s，不能当作完整正确结果。10:49补充D137完整
结果：4,000原生契约成功、平均/P95为603.644819/1,141.327139 s，
仍无数值adapter/共同SLO资格。旧主表与消融Full来源差异已在P0审计，
不能手工改数字统一。

## 3. 已证实的差异与尚未证实的原因

| 项目 | 已核查事实 | 解释边界 |
|---|---|---|
| 到达/计时 | 旧原始 metadata 也定义从 scheduled trace arrival 起算 TTFT | 不能笼统说旧版不含排队；旧事件实现仍需逐字段核验 |
| 生成合同 | 旧 max input/output cap 为 null、generation seed 为 null；新为固定 greedy、prompt≤759、输出 min(source,256) | 记录的 token 总量与逐请求数量已补核见§6；旧 prompt hash 缺失，不能称相同计算工作 |
| 实例范围 | 两份旧 3B metadata 的 max_instances=2；D118 上限4且实际持有4张卡 | 新代码不能用“多用了卡所以正确”作优化结论；也不能按旧折扣成本推导新物理 GPU-s |
| 单实例声明容量 | 两份旧 3B 与 D118 均为 max_num_seqs=8、runtime cap=8、max_loras=8、max_cpu_loras=32 | 不能把 3B 差距简单说成新后端 batch 上限更低；实际有效并发仍需事件证据 |
| 原生批处理 | 三者 max_num_batched_tokens=4096、max_model_len=1024，prefix/chunked prefill 均关闭 | 相同声明值不保证实际调度和内核相同 |
| 路由/缓存 | 旧 adapter_affinity、HOST4GiB；D118 IEEE confirmed 路由、HOST16GiB及原生 HOST 预算 | 不只是旧系统调几个参数；控制路径实质改变，不能将所有开销归因于论文公式 |
| 远端 | 旧版也有真实远程结果；D118 132 对传输、306,360,162 B、请求内打包0 | 不能以“换真实远程”解释全部百秒等待；各版交付/限速/缓存初态仍不同 |
| 资源与账单 | 旧 metadata 明确为 simulated lifecycle 加 idle factor；D118 为真实 allocation/release 并集 | 旧表 active+idle+startup 不能直接冒称同口径物理 GPU 生命周期 |
| 执行资源 | D118 内存 high/max/OOM/swap/告警均0，实际4个 lease 全释放 | 没有证据把本轮长等待归咎于 OOM 或未释放卡；不能据此排除所有 CPU/控制瓶颈 |

D118 同轮平均 TTFT 的 **93.48%** 是 dispatch/admission 前等待；native vLLM
首 token 时间平均 0.475647 s，占平均 TTFT 约 **0.389%**。这两个比例是同轮
同请求集合的阶段占比，和上面的跨合同数值比不同。

dispatch 前 114.177866 s 包含：global dispatch window 99.994035 s、runtime
slot 11.679442 s、到达释放迟到 2.504390 s。service TTFT 中 admission 至 native
dispatch 为 7.493005 s。说明优先研究控制/准备路径及其造成的有效服务容量损失
有依据，但“等待在哪”还不等于“哪个函数造成全部等待”。不得删去排队计时，
也不得用 native TTFT 替代用户 TTFT。

现有 D136 已通过实际 runner 的反例证明旧外层许可可能被新到达请求抢先获得，
修正为唤醒前移交许可、FIFO 等待与取消归还。它证明一个实现问题存在，尚未
证明能解释上述全部差距，更没有提供新 3B Full 的效果。D137 检验的是该修正的
普通 7B 完整回放，不是为3B补造的证据。

## 4. 不能过早结束的验收规则

1. **执行层：**完整原生生成、真实 adapter/输入身份、可靠时间线、真实资源释放。
   本轮工件的数值可区分性限制单列，不能把 native 数量检查升级成数值身份通过。
2. **历史差距层：**为旧、新 3B/7B 保留同一来源对照表；能够离线恢复的先恢复，
   不重新生成工件或 trace。受合同影响而不能比较的字段必须明确，不按想要的结论
   挑旧 run，也不把不同条件当作无需解释大幅长等待的免责理由。
3. **当前性能层：**最终候选必须有自身的两模型完整普通 Full 结果。共享改动在
   7B 上运行不能代替 3B 回验；3B 的 D118 不能永久作为后续新代码的验收结果。
4. **正式服务目标层：**共同 warm 阈值和 Resident 参考尚未冻结，不能虚构
   “应达到”的数值。按 V1 完成标定后检验 G1/G2；不以开发5000ms或旧 CE 结项。
5. **优化结论层：**持续以比旧实现和合格对照更好的资源–服务表现为目标；当前
   百秒等待明确未达到该目标。若最终同合同证据仍不领先，状态保留未达成并继续
   有依据的开发，而不修改阈值、删失败、降工作量或声称已完成。若确实缺少授权/
   外部条件，则如实报告所缺条件，不能无限做无证据的小修补。

顺序：D137 完整收尾与表格 → 判断该修正是否解决其目标问题 → 返回3B最新
候选与旧结果的差距核验/性能改善 → 共同资格与reference → 恢复既定基线顺序。
每个新候选仍须历史证据、原始文献/官方源码、可证伪假设、最小验证及普通 Full；
本次仅做只读历史分析和执行文档补充，没有选择或叠加第二个优化。

## 5. 来源与本次读写范围

08:32 初次分析只读取既有小型表/汇总、两份旧原始 JSON 的前7,500字节 metadata，以及
D118 配置；没有整载/重投影 D118 9.18GB 原始结果，也没有运行新 GPU 实验。
表格用于记录精确值和比较限制，按计划§11采用表格，不画暗示公平排名的柱图。

| 来源 | SHA256 |
|---|---|
| `paper_results/final_v2/tables/table1_end_to_end_data.csv` | `8f73d75415e3131aad67f1378f241b3efc649c13b6f4d11e9af2033fb7b90a32` |
| `paper_results/final_v2/tables/table_ttft_decomposition_data.csv` | `5ba33f7f919e63155e04cba67e7e8e6c308a1c09abd8b12a7171021cc5b3c808` |
| `paper_results/final_remote_full_real_remote_v1/figs/paper/main/table1_end_to_end_data.csv` | `f36b4ad8d04cabe9ad7146b08bef6ea927dde5f547f84df3ced70ca0d393308b` |
| `paper_results/final_remote_full_real_remote_v1/figs/paper/main/table_ttft_decomposition_data.csv` | `0339cc42d6dbb5bff6d5d60bff1499bfde55564c3f3e276924af94df88e94342` |
| `paper_results/ieee_tc/p2_backend/20260930_d118_3b_full_w0_full11.json` | `ff39d2b0fbb952f6bf1d0601115eff9888895b13d0a9b6c2d062c2fbef376eaf` |

旧原始路径和生成日期由上述表的 `source` 列及其 metadata 对应：local-sim
为2026-05-10 11:58:05，真实远程为2026-05-14 01:53:27。D118 配置SHA为
`6da8d8964cd54d00cfb1589bb0c8a4cdefca0916c2cecc6ebb6be7749d9e6dec`。
初次分析未凭部分读取宣称完成整个历史来源审核；10:07 完整原始 SHA 与
可恢复逐请求字段核查已经补充，见下一节。缺失的原生身份/时间证据仍未补造。

另参见 `P0_FULL_PROVENANCE.md`、`WHY_LEGACY_RUNS_AND_IEEE_FAILURES_20260928.md`、
`D118_FULL_W0_FULL11.md`、`D135_FULL_W0_FULL1.md`、`D136_DISPATCH_PERMIT_OWNERSHIP.md`。
冻结 V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
在途 D137 所绑定 Plan SHA `fe6c05b008c01b89316b7953d73fc7ad9e3b4763e35bd63d3c46594049310c5c`，
二者未改。本文是新执行补充，不追改 D118 或 D137 的实验身份。

## 6. 10:07 补核：实际记录的工作量，不再仅看 cap 配置

D137 已完成推理和资源释放；等待其一次大文件提取时，只读取两份约12.3 MB
旧原始日志和已存在的33.9 MB D118投影，没有重读D118约9.18 GB原始结果、
启动新模型或改变候选。完整来源与精确核查值保存在
`paper_results/ieee_tc/p2_backend/20261001_legacy_3b_contract_audit.json`。

| 3B 逐请求记录 | 旧 local-sim | 旧真实远程 | D118 |
|---|---:|---:|---:|
| 请求数 / 唯一请求 ID 数 | 4,000 / 4,000 | 4,000 / 4,000 | 4,000 / 4,000 |
| input_tokens 总和 | 2,981,921 | 2,981,921 | 2,594,938 |
| output_tokens 总和 | 447,515 | 447,447 | 458,224 |
| 最大 input_tokens | 985 | 985 | 760 |
| 最大 output_tokens | 394 | 394 | 256 |
| canonical prompt hash 条目数 | 0 | 0 | 4,000 |
| 请求级 generation_contract | 未记录 | 未记录 | fixed_length_greedy_v1 |

旧数值是日志记录字段之和，不提升为重新验证过的原生 token 数。D118 最大
input_tokens=760 含其记录的 tokenizer 特殊 token，不与“内容最多759”混淆。
补核4,000条记录的 input_tokens 均等于 native_token_timing.actual_prompt_tokens；
canonical_prompt_tokens 总量2,590,938、最大759，逐条 native−content 均为1。
共享 canonical_fixed_prompt 明确分别编码 add_special_tokens=False/True，
并保留两个身份；没有根据760这个数值猜测或放宽上限。

按 request_id 排序逐条比较：

- 旧 local-sim 与旧真实远程：ID、adapter、scheduled arrival 全部一致；
  input_tokens 4,000/4,000 一致，output_tokens 3,789/4,000 一致。
- 旧真实远程与 D118：ID、adapter、scheduled arrival 全部一致；
  input_tokens 仅4/4,000一致，output_tokens 3,807/4,000一致。
- 相同 token 数不证明相同文本；旧 prompt hash 缺失，不能宣称 canonical
  prompt 内容逐条相同。不能把旧运行的 metadata token_source 当作原生计数证明。

D118 记录的输入总量比旧真实远程少约12.98%，输出多约2.41%。这些数据
不支持“计算工作量增加了约百倍”这一解释，也不能以一句“生成合同不同”
结束性能责任分析。但总 token 量不是精确服务时间：接近容量边界时，小幅
服务能力变化也可能放大为长队列；仍需结合控制/准备阶段、有效批处理与
实际并发定位，不据此宣布生成差异完全没有影响。历史证据仍分类 R2。

完整 SHA：

- 旧 local-sim：`7e084b6d60c8ca34a78cf117a2a3e251ea188e934c6b4e3739604f328918949e`。
- 旧真实远程：`ed4a934211ee36301fc87adaa1d295908bcfd230e2ad38fcac34b9fb6f2c0f02`。
- 既有 D118 投影：`b40fe880592473f74a654323f8f291c910af0517cd8acf67dfa67fa3abb6a840`，
  与已提交 D118 source_refs 一致。原 D118 结果与封存报告未改。

10:37 独立重算核验：使用单独的 Python 聚合逻辑重新读取上述三个小型来源，
逐项核对表内总数、最大值、逐请求对应关系和 native/content token 差；
全部一致，12,000 条记录均纳入。来源读取前后 SHA 不变，D118 投影 SHA
与封存汇总一致。此检查不解析 D118 大型原始 JSON，也不是新的性能运行。
检查耗时1.14 s，峰值 RSS134,524 KiB，受限资源域全部内存事件为0并已释放。
回执位于 D137 原始目录 `legacy_audit_verification.json`，SHA为
`85a00d13bfda3fe060a675ef32ff9a3276501e756d883e4e8b8966fd1479b1ce`；
对应被核验表SHA为
`39d086ee40d7d8c182f1c6c6f6bb95d8dec162897e8435544b8b3387895db272`。
该核验只确认统计与原记录相符，不弥补旧日志缺失的原生计数/prompt身份。
