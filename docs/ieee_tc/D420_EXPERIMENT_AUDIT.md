# D420 experiment-audit：D419 证据复核

日期2026-10-11；审阅者gpt-5.6-sol ultra，全新只读agent。
总体WARN；review_independence=same-family，acceptance_status=provisional。
确定性检查与语义结论分开，建议性审阅不改变冻结协议。

| 检查 | 状态 | 证据／解释 |
|---|---|---|
| A 来源 | PASS | analyzer:68–199：源trace目标、请求绑定、独立数值参考分开；不以输出造目标 |
| B 分母 | PASS | analyzer:202–250：冻结共同SLO、offered分母、条件TPOT；未自归一化 |
| C 文件与数值 | WARN | 103个证据文件校验一致、100请求指标可重算；旧CPU测试缺独立文件回执 |
| D 执行路径 | PASS | runner:231–264、623–627及worker receipts：实际原生观察入口被调用 |
| E 范围 | WARN | 100请求／29 ID／4权重SHA／单轮；只支撑局部路径资格 |
| F 类型 | PASS | 真实系统合同/计量检查；HF/PEFT比较是有限独立数值参考，不是任务语义真值 |

analyzer为`scripts/ieee_tc/audit_d402_serverlessllm_path.py`；runner为baseline的
`scripts/replay_vllm_engine_trace.py`。精确逐项引用及完整请求／回应见
`.aris/traces/experiment-audit/2026-10-11_run01/`，机器报告见
`paper_results/ieee_tc/baseline_audit/d420_experiment_audit/EXPERIMENT_AUDIT.json`。

审阅者复算：100/100原生合同、29 ID、17369 token；平均/P95 TTFT
18634.907506/74980.162867ms，TPOT25.349100/27.612873ms，平均E2E23055.655692ms；
共同TTFT/TPOT/联合47/100/47；四租约合计1019.2586401619483GPU-s；29次真实
传输50650865线上字节、724665100物化字节、0请求时打包。上述仍不是完整W0。

执行裁决：

- 采纳D419短路径证据可复用、500-ID缺口不能用名义池大小替代的判断。
- 采纳测试回执不足：旧0.090秒陈述只具有当时执行记录，不能用新测试证明旧
  耗时；D421结束后保存一次新的CPU测试回执，明确新日期、不改旧manifest。
- 原始`n_correct=null`保持不变。审阅建议第5条若理解为“只要原字段未知，就
  永远不能在新增分析中依据完整证据导出正确完成”则过强，不采纳该扩张解释。
  V2“参考标定与复用”明确允许新产物按证据导出指标，同时禁止原地补写未知值。
  D419本身无完整4000，不能设置G1/G2/参考；今后仍须按冻结z定义逐字段核验，
  不凭HTTP200或输出长度单独推定adapter正确，不以原字段null替代证据审查。
- 回应中的D385相对路径少一个`..`；实际文件在
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d385_20261010/7b_peft_sequence_reference_100_retry1.json`。
  trace保留原回应，不把引用拼写问题改写成数据不存在。
- 审阅附属shared-references五文件在本机缺失，此限制保留，不宣称跨家族独立认证。

没有因此新增GPU重跑、改变评价门槛或扩大数值一致性要求。D421完成后继续原
1000兼容与4000主线，而不是反复审阅或重新启动100请求。

2026-10-11 D421终态后新增留档：相同analyzer/test源码SHA的27项CPU回归通过，
本次耗时0.174秒，raw `d421_20261011/cpu_contract_tests_receipt.json`和log已纳入
D421证据清单。它补齐今后的回执链，不证明旧0.090秒，原审阅WARN原样保留。
