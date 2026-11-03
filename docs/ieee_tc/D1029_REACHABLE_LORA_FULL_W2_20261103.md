# D1029：可达 active-LoRA 执行图的完整 3B W2 验证

2026-11-03。测前登记：D1028 画像已结束，不重跑、不追加短 Full。
本文件承接当前主线，不覆盖用户 dirty EXECUTION_STATUS.md。完整目标 active。

## 科学问题、已有证据与唯一候选

D1027 完整 4000 请求原生合同完成，联合 SLO 3784/4000（94.60%），物理
生命周期 8316.260703600012 GPU-s，P95 TTFT 1970.868099015 ms；差16条
达到冻结要求。其初始编译缓存与 D1022 不同，不能将改善全部归因文件校验。
D1027 的6个worker仍重复捕获88/32个图描述，编译已热的扩容初始化34–37秒。

D1028 根据实际 vLLM 0.30.0 V2 scheduler 的 max_num_seqs=8，保留32槽、
token轴、0及dummy=33，只裁掉不可能出现的中间16/32活跃LoRA图分支。
完整推导、历史否决方向和官方源码依据见 D1028_REACHABLE_LORA_CAPTURE_20261103.md。
30项相关CPU检查与一次423请求原生画像已经完成；实测服务图66/24，32槽
不变，KV空闲block增140。该画像不证明主工作负载收益或独立数值正确性。

唯一待检验假设：这一实现及其必要重新测量的初始化画像，能否在完整W2
改善联合SLO、尾延迟与GPU生命周期？不能把它写成纯启动单因素消融，不能
离线扣启动时间，不调整扩缩容阈值、并发或加载策略，不重开文件校验线程扫描。

## 冻结执行与比较合同

配置：configs/ieee_tc/20261103_d1029_3b_reachable_lora_w2.yaml；SHA
8a4dbd1945b04fcf4a8829c16e3b83930df2851454be9b19f930351e28c33b65。
父配置为D1027，结构比较仅14处差异：3个独占输出/缓存路径、1个capture
策略、2个completed-length身份、两份新画像各4项路径/SHA/上下文身份。
没有其他字段变化。32槽、并发8、max_num_seqs8、FP16/TP1、1–3实例、
D747 descriptor、stdlib gzip、真实已发布remote、全部IEEE公式与控制保持。

D1028 manifest SHA：83a9c988443ee005ca6586a030a50ea61cee136b797e6cdf111a34c6da2be12f。
service SHA：5d31393cbdfc4e6b32b652e776a0dfb9f076d66539bbdd1e8ab493058e278ef8。
preparation SHA：fda596df99900379bcfeba8bf91346f14b16b72a8bd394d49d93862dc40bbf20。
使用实际 dedicated subprocess 配置解析和原有严格loader，CPU加载验证160个
service类与96个preparation类通过；启动用fastsafetensors策略仍明确记录。
没有移除画像身份约束或使用缺类兜底。

输入复用seed42的4000请求、500逻辑adapter、W2 rotation100；arrival、prompt、
target和fixed_length_greedy_v1不变。60秒共同部署窗口，准备与收尾持卡均计量。
source trace SHA：4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d。
W2 map SHA：28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23。
这是已知负载的开发候选，不重命名为独立留出数据。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
单轮V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
D648 SLO不变：prompt≤506/>506的TTFT分别1466.050801102/2539.648668191ms，
TPOT分别29.921872367/36.215515193ms。G1要求全部正确且联合≥3800，再比较
物理GPU-s；G2另要求U≤12083.063870，再比较W2 P95及本次观测SLO差。
当前独立数值正确性未闭合，n_correct=null，ranking_eligible=false，不能由
binding/token或画像通过补造。没有CI、重复同键、挑阶段或按胜负删点。

## 测前检查、入口与保护

14处结构差异和严格画像加载在独立CPU2/3/26/27、3/4GiB、swap0中通过，
scope为primelora-tc-aux-d1029c0f00004f07989c33a15f8480001.scope，退出0。
未保存退出前全部资源计数，不能声称这项CPU检查全部资源事件为0。
147项旧投稿保护verify通过，回执raw/protected_before.json SHA为
5a88ff8a6eb65454973364d55797e9d5445c66b2d6b589fe532224faffe7b2f7。
用户manifest SHA 3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df，
EXECUTION_STATUS SHA 41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374不变。

10:08远端认证health通过，PID1984520、session-8529.scope及clock
remote-process-monotonic:b7a04b62f38c4ae6adabb4da5c0fd8dc未变；两端1000/full。
远端可用132651040768B。无服务重启、重新打包或共享k3s操作，仍保留无连续
远端CPU监测的限制。已有只读发布缓存直接复用。

复用scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh，文件名
含W1但实际固定W2。FAASLORA_D1000_RUN_ID=d1029_20261103，RUN_TAG为
d1029_3b_reachable_lora_w2_full，CONFIG为本配置。显式unset三个详细TC timing、
D1000_PREFIX_COUNT、D1000_NATIVE_PROFILE及TC_DIAGNOSTIC_PREFIX_COUNT。
descriptor library与SHA沿D1027；不设prefix、不重新构建库或升级后端。

raw独占父目录：/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1029_20261103。
实际run子目录d1029_3b_reachable_lora_w2_full由原runner mkdir独占创建。
外层tmux；service72/80GiB、swap2、CPU4–23/28–47；aux3/4GiB、swap0、
CPU2/3/26/27及独立watchdog。启动内存≥102GiB、磁盘≥150GiB并覆盖32GiB
预测增长；运行底线16GiB内存/100GiB磁盘。一次一个重型任务，测量中不维护、
备份、改源码/参数或管理远端。测前代码/配置备份后启动同一完整键。

## 终态动作与未完成范围

先核对实际release，再复用D703分析器恢复完整SLO、阶段、画像绑定、原生
adapter/token与双端传输。必须显式指定本配置和D1028 manifest。核对每次
实际worker的66/24及capture receipt，保留编译冷/热状态、KV差异和全部失败。
达标后立即推进同版本W0/W1和完整外部比较，不追加小规模调参。未达标只
保留反例、根据新证据决定下一候选，不自动重跑。

D810外部baseline18点及适配局限继续复用，不重启HydraServe/Loquetier适配。
W2 vLLM为100%联合、16110.752GPU-s、P95TTFT441.635ms，ServerlessLLM
为98.125%、16084.201、480.724ms；它们超预算但延迟更好，不隐藏。
全部未完范围保留：两模型同版本W0/W1/W2与G1/G2、独立数值正确性、A1–A5、
S1–S13、CE/Pareto、B1/B2/C1/C2/C3及两轮审稿文档、后期图表和最终V2集成。
PrimeLoRA达标前不绘图；本候选不代表论文完整实验已完成。
