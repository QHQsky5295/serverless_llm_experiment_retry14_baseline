# D1037：缓存负结果后的控制冲突核查与一致性修正

2026-11-03。主线仍为3B PrimeLoRA，目标active。上一轮D1036完整测量、
负结果分析和备份为PROGRESS；本轮是已有日志复用及确定性缺陷修正，
**没有新增GPU性能运行，不重跑D1036，不接纳110条缓存配置**。
本文件为最新交接，用户dirty的EXECUTION_STATUS.md保持原字节、不暂存。

## 原始观察与解释边界

复用D1033/D1036各4000请求及完整控制事件，调用既有D997忙实例下界分析。
请求按native owner合并confirmed admission至末token区间，在撤出路由时
截断；缺少准入前预留和末token后清理，仍只是忙时间下界，不是精确真值。

| 项目 | D1033 | D1036 |
|---|---:|---:|
| 联合时延达标 | 3772/4000（94.30%） | 3320/4000（83.00%） |
| 生命周期GPU-s | 8155.283735964913 | 8791.96267249249 |
| 控制记录 | 1944 | 1926 |
| 有服务率窗口 | 1943 | 1925 |
| 采样忙时间不足下界90%的窗口 | 8 | 20 |
| 采样/下界比P50 | 1.042276119 | 1.088909795 |
| score>1时实际缩容 | 0 | 1 |
| 实际缩容 | 2 | 11 |

90%仅沿用D997离线误差描述，不是新在线参数或资格门槛。窗口重叠，不作
独立重复或CI；比值大于1可能包含下界未覆盖的真实占用，不能一律称高估。
D1036的11次回收点比值均不低于0.99595，不支持把此次退化主要解释为
缩容点忙实例时间的严重低估，因此不重开事件积分/服务率扫描。

新证据是D1036首次缩容：observed_at=3196986.645513331，ready=3、pending=0、
queue=0、active=2，窗口P95 TTFT=14621.920608915389ms，score=2.9243841217830777。
当前容量已到上限，原max-score分支本应保持；但forecast target=1及已成熟的
回收计时器将最终动作改成3→2。随后确有物理释放，不是只提出了未执行动作。
下一次扩容在612.1695308280177秒之后，**不能据此宣称它造成全部452条
联合通过损失**，也不能把它和后期12.28/28.77秒内重新扩容的事件混为一次。

## 第一性原则、历史与官方实现

当前瞬时压力与窗口平均容量预测是不同证据。平均需求较低不能否定已经
触发的IEEE上阈压力；达到物理副本上限表示不能再加卡，不表示可以反向减卡。
本次修正动作合并关系，而不是提高阈值、延长超时或添加新调参系数。

D997曾在D992/D995中发现0次高压缩容，因而拒绝为未出现的现象追加候选；
这次有实际执行记录改变了该前提。源码历史核对be5ba93c（busy-runtime服务率）、
67cbd79d（预测释放）、88178f60（预测扩容），未重复实施这些已有机制。
D998只读调用延后、D844 stat批量化、D860文件投影、D864–866整图遍历等
历史效果均已复核；不将其重命名为本轮新优化。

联网核查[Knative控制源码](https://raw.githubusercontent.com/knative/serving/main/pkg/autoscaler/scaling/autoscaler.go)：
其高压力模式保留较大容量需求并不允许缩容。本项目只借鉴“容量预测不得
覆盖当前高压力”的决策原则，不移植其panic窗口、阈值或声称收益相同。
另核查[vLLM 0.30管理器](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
及[worker入口](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)，
确认CPU注册缓存与GPU激活槽为不同集合。它们不证明本地全图观察没有代价。

当前本地HOST观察仍逐个访问所有现存tensor；D1036最大110条相比父点32条，
说明有更多遍历对象，但完整运行没有各次清点CPU计时。仅凭总等待增加不能
量化这部分贡献。D866已测过整图原生实现，其局部收益不能直接当成当前
系统收益；本轮没有再开图遍历、线程、TTL/epoch缓存或容量网格试验。

## 实际修改与验证

`IEEEReplicaControl.evaluate`只增加一个动作仲裁分支：预测要求更多副本时
仍允许扩容；否则只要原score>1，保留原反应式动作，并清空预测释放驻留
计时器。压力下降后重新积累原有cooldown。低压力预测释放、score恰好等于1
的严格边界、pending/最小最大副本语义不变。论文九公式、冻结SLO、工作负载、
缓存预算、CPU/GPU资源包络及任何baseline均未改。

父版新增测试先运行：queue/active/TTFT分别在未到上限和到上限时均错误缩容，
另一个驻留计时测试也失败；共7个失败断言，原有较大预测扩容用例仍通过。
保留parent_tests.xml，不删失败或改断言使其通过。

修正后定向执行IEEEControlContract及现有control-profile分析测试，
**56/56通过**，19项既有警告。覆盖高压不反向缩容、压力解除后重新驻留、
原低压释放、预测可提出更大扩容，以及压缩来源身份和负结果事件保留。
这是CPU语义验证，不是新的4000请求性能结果，不宣称SLO已恢复或达标。

一次额外的旧IEEEActualControl整类回归因fixture启动了多个导入进程，
在辅助3/4GiB限制内出现停滞；13项完成后人工SIGINT结束，exit=2，scope已
inactive/dead，保留fixed_control_tests.xml。它不算全通过，未放宽内存限制。
首次systemctl使用本机不支持的--kill-whom无效，改用--kill-who成功；随后
SIGKILL请求时scope已退出，没有实际执行额外kill。未操作其他用户进程。
改为与本次纯控制修改直接相关的56项完整测试，不为此重开无关后端资格。
无连续测试资源峰值/事件记录，不虚构OOM/high为零或把配置当实测。

## 压缩证据复用和来源

既有analyze_d455_stages.py新增显式无损归档读取：仅当原plain不存在，且
逐文件回执路径、archive SHA和解压内容SHA匹配时读取gzip；不恢复600MB
副本、不改旧来源manifest，原plain若变化也不会静默改读归档。
复用compress_closed_main_outcomes.decompressed_digest，流式处理原数据。
新的summary.json包含原内容身份、当前归档/回执身份和全部异常事件。
没有重建数据、修改旧结果或在后台启动GPU。

产物：`paper_results/ieee_tc/p2_backend/d1037_native_host_control_audit/summary.json`。
测试原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1037_20261103/`。

| 测试回执 | SHA-256 |
|---|---|
| parent_tests.xml | 3e810fa6a032ce09bdce6643feaac6daacb81d0dc1468b56ad11b539bd128537 |
| fixed_control_tests.xml（中止） | 1aee7c383be28802d04b11ba5bcd91723f34c2c32d537c0cd9c7af1f9bb974d8 |
| targeted_tests.xml（56通过） | 57e3c2310d2c83c6c7ae9ca18731c513dc0f88216cc8c7d1a825e774efc29646 |

## 决定与完整主线

保留本次确定性一致性修正，但**不晋升为已证明有效的性能候选**。
D1036继续拒绝，不恢复110容量、不改109/64/128重跑。D1033也不是最终工作点。
当前纯仲裁修正尚未经过完整GPU回放，下一新执行须记录新源码身份；不能将
已有D1033/D1036结果改挂到新代码下，不能承诺这一低频冲突足以弥补3B缺口。

下一任务是利用当前32条父配置的已封存阶段、原生控制及执行所有权证据，
找出剩余排队/解码代价的一个可验证原因。只有具体新前提支持时才重开历史
方向；不因本次修正就自动重复画像或启动同键Full。若所需关键观察确实未
记录，先说明缺少的具体量，再做一次有界诊断，不连跑无目的优化。

两模型同最终版本W0/W1/W2/G1/G2、独立数值正确性、最终外部baseline比较、
A1–A5全部变体/confirmed-state/first-service、S1–S13、CE/Pareto、B1/B2/C1/C2/C3、
两轮审稿/新颖性文档、IEEE图表和最终V2集成全部保留。HydraServe适配和
既有18项baseline观测不重开；不接管共享k3s。性能达标前不为展示作图。

PLAN SHA：d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
冻结V1 SHA：5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22。
单轮V2 SHA：193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。

提交前12项分析来源SHA与147项历史保护均核验通过，两个用户dirty文件
SHA不变，git diff检查通过。summary SHA为
667c7b5af7d5713851117e9ac82fcc45ad3c6bc4575b4781edb1e25f3ecf23a3。
第一次保护核验入口误传str而非Path，在读取seal前报错；修正调用类型后
通过，未修改seal或重跑分析。当前无GPU计算进程，可用磁盘161072783360B
（仅略高于150GiB启动底线），下一重任务仍需核查预计增长；本轮未删除
或再次压缩任何历史数据。代码/测试/分析只按明确文件清单提交，现有
FaaSLoRA V2分歧不强推，检查点备份不冒称最终集成交付。
