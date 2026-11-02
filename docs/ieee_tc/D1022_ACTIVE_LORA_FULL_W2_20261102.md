# D1022：活跃 LoRA 专门化的完整 3B W2 验证

2026-11-02登记，2026-11-03完成。状态：唯一完整Full W2及离线核验已完成；
所有GPU已释放，禁止重复启动。下方运行中段落仅保留历史身份。
当前交接以本文及D1021完成节为准；用户dirty的EXECUTION_STATUS.md不覆盖。
上一目标回合属于progress：新候选接入、必要画像实际启动；本回合闭合该唯一
画像并推进完整回放，不缩小完整目标、不重复已完成基线。

## 完成结果与当前交接（2026-11-03 00:56）

唯一回放于00:48前后完成4000请求，00:50确认实际GPU释放，随后完成共同
SLO、全部请求阶段、画像绑定、双端交付和资源闭合。没有第二次性能运行。
使用monitor-experiment监测原handle，使用analyze-results按冻结V1/单轮V2
区分观测和归因；没有运行间CI、结果筛选、数据生成或生产算法修改。

| 3B W2观测 | D1020：32槽 | D1022：活跃LoRA专门化 | 变化 |
|---|---:|---:|---:|
| 原生完成/失败 | 4000/0 | 4000/0 | 不变 |
| 联合TTFT/TPOT达标 | 3746（93.65%） | 3770（94.25%） | +24请求，+0.60个百分点 |
| 生命周期物理GPU-s | 8135.629131 | 8062.415716 | 减少73.213416（约0.90%） |
| GPU-s/offered-request | 2.033907283 | 2.015603929 | 降低 |
| P95 TTFT (ms) | 1940.053078 | 1997.601528 | 增加57.548450（约2.97%） |
| 平均TTFT (ms) | 717.382658 | 825.647517 | 增加108.264859 |
| P95 TPOT (ms) | 28.988171 | 26.119779 | 降低2.868392 |
| 平均TPOT (ms) | 16.490769 | 15.491261 | 降低0.999508 |
| 平均E2E (ms) | 2524.890020 | 2542.603823 | 增加17.713804 |
| P95 E2E (ms) | 5109.083131 | 4899.350524 | 降低209.732607 |
| 原生按需load数 | 1021 | 1040 | 增加19 |

**决定：保留有改善也有退化的完整开发候选；未达标，不宣称G1/G2最优。**
联合要求仍至少3800，差30条；U虽低于12083.063870预算，不能抵消服务要求。
输入/原生输出/adapter绑定核验通过不等于独立全量数值正确性证明；后者仍
`n_correct=null`、`ranking_eligible=false`。D1020与D1017等旧结果全部保留。
不能把D1022与其他版本W0/W1拼成同一冻结工作点，也不重跑同键挑赢家。

### 阶段解释及下一项

- 边际TTFT达标3839、TPOT达标3925；违约为TTFT-only155、TPOT-only69、
  both6。对应上一点为140/102/12；TPOT改善并未消除TTFT问题。
- 同请求配对：联合fail→pass133、pass→fail109，净增24；TPOT为89/50，
  TTFT为58/67。两方向完整保留，不将133写成净改善。
- 六段TTFT平均差依次为：arrival→admission +123.432358 ms，
  admission→acquired −3.243115，executable→engine −1.682319，
  engine-entry→queue −4.940196，native queue +0.159438，prefill −5.461307。
  恒等式最大误差0；nested routing不重复相加。
- 第一500请求phase平均TTFT增加972.013816 ms；其余七组变化分别为
  −48.579180/−35.129907/−40.131837/+10.759555/+2.321203/+17.864971/
  −12.999751 ms。联合数459/476/483/476/477/467/470/462。
  初段等待集中了平均TTFT退化，不能据此删除启动请求或宣称已证明kernel因果。
- dispatch前tier为GPU2955、HOST12、NVMe540、remote493；100个service类、
  52个preparation类实际使用，全部在新D1021画像覆盖中。4000条D747 fresh
  descriptor SHA一致。没有新增实际kernel-grid/capture-count测量，不虚构。
- 外部比较继续复用D810全部点：vLLM W2为100%联合、16110.752 GPU-s、
  P95 441.635 ms；ServerlessLLM为98.125%、16084.201、480.724 ms。
  D1022节省资源但服务质量仍落后；这两个基线点又超75%预算，不能冒称
  同预算可行对照。其余低SLO/失败、后端适配范围和数值证明限制不删除。

下一步仍是3B主线：先复用本轮phase/原生启动与request阶段记录，区分
初始ready前等待和后续TTFT-only违约的实际事件，再对照已关闭方向、历史
代码及官方实现确定一个新可证伪候选。不能仅因差30条而扫描参数或重跑。
不重开无新前提的控制延后、扩缩容搜索、token/chunked预算、文件优先及
HOST图方向。达到可用工作点后推进同版本W0/W1与完整外部比较。

### 来源、闭合、存储与剩余范围

- 测量源码：4fff8fe487e2ab4b19225a50a82d346f3789570d；配置及D1021身份见下。
- 原始结果SHA：c4910ebfd3f3b61981f2df3c3fcd94233f5060d4b601167dbea8299acfcac317。
  launch SHA：4764fa0f3fb0a2b8bf1007acb8527448d53d3610dfec82dcaf4c1eb04755ea8c。
- 审计：`paper_results/ieee_tc/p2_backend/d1022_3b_active_lora_w2/completion/`，
  含full4000、stages、paired_d1020、profile_bindings、native_descriptor、
  delivery_and_resources、closure、manifest和复用收集脚本snapshot。
- 503次传输/1167457984线上字节，双端ID/字节/归档SHA/无请求打包全部匹配；
  远端既有服务未重启。没有连续远端CPU监测，不能声称排除全程远端干扰。
- 自然scale-up2、scale-down1，3个物理allocation全部释放。service/watchdog/
  replay均返回0，资源域消失，实际GPU compute为空。4045个资源采样无foreign/
  escaped；high/max/OOM/oom_kill/swap为0。服务峰值30623866880B，主机最低
  available94000615424B，最低磁盘144850964480B，无运行期保护事件。
- 最后离线收集使用3/4GiB、swap0、固定辅助CPU、CUDA隐藏；快照峰值
  531316736B、high/max/OOM/swap为0，scope已inactive/dead。前面短分析未保存
  退出前峰值，不扩写完整资源证明。一次配对CLI误传JSON而非目录，未产生
  输出；修正参数后通过，未改analyzer或测量值、未重跑GPU。
- 147项旧投稿保护文件与两个用户dirty SHA均不变；只新增本目录和交接。
  复用收集脚本按实际runner的per-run缓存覆盖路径检查已清空，没有按YAML
  默认路径冒称实际清理。未进行维护与GPU测量并行。
- 工作缓存释放后磁盘约149.84GiB，略低于下一重任务150GiB启动线；下一次
  构建/GPU前按现有存储回执做有界安全压缩，不另起无边界扫描，不删唯一证据。

全部两模型同版本W0/W1/W2/G1/G2、A1–A5、S1–S13、数值正确性范围、
B1/B2/C1/C2/C3和两轮审稿文档、图表及最终V2集成仍未完成，目标保持active。
不画图、不改论文、不生成新工件/trace，不接管共享k3s。

00:58后置备份：12文件结果检查点`d0dbd977edf0816dc7e65ad58b771ca088ef2177`
已推送并经ls-remote核对origin/retry14_continuous_queue_v2和faaslora_origin的
`backup/d1022-active-lora-w2-20261103`。后者是回退备份，不冒称已分歧V2
分支的最终集成；未强推。10项curated manifest校验、diff检查、staged凭据
模式检查通过；147项保护与两项用户dirty不变。无新生产代码，所以复用此前
针对性测试及本次实际严格分析验证，不增加重复GPU smoke或广泛资格测试。
当前没有GPU/离线分析进程运行。此后置文字在下一正常检查点提交；用户dirty
EXECUTION_STATUS与manifest继续不触碰。先有界存储维护，再沿上述3B问题推进。

## 问题与保持不变的部分

D1020已在4000请求上观测到3746联合达标（93.65%）、8135.629131GPU-s、
P95 TTFT 1940.053078ms，仍少54条达到3800门槛。D1021测试原生vLLM
specialize_active_lora=true：32槽的驻留容量与当前活跃LoRA计算规模分离。
历史、官方源码与可证伪假设详见D1021，不重复已否决的控制延后、文件优先、
扩缩容参数、HOST图、token预算及槽数搜索。

本次只相对D1020改变原生specialization选项及其必要新画像/身份；请求并发8、
GPU槽32/CPU槽32、其他backend配置、D747 fresh descriptor、stdlib gzip、
路由/handoff/residency/admission、扩缩容和IEEE九公式保持。D1021画像启动
增加16.228秒、KV减少349块，完整运行必须包括这些代价。官方捕获集合不是
实测kernel证据；不在本次顺便裁减捕获集合或修改策略。

完整原4000请求、500-adapter universe、W2 rotation100映射、arrival/prompt/
token、fixed_length_greedy_v1、业务前60秒通知和真实已发布远端合同不变。
源seed42是已知开发回放，不冒称独立留出；同键单轮、无CI、不筛赢家。

## 冻结输入及静态检查

- PLAN SHA：d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
- 指标V1 SHA：5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22。
- 单轮V2 SHA：193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
- 配置：configs/ieee_tc/20261102_d1022_3b_active_lora_w2.yaml，SHA
  f8d4ac96c9121dba134e2f3ec5e6a9e354b09dcf44447bb5cc65ecefc8fda171。
- D1021 manifest SHA：1fa53f43ffec59d472fbe6fdc070b02ca63d60f9e0564405708cefa2df7281a2。
- 服务画像SHA：e846698d6c8cb3aa0d94c31b92fcbf65e852a5962ad9e6fb86c7f8cd955ef643。
- 准备画像SHA：b35a6d32b36b445fcadbff301794159165fe2263d24f7f70ec1f059feaa5fc12。
- 独占raw：/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1022_20261102/d1022_3b_active_lora_w2_full。

CPU-only实际ScenarioRunner组装和两个画像加载器通过：160服务类、96准备类，
模型/画像身份一致；关闭specialization时两个加载器均拒绝。父配置差异检查
确认只有目标字段、画像及专属输出路径变化。该检查未运行GPU；3/4GiB、swap0、
辅助CPU2/3/26/27，完成退出，未保存终态资源峰值。D1021已覆盖变更的原生
混合批路径，不再追加无问题目标的100请求Full。

## 唯一完整回放入口

```bash
env -u FAASLORA_TC_CORE_TIMING -u FAASLORA_TC_CORE_STEP_TIMING -u FAASLORA_TC_HOST_INVENTORY_TIMING \
  FAASLORA_IEEE_HOST_INVENTORY_MODE=descriptor \
  FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so \
  FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044 \
  FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=stdlib \
  FAASLORA_D1000_RUN_ID=d1022_20261102 \
  FAASLORA_D1000_RUN_TAG=d1022_3b_active_lora_w2_full \
  FAASLORA_D1000_CONFIG=configs/ieee_tc/20261102_d1022_3b_active_lora_w2.yaml \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

入口旧文件名含W1，实际固定W2；无prefix变量。测量前提交备份、即时安全检查；
服务72/80GiB、swap2，辅助3/4GiB、swap0、固定CPU和独立watchdog。空机内存
≥102GiB，磁盘≥max(150,100+1.5×32)GiB；运行期16/100GiB保护不变。
不与压缩、构建、推送、其他GPU测量或远端配置操作并行。复用既有健康远端，
不接管共享k3s。无连续远端CPU监控，如实保留证据范围限制。

## 完成判定、解释与后续

先核验实际资源释放，再以现有audit_d703显式指定本配置/D1021 manifest，
恢复共同D648 SLO、4000请求阶段分解、原生绑定/输出计数、真实远端传输与
物理GPU积分；不使用runtime旧5000ms显示替代冻结评价。G1全部正确及至少
3800联合达标后比较资源；G2还需U≤12083.063870GPU-s，再比较W2 P95与
观测SLO差。独立数值正确性未知仍记n_correct=null，不偷换正式排名资格。

与D1020和D810全部baseline观测一起看，不仅与自己比；vLLM W2为100%联合/
16110.752GPU-s/P95 441.635ms，ServerlessLLM为98.125%/16084.201/480.724ms。
它们超过当前75%预算，不把这些点直接称同预算可行对照，也不隐藏其更好延迟。
其他基线低分/失败及适配局限全部保留，不为追求95%重开其优化。

结果若退化则保存并否决，不重跑同键；结果改善也需满足完整目标，不能只凭
内核开关或画像通过宣称最优。达到合适工作点后推进同版本W0/W1及比较，
不陷入局部无限优化。全部两模型最终G1/G2、A1–A5、S1–S13、数值正确性
范围、B1/B2/C1/C2/C3及两轮审稿文档、后期图表和最终V2集成仍未删除。
当前不绘图、不改论文、不新增工件/trace，不覆盖历史结果。

## 运行中交接：同一handle继续监测

测前11文件检查点提交为4fff8fe487e2ab4b19225a50a82d346f3789570d，已推送并
ls-remote核验origin/retry14_continuous_queue_v2及faaslora_origin的
backup/d1022-active-lora-full-entry-20261102。后者为可回退备份，不冒称已有
分歧的FaaSLoRA V2最终集成。9项来源引用SHA、147保护文件、staged diff和
凭据模式检查通过；用户manifest和EXECUTION_STATUS均未stage、SHA不变。

23:40:43测前可用磁盘161605341184B、MemAvailable109077004KiB、GPU compute
为空；gated-launch随后实际放行。历史D998 aux scope TasksCurrent=0，非
竞争中的旧实验，未清理或重启。远端健康服务直接复用、未操作k3s。

23:41:18 tmux `tc-d1022-3b-active-lora-full-w2` 启动上述唯一命令，并额外
显式unset PREFIX_COUNT、NATIVE_PROFILE和DIAGNOSTIC_PREFIX_COUNT，确保不是
短回放或详细插桩运行。实际句柄：

- aux：primelora-tc-aux-7449712f95714ef19a0a7592fae8ca2d.scope；
- service：primelora-tc-svc-3c1fd5edadf8480fbd6c667b3698043b.scope；
- service InvocationID：f3fecea21c3c4df0ab6f5e8ffec736d1；
- 原生GPU worker PID2317469，GPU0，样本52时8324MiB。

样本52服务active/running、154tasks；memory峰值13058338816B、主机可用
102050328576B、磁盘161604939776B，high/max/OOM/oom_kill/swap为0，foreign/
escaped为空，无警告或中止。这只是启动阶段观察，不是全程资源或性能验收。
本追加为轻量运行中交接，待本轮资源释放后再备份，测量期间不推送/压缩/构建。

继续同一raw目录的launcher.log、launch.launch/service.log、watchdog.jsonl、
physical_deployment及最终launch.json；以实际终态与allocation释放判断结束，
不能因观察超时重新启动。完成后按上述步骤闭合、与D1020比较、保留全部失败/
代价并决定下一项。完整目标仍active，尚未满足最终完成条件。

### 23:51持续运行核查

本回合为verified wait：实际tmux、service scope及原InvocationID均live，
未重启、未新开候选。最新Live为arrived436/4000、done433、ok433、fail0；
两个runtime，累计scale_up2/scale_down1。这里只是运行进度，不替代原生计数、
独立正确性或共同联合SLO验收；开头请求的启动等待完整保留。

watchdog样本588仍无high/max/OOM/oom_kill/swap、foreign compute、escaped或
保护告警；当时主机可用约91.806GiB，服务峰值16.195GiB，磁盘147.556GiB。
运行期磁盘底线为100GiB，故不因跌破下一任务的150GiB启动线而中断有效测量。
待本次释放后、下个重任务前再按现有回执压缩已关闭证据；不在GPU运行中维护。
本回合仅轻量tail/状态读取和此交接追加，没有配置/源码/远端变更、推送或绘图。
继续同一D1022至终态，完成前不进行完整结果分析或配置选择。

### 2026-11-03 00:01持续运行核查

继续verified wait，原tmux与service InvocationID保持live。最新Live记录
arrived1052/4000、done1050、ok1050、fail0，两个runtime；仍是11月2日登记的
唯一完整运行，不因跨日换键。watchdog样本1167：主机可用91.636GiB、服务
峰值17.785GiB、磁盘144.969GiB，high/max/OOM/swap/foreign/escaped和告警均0。
这只说明截至该时点的运行进度，不是整轮原生/正确性/SLO验收。无代码或配置
改动、无维护或备份并行；下一动作仍是监测同一运行至终态，然后离线闭合。

### 2026-11-03 00:12持续运行核查

本回合仍为verified wait，不是新增实验或性能结论。原tmux/service及
InvocationID f3fecea21c3c4df0ab6f5e8ffec736d1均live。最新Live为arrived1884、
done1883、ok1883、fail0（总计划4000），两个runtime、scale_up2/scale_down1。
watchdog样本1843：主机可用91.315GiB、服务峰值21.424GiB、磁盘141.447GiB；
high/max/OOM/swap/foreign/escaped及告警无记录，运行期底线未触发。
只做了轻量监测；8次45秒间隔的只读轮询工具已正常结束，**被测实验并未结束**。
不要把轮询工具完成视为Full终态；继续原tmux和raw，未改配置/代码/远端服务。
