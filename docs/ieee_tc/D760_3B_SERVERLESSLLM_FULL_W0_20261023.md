# D760：3B ServerlessLLM完整W0单轮观测

## 最终观测：完整终态但含失败，不进入达标排名

14:24确认原运行结束，服务返回1（qualification_request_failure），实际
GPU/服务域释放；本轮launcher、分析域均按身份关闭，loader恢复，三个
本项目远端服务停止。原1800秒保护和资源配置全程未改，未重跑同键。

| 量 | 观测值 |
|---|---:|
| offered/submitted/terminal | 4000/4000/4000 |
| 原生响应/失败 | 2522/1478 |
| 有效原生输出tokens | 267459 |
| 联合SLO | 2304/4000 = 57.6% |
| 物理GPU-s | 23240.876161672175 |
| 到达前/到达窗/drain/清理 GPU-s | 209.640191/15855.635612/7135.328589/40.271768 |
| 条件平均/P95 TTFT（2522响应） | 24753.559038/3800.477953 ms |
| 条件平均/P95 TPOT（2522响应） | 17.920855/20.242837 ms |
| 条件平均/P95 E2E（2522响应） | 26590.937805/7373.388233 ms |
| 远端下载/线上字节 | 132/306360162 B |
| 失败请求作为下载所有者的字节 | 83552852 B |
| 请求期打包 | 0 |
| 服务peak/high/max/OOM/kill | 79027851264 B/9124680/0/0/0 |
| 原生计时重算最大误差 | 0 ms |

HTTP/native绑定不等于独立数值正确性：n_correct=null继续保留。
全部请求终结不等于全部正确完成，故workload/formal/ranking均false。
条件P95不包含1478失败，不能单独用它描述整轮服务或证明相对优势。

按源序号的八段500请求，响应数为500/500/500/500/489/0/0/33，联合
达标数447/468/485/495/409/0/0/0。后段33条恢复响应的平均TTFT约1784秒。
这是完整分段诊断，不删启动段、不把切片当独立运行或前段当新主结果。

### 分析修正与证据边界

原分析1在成功响应引用transfer应覆盖全部远端transfer的断言失败；该
断言不适用于已下载但HTTP超时的请求。保留原失败日志/分析SHA，仅在
complete_with_failures路径补严格的全请求下载账目：offered请求、resolver
事件、receipt哈希、内容清单、服务端transfer、时钟/字节一一核验；重复
下载所有者、未关联transfer、成功响应不一致仍拒绝。成功分析规则不变。
4000解析完成、132实际下载，其他均cache hit；未由成功子集推造全量消耗。
新增10项针对测试和83项相关回归通过。未改执行、生成、超时、阈值或失败数。

修正分析SHA e7d4561861e6f8ac6600a9bb5930bd25a7ca9af57b1a28d267e2a4b7207620d6。
原分析器SHA761ef6758ae85c665aa8ee38ef1a6fdbe740b2a7edb9a26912a9273b419b3a17，
新SHAe89ac9f8e79e3b42901bf6df8615b4842f7132adb0ebc0847dd27359f12d3156。
原脚本快照与失败回执保留。离线review内存high3149、peak3221745664B，
无max/OOM/swap；不混为被测服务压力。Ray日志归档并逐文件复核哈希，
原始日志和唯一工件不删。旧投稿145文件及三个用户修改继续校验保护。

### 尚未解决的归因，不以异常低结果支持PrimeLoRA

14:17–18内存下降时原四个backend和EngineCore仍在，不能归因于它们退出；
14:23出现尾部恢复，不能写成一直完全无响应。Ray日志有磁盘使用超95%
预警，但末次debug_state累计spill请求为0，预警本身不足以证明spill失败。
现有证据支持内存回收压力与长停顿并存，尚不能定位泄漏、原生机制缺陷
或适配错误。先完成一次基于现有日志的有界核查，再决定必要修复或继续
W1/W2；不调大保护/预算、不搜索baseline性能参数、不隐去失败。

以下保留运行前登记和运行中原观察，不追改其时间身份。

封存后存储维护仅删除本轮1056个已核对原池、哈希、引用和打开文件的
下载副本，释放5143777280B，余量161265479680B。保留所有原始失败日志、
工件、trace及稳定编译缓存。FD核查保留三个同UID不可读进程的范围说明，
不宣称全系统可见；旧投稿145文件与用户修改哈希均未变。该维护不是新的
性能实验，也不把删除空间写为系统节省。

2026-10-23，运行前登记。D756原生100、D758全500-ID功能、D759原1000
兼容均已完成，结果和下载副本维护已备份闭环；不再重复这些检查。本节点
执行缺少的原4000请求W0，随后同配置W1/W2，不以资格前缀推算完整结果。

## 研究问题与输入合同

测量完整工作负载下的共同TTFT/TPOT联合SLO和物理生命周期GPU占用，
区分启动早期、稳态、轮换与终态清理。D759的88.5%联合达成属于1000前缀，
不由该值预判本轮、不删早期请求、不因低SLO搜索基线参数。

- 原source42/development41，不是独立留出正式运行；保留1800秒共同验证
  保护和`formal_run=false`，不冒称已冻结所有参测系统正式超时证据。
- 原4000请求、500-ID工件宇宙，实际132个逻辑adapter、458224目标输出tokens。
  源SHA `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`，
  W0视图SHA `0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37`。
  复用D649/D753已登记映射，不重新生成trace、工件、prompt或target。
- `fixed_length_greedy_v1`、`role_lines_v1`、原open-loop到达，部署通知
  提前60秒；实际准备和持卡均计量，不平移业务到ready时刻。
- 使用D648共同3B阈值及D657 Resident参考，V1+单轮V2不变。原生token和
  adapter绑定逐请求核验；未知独立数值正确性保持未知，不凭HTTP成功升级。

## 原生配置及适配复用

vLLM0.10.2、FP16原生store、RR、原autoscaler，min1/max4/target8/keep10，
maxseqs4/token1024、GPUfraction0.72、GPU/CPU LoRA各4、maxrank64、
prefix/chunked-prefill开启、非eager。与D759四个实际native文件SHA相同，
无生产修改或参数优化；114项原生、89项3B输入、17项缓存所有权测试复用。
仅按已有入口机械改动campaign身份和请求数；AST/bash及原4000索引核验通过。

物理GPU独立稳定编译缓存继续正常复用，记录启动前完整清单；不是GPU预热。
本轮冷LoRA下载目录`/tmp/tc-d760-3br`，D78/D80已发布只读gzip远端复用，
请求期无打包；必要读取、传输、竞争、本地物化保留在时延中。backbone与
原native checkpoint复用，无格式转换或整池复制。

## 安全、执行及后续

MAIN执行检查点`90f7c4a5fdef53965704dcf7f8eee3359d164f74`已推送核对
`backup/d759-3b-prefix1000-20261023`，BASE为已备份
`2bdf53cb25040b98e84e587f421bd772528c96b9`。新机械运行配方在launch_gate
保存全文/SHA，结束后正常结果检查点备份，不声称旧commit包含新配方。

服务72/80GiB、swap2GiB，辅助3/4GiB、swap0，原CPU集合及外置保护不变。
启动实时检查102GiB可用内存、150GiB磁盘及32GiB预计增长；D759已完成的
存储维护不重复。单重型任务，性能期间无扫描/清理/构建/推送/远端修改。
原始目录`results/ieee_tc/serverless_qualification/d760_20261023`，tmux
`tc-d760-serverlessllm-3b4000`；是否实际启动/完成以回执为准。

终态后实际释放GPU、关闭本轮辅助域、恢复自己的loader、按InvocationID
停止远端、保存原始证据，再以原严格分析器`--population full4000
--model-key 3b`核验。失败保留并走对应分析，不补跑成功数、不降低分母。
保存/必要维护/备份后继续W1/W2；3B其他基线、PrimeLoRA跨场景/G1/G2、
dLoRA、7B最终同版本主表/Full、A1–A5、S1–S13、CE/Pareto、两轮审稿文档
及最终图表全部保留。共享k3s不动，完整目标仍active。

## 运行中观察：13:19–13:29 CST，非最终结果

原运行持续存活，InvocationID仍为
`ca02ee20b4af4b8481af1ca1e5ae998a`。约13:19开始出现共同72GiB回收阈值
事件，13:24前后本服务swap达到2GiB上限；没有调大限制、重启或追加回放。
13:29:25的sample2515：service 78673932288B、MemAvailable 31132930048B、
disk free 157611388928B、high 882783、max/oom/oom_kill均0、swap 2147483648B。
主机PSI full avg10为60.02%，服务为66.48%；现有保护未告警/中止，因为
主机可用内存尚高于既定24GiB组合条件。保护协议不在运行中改变。

13:25–13:26两个轻量观察点的日志尾部最后响应均为`req_02482`；13:29
已见`req_02779`创建，最近100行未见响应。尾部ID不是累计成功数；该观察
表示需要核对的响应停顿，不是最终timeout/失败判定。原生serve日志在
13:28仍报告四个ready实例及容量等待，store日志末段为13:06原生加载成功。
有限日志尾部无新异常栈不等于全量日志无错误，仍等待原请求保护和实际终态。

为保留瞬时压力证据，仅作本服务域的只读轻量检查（无磁盘扫描/哈希、
无调试器、无服务修改）：约13:27的memory.stat为anon 77630394368B、
file 699899904B、shmem 699826176B、kernel 281305088B。因此本快照不能
解释成主要由大量可回收磁盘page cache占用。约13:28的主要进程RSS如下，
单位KiB；RSS含共享页，不能将各行求和冒充物理总内存或PSS。

| PID | 原生进程 | RSS KiB |
|---:|---|---:|
| 986090 | sllm-store | 34255464 |
| 991443 | ray::VllmBackend | 9683628 |
| 990137 | ray::VllmBackend | 9664680 |
| 992035 | ray::VllmBackend | 9491004 |
| 1034817 | ray::VllmBackend | 5931104 |
| 1035548 | VLLM::EngineCore | 2279044 |
| 993003 | VLLM::EngineCore | 2250968 |
| 990784 | VLLM::EngineCore | 2229356 |
| 992311 | VLLM::EngineCore | 2180792 |

这些证据尚不能断言内存泄漏、核心机制缺陷或适配错误。运行结束并保留
全部请求/资源证据后，先有界核对原生store和backend内存的增长来源及
生命周期，再判断后续W1/W2是否受同一具体问题影响；不以低SLO为由搜索
baseline配置，也不因观察停顿而重启同一执行键。独立数值正确性、最终
SLO、物理GPU占用及完成比例均待终态后原严格分析，不能预先填成成功。

## 有界只读配置/源码核查：13:34–13:37 CST

未停止、重启或修改同一运行，未执行测试/扫描/内存剖析。实际store日志
`model3b_repaired/store.log:2,12`确认启动时预分配34359738368B（32GiB），
不是仅由脚本默认值推测；这能解释固定基线占用，不能独自解释后续增长。
实际配置仍为max_num_seqs=4、max_loras=4、max_cpu_loras=4，未启用
`trace_debug`（已执行backend默认false）。

检查的释放/持有路径：

- 已执行`source_repaired/sllm/backends/vllm_backend.py:222–245,529–551`：
  每请求只替换最新RequestOutput，正常完成且非trace_debug时删除记录。
  这不是证明所有异常/取消路径均释放，也不支持“所有历史输出无界保存”
  的直接断言。
- 当前测量层`ieee_tc_serverless_measurement.py:466–513,570–605`对单请求
  快照/时间线进行观察；ready回执仅传递ID和SHA，不在每请求内复制整份
  500-ID配置。此有限静态检查不等于动态内存正确性证明。
- 本机vLLM0.10.2 `TokenizerGroup`按max(max_loras,max_num_seqs)设置LRU，
  本配置容量为4；原生output processor在finish_reason非空时pop请求状态。
  因此不能仅由adapter数增长推断无界分词器缓存或原生已完成状态泄漏。
  已联网核对同版本官方[TokenizerGroup源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/transformers_utils/tokenizer_group.py)
  和[output processor源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/engine/output_processor.py)。

13:36:56仍为原InvocationID，sample2960：service78788952064B、主机可用
30952214528B、磁盘157604552704B，high2299647、max/oom/oom_kill0、
swap2147434496B；最近100行已见req_03216创建、无响应，不代表全量计数。
仍为内存回收压力下的活跃未终态运行，不能把存活写成正常推进。保持原
保护规则，结束后优先用已有日志检查增长与副本/请求/adapter的对应关系；
只有形成具体适配错误证据才修复，不搜索baseline性能参数、不重复已通过
资格、不由此次源码排查宣告“无泄漏”或“原系统机制缺陷”。

## 13:54–13:58 CST：末条到达后，已确认请求超时

13:54日志已有末条req_03999的request_created和http_headers_sent。
13:57尾部事件转为http_request_failed；13:58:01观察的最近100条均属此
事件，末条req_02668为TimeoutError，从http_task_started_s到
client_completed_s为1799.9998073079623秒，符合原1800秒验证保护。
这已证明本轮含失败，不能作为全正确完成的达标点；尚不代表全量失败计数
或已完成收尾。此前“最近100行无request_error”只是有限事件筛选，不能
解释为无失败：实际事件名是http_request_failed。临时监测摘要已纳入这个
名称及全部尾部事件直方计数，原始日志/运行/分析器均未改，不丢弃失败。

同一原生服务仍active/running且InvocationID未变。sample4207：
service78981144576B、MemAvailable30734753792B、disk157578522624B，
high5935058、max/oom/oom_kill均0、swap2147250176B；无安全中止。
只结束了临时观察循环，未停止实验或变更其超时/资源/配置。下一继续等待
原运行的全部终态与实际释放，按真实结局保存；若完整人口含失败，则用
已有complete_with_failures分析入口，不能改成成功、删掉停顿或重跑挑优。
内存压力与停顿同时出现，但根因及必要适配范围仍待终态后的有界核查。
