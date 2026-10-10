# D411：ServerlessLLM 完整 W0 单轮观测

状态：2026-10-11 04:20（Asia/Shanghai）已结束：4,000请求全部到达并终态，
3,997响应通过原生合同核验、3次本机转发失败。不是G1/G2合格点。
原tmux为`tc-d411-7b`，不得重复启动。main执行版本
`896af899396983350c4544a9d0277283bc28339d`已推送并核验远端SHA。
沿用 D410 已验证公开配置，
仅将原始 1,000 请求验证推进至同一源文件的全部 4,000 请求。不增加参数扫描，
不重复短资格检查。PrimeLoRA D371 已封存点保持不变。

## 研究问题与复用依据

D410 已完成原生请求、内容、计时及物理资源校验；共同联合 SLO 为91.5%，
所有85次违约来自TTFT。较快解码没有转化为四卡store池的资源释放，因此
必须测完整业务窗口，而不能从1,000请求线性外推GPU-s或与PrimeLoRA全量比较。
本轮保留启动与所有轮换阶段，回答完整W0下的服务质量和资源占用是多少。

复用D406全500逻辑ID功能证据、D409有界独立参考、D410兼容性验证，
以及D78/D80不可变只读交付缓存。不新增权重、trace或解压工件池，不再次
发布远端缓存。真实请求读取、传输、竞争和本地物化计时保留，请求路径不打包。
原始未知数值正确性标志不改写；500逻辑ID不等于500独立训练权重。

## 冻结执行合同

- 原始seed42 W0、development identity41、非正式完整工作负载观测，
  `request_count=4000`，无adapter remapping或arrival变换。
- 既有runner内部仍使用`trace_prefix_v1`选择接口；分析器必须核验
  `count=source_count=4000`、完整请求集合、W0和rate_scale=1，不能靠标签判全量。
- `fixed_length_greedy_v1`、同prompt/target/token规则，60秒部署通知、
  open-loop、1,800秒开发资格请求保护；不按ready平移业务。
- 原生vLLM0.10.2/FP16/store loader/round-robin，已披露的轮询与身份队列修复。
  min1/max4/target4/keep10，max-seqs4/max-LoRAs4/CPU-LoRAs4，token预算1024，
  GPU内存比例0.72，`enforce_eager=false`。本轮不改任何服务算法或公开参数。
- 该公开执行选项依据[固定版本官方文档](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html#enforce-eager)，
  原生Graph路径已在D409/D410实测；不因版本号直接推断性能。
- 本轮LoRA缓存冷启动。保留原生编译磁盘缓存：3,858文件/104,734,877B，
  聚合SHA `5a75a00e9e341e6ee53e5afb2195c4a193a010c6d3446adcd7c4dfc10ca50791`。
  这不是HOST/KV/GPU warm状态；不预启动引擎，实际编译和capture开销照计。
  不声称与历史编译缓存初态完全相同。
- 完整计划SHA `f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5`。
  冻结V1与单轮V2共同生效，D394共同SLO不变。一个执行键一轮，无运行间CI。
- baseline执行版本 `2a5dfd1154ba93c82d9b7076fde0a8521543933b`；
  本轮源码视图四个实际源码SHA与D410一致。main版本在原始PROTOCOL中登记。
- 原始目录 `/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d411_20261011`。

## 验收、范围与安全

先校验全部4,000原生请求、输入/adapter/内容关联、原生token、时间分解和
物理allocation/release，再按共同TTFT/TPOT判联合SLO，报告GPU-s和原始延迟。
全量8×500源顺序片段只作穷尽诊断，不去掉启动、不充当独立重复。
低于95%也保留完整观测，不因胜负重跑或删除。G2、W1/W2和跨场景最优仍开放。
跨系统比较另外核对全部合同；不能用同是4,000条代替公平性检查。

服务72/80GiB、swap2GiB；外置回放/监控4GiB、swap0；既定CPU分离，
一次一个重型任务。启动磁盘至少150GiB且满足100GiB+1.5×预测新增量，
本轮预测新增32GiB；运行停止线100GiB，内存防护不变。
业务中不管理或哈希扫描远端，不清缓存，不开展并发构建。

此前只回收D410核验后的600个下载副本，实测释放1,476,268,032B；
原池/远端缓存/原始日志/收据不动。删除前原文件SHA与不同inode、所有权、
可读进程引用及未来配置引用检查通过；三个不可读同UID进程条目如实记录。
两份cleanup回执保留在D410原始目录，历史audit/manifest不改写。

现有分析器新增显式full4000校验，19项CPU测试通过（0.095s、4GiB/零swap）。
不重建实验框架。收尾顺序：GPU实际释放→恢复overlay→按InvocationID停止
本轮远端服务→保存worker/resolver/远端证据→全量核验→结果备份→7B下一基线。
保留所有失败；未获得跨系统证据前不重调PrimeLoRA，不跳3B或消融。

## 启动证据（不是最终结果）

完整source view核验4,000条、132个实际逻辑ID、458,224目标token；
到达窗口3,963.908903秒。4000条request contract已登记，业务按原始到达进行。
部署通知1166817.112284243，业务t0=1166877.112284243，同一monotonic时钟。
预检通过，147项历史保护结果与上一轮完全相同；原有用户replay修改仍单独保留。
远端链路1000Mb/s/full，health确认prepublished_gzip_v1/artifact_timing_v2。
服务和远端所有权、InvocationID及收尾指令记录在EXECUTION_STATUS和原始回执。
运行中不依据部分请求宣布达标、领先或完整复现。

## 运行中失败证据（03:56 检查，尚未终态）

截至2026-10-11 03:56:37，已提交2,846条，原始响应2,845条，其中通过响应校验
2,843条、HTTP失败2条。原4,000请求回放仍在进行，没有重启、重试或修改配置。
不能把当前成功子集视为完整达标结果，最终分母仍是4,000。

| 请求 | ingress forwarded（monotonic秒） | transport failed（秒） | 转发后失败间隔 |
|---|---:|---:|---:|
| req_02460 | 1169115.425325445 | 1169115.426527416 | 1.202 ms |
| req_02751 | 1169338.027686718 | 1169338.028803797 | 1.117 ms |

两条均在`model7b_repaired.ingress.jsonl`记录`ServerDisconnectedError`；
回放器收到502及`native transport failed; not retried`，没有隐式重试。
本机入口仅转发至`127.0.0.1:8348/v1/chat/completions`，不是artifact远端。
在此时保留的`model7b_repaired/serve.log`中，两请求ID均未出现；该日志包含
相邻请求的原生router处理记录。没有相应异常栈，不能把日志缺项当作完整抓包证据。

一个可证伪假设是连接保活关闭与复用竞争：两次失败分别发生在
`req_02449`与`req_02742`的ingress响应记录后5.000948及5.001383秒。
本轮共同native环境的aiohttp版本3.13.3、uvicorn版本0.41.0；实际源视图
`sllm/cli/_cli_utils.py:80`以`uvicorn.run(app, host=host, port=port)`启动，
未覆盖5秒keepalive默认值。现有测量入口使用默认15秒keepalive的
`aiohttp.TCPConnector(limit=0)`。官方固定版本分别确认
[客户端连接保活默认](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/connector.py)、
[服务端默认](https://raw.githubusercontent.com/encode/uvicorn/0.41.0/uvicorn/config.py)；
[aiohttp客户端源码](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/client.py)
只对幂等方法启用连接失败重发，POST不在其中。以上与实测时间特征相符，
但当前没有连接ID和socket生命周期记录，**不是已经证明的唯一根因**。
不能据此认定远端传输、GPU能力或ServerlessLLM调度机制导致失败。

待当前回放自然结束、GPU释放后，在既有入口测试中验证该假设；不在业务
期间执行测试、不变更连接策略、不增加自动推理重试、不重放这两个请求。
现有helper会把终态失败、原始响应及物理终态账本保留下来。全量成功分析器
必须拒绝此失败集合；若补离线失败分析，只扩展明确的失败分支，不绕过正确性
检查、不把缺失token或原生时间填成成功值。归因确认后再决定测量错误修复的
受影响范围，保留D411及所有失败身份，不为获得更优数字重复同一执行键。

此检查service内存约52.030GiB、主机可用52.882GiB、磁盘148.265GiB，
high/max/OOM/swap事件均零，外置watchdog未要求中止；这是当前采样，
不是整轮安全总结。下一重型任务仍须重新满足150GiB启动门槛。

04:11补充（仍在运行）：第三条失败`req_03635`，入口转发时间
1170337.970806340、失败时间1170337.971942494，间隔1.136ms，
异常类型`ClientOSError`（现有记录未保存errno/异常消息，不能补猜）。
它在`req_03628`的入口响应记录后5.000160秒发生；截至检查时原生
serve.log同样没有该失败请求ID。这个独立事件加强了5秒关闭竞争的假设，
但仍缺同socket关联，不能写成已完成根因验证。当前共3次失败，未重试、
未改运行配置。前述03:56检查保留为当时快照，不追改其计数。

## 完整终态与离线核验（2026-10-11）

| 项目 | 观测值 | 范围 |
|---|---:|---|
| offered / submitted / terminal | 4,000 / 4,000 / 4,000 | 完整原始W0 |
| 原生合同通过 / 失败 | 3,997 / 3 | 不抹去失败，不补发 |
| 原生实际输出token | 457,845 | 通过响应；失败目标共379不冒充输出 |
| 平均 / P95 TTFT | 950.811 / 730.631 ms | 3,997条通过响应，Type-1 |
| 平均 / P95 TPOT | 26.413 / 31.834 ms | 3,997条通过响应 |
| 平均 E2E | 3,940.954 ms | 通过响应的条件时延 |
| TTFT / TPOT / 联合达标数 | 3,939 / 3,996 / 3,938 | 联合分母4,000；TPOT条件分母3,997 |
| 联合达标率 | 98.45% | 三条失败均计未达标 |
| 生命周期物理GPU-s | 16,090.164120 | 四卡store所有权并集至真实释放 |
| GPU-s/offered-request | 4.022541 | 失败诊断，不冒充达标成本 |
| 真实HTTP次数 / 线上字节 | 132 / 131,480,060 B | 全部关联至已发布对象 |
| 核验物化字节 / 请求内打包 | 3,300,789,780 B / 0 | 本轮缓存内容已核验 |

共同阈值仍为D394，没有事后改变。原生时间线与E2E/TPOT重算误差均为0ms。
首500条联合达标445/500，其余每个固定500条片段为498–500/500；
这是完整片段诊断，不删除启动段，不把片段当独立重复。少量长启动等待使
平均TTFT高于P95，二者不矛盾。首次500条的均值为4,372.516ms、P95为
45,272.295ms。没有用短回放线性外推资源。

5个实际served epoch的ready-time身份可追溯，scheduler收尾时队列为空、
控制循环仍活跃且未记录异常。此证据不证明三次HTTP失败的唯一根因，
也不能把5个epoch称为5张GPU。物理四卡store池直到真实释放均计入账本。
互斥GPU-s分解：准备208.522207、到达窗口15,855.635612、drain3.784399、
清理22.221901，总和与物理并集一致。

**结论：**联合SLO本身超过95%，但未完成全部请求，`observed_feasible=false`、
`performance_ranking_eligible=false`。原始`n_correct=null`和未独立证明的
数值正确性标志保持不变。不能据此宣告PrimeLoRA优胜，也不能把三个入口
失败写成ServerlessLLM算法缺陷。下一步验证5秒连接关闭竞争，必要修正限于
已证实的测量入口错误；不得添加隐式重试或为更优数值重复本执行键。

复用原分析脚本增加显式`--outcome complete_with_failures`。默认成功分支仍
拒绝任何失败；完整分母、worker、原生合同、内容和物理账本验证均保留。
23项CPU测试通过（0.119s，4GiB/零swap）；随后本轮真实离线审计通过。
成功响应才读取原生metrics，失败没有填造token/时延，片段缺失字段为N/A。
零成功、缺终态或其他清理错误不能被标为成功。

分析：`paper_results/ieee_tc/baseline_audit/d411_7b_full_w0_failure/audit.json`，
SHA `eb00a0e729ac9e43b53f67494c25ac99425e72b0c5b88283b783f1517dee617f`。
同目录`ingress_failure_evidence.json`保留三次失败及有限的时间关联证据，
SHA `a6186d3168456deaeafc41670c92b509fb4047511cb74012894ca87e72c26021`。
`evidence_manifest.json`绑定234个保留文件，全部SHA复核；manifest SHA
`7de3171d8e8ea7655aa4a94965e80d69355d15bb49fd1a503ef5907309c2e00d`。
最初证据清单生成调用因系统Python不提供`hashlib.file_digest`而在写文件前
退出，随后改用流式SHA-256生成；不是性能重跑，也未覆盖任何结果。

推理GPU已释放、服务域移除、空辅助域停止；loader overlay按原收据恢复。
三项本轮远端服务核对InvocationID后停止、端口已关闭。远端传输/监控日志
复制后与来源SHA一致。保护清单前后相同，仅保留既有用户replay修改差异。
整轮high/max/OOM和服务swap均为0，服务峰值57,622,568,960B。
尚未回收本轮可重建下载对象；下一重型实验前必须重新满足磁盘启动门槛。
