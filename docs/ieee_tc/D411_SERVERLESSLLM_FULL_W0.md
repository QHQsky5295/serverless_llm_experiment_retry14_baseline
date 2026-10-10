# D411：ServerlessLLM 完整 W0 单轮观测

状态：2026-10-11，启动前登记；尚无本轮性能结果。沿用 D410 已验证公开配置，
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
