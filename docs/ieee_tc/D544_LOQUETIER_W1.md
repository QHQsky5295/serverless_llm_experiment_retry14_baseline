# D544：Loquetier 7B W1 单轮观测登记

2026-10-15，启动前登记。沿既定主线完成尚未测量的突发负载点，不重做
D445的W0、100/500-ID/1000请求资格，不搜索基线性能参数。

## 不变的实现与输入

复用D445的原生Loquetier、Python3.9环境、SMLM、FP16、batch2、context1024、
四个125-ID静态分片、顺序启动及原生异步批处理。静态映射不加入PrimeLoRA
路由/准备策略。本机静态分片部署的适配范围继续公开，不声称原生集群调度。
旧W0的较长注册前等待仍是待解释的观测，不能无证据归为原论文机制缺陷。

旧测量视图失效仅因公共source adapter新增HydraServe/W1支持；原生源码、
测量代码和kernel未变。对照D445执行的baseline commit311663aa，24个相关
函数/类（含完整resident service）AST一致。复用既有source-adapter函数建立
新视图，生成代码SHA与旧视图一致，未关闭来源检查或更改原生策略。
39项CPU source-view/transport/worker/lifecycle检查通过，2.565秒；不是新的
GPU数值或性能证明。D541公共pidfd计量修复沿用，不重复测试或回退。

- 原始配置SHA：d9f3b81d3d90340739ec66b45d36c1c71c706640dd2cd790a19d3ea4a80a617a。
- 新运行配置SHA：ecfc72f92175fc8d115965129602ec4c242facfa47370b928ae55a92bd7598bf。
  仅source_view路径与其manifest SHA不同，原生配置值不变。
- 新测量视图SHA：6bf4e21d78ded9aedfa3a0f54060a6afc9c1072346d100ed497fa5fb584dcc36。
- HTTP配置SHA：ad2868ca14b33af9620807e3ec073ece1a83fb0302abe5cbecba6a53d1747440。
- W1视图SHA：103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3。

源trace/工件池不变，4000请求、132个实际请求adapter、458224目标输出token。
W1为8×500、phase内到达间隔减半、phase间30秒；fixed_length_greedy_v1不变。
通知后准备完整静态500-ID池，业务在通知后60秒开始，不根据ready平移。
保留498零权重/2非零ID的工件局限，不冒称500个独立训练权重。
开发身份41、formal_run=false；输入已知，不能改名为独立留出实验。

## 安全、远端与复用

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d544_20261015`；
新私有下载缓存`/tmp/tc-d544-7bl`，tmux为`tc-d544-loquetier-w1`。
复用现有受限runner、监控、分析和远端管理脚本，不构建第二套实验框架。
服务72/80GiB、swap2GiB；外置3/4GiB、swap0；CPU集合、102GiB空载启动
内存、150/100GiB磁盘门槛不变。新增峰值预估20GiB，实际启动前重新检查。

D543已回收封存D542的重复下载对象，不重复清理；所有原始成功/失败证据
保留。旧投稿147文件与用户修改不变，不stage用户manifest。启动前按明确
文件备份并核验两仓库远端SHA，记录现有Triton缓存，不额外预热GPU或清全局缓存。

真实远端仍用D78/D80一次性只读gzip；不重新发布、不在请求中打包。
真实读取、传输、竞争和本地准备完整计入时间，远端不在测量期重启或清理。
只启动/关闭准确身份的本轮服务，既有无关k3s保持不变；同一时刻一个重型任务。

## 验收及后续

完整回放结束后先确认所有原生worker实际退出及GPU释放，递归空辅助域关闭，
再停止本轮远端、保存双端日志。使用现有
`loquetier_static_shards_v1/full4000_w1`严格分析器与D394共同SLO；逐请求
核对原生数量、实际forward绑定、生成合同和完整物理积分。未知独立数值
正确性保留未知；不能以HTTP成功替代。失败及未达标都保留，不放宽阈值。

联合达成至少3800/4000才具备主SLO可行性；G1/G2仍采用冻结V1/单轮V2，
不因观测延迟差而换指标，不计算运行间CI。保存同输入比较和部署适配限制。
本点完成后继续HydraServe W1，再推进W2/G2、PrimeLoRA有证据优化、3B、
消融/机制/敏感性及论文文档；全部剩余范围不删减。接近目标前不画图。
