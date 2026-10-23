# D758：3B ServerlessLLM 全500-ID功能覆盖

2026-10-23。承接已通过并备份的D756原100请求；不重跑该点、不进行基线
参数搜索。本节点只补计划§6.7缺失的500-ID执行覆盖，不是完整W0性能排名。
实际启动/完成状态以原始回执及EXECUTION_STATUS为准。

## 已完成结果

同一轮自然结束，原严格分析器首次通过，没有推理重跑或放宽检查。

| 观测 | 数值 |
|---|---:|
| 计划/到达/提交/终态/原生合同 | 各500 |
| 不同逻辑adapter / 失败 | 500 / 0 |
| 实际原生输出tokens | 68,925，逐请求与目标相等 |
| ready且实际服务实例 | 4，物理CUDA0/1/2/3 |
| 真实远端交付 / 请求期打包 | 500 / 0 |
| 线上 / 物化字节 | 1,160,474,851 / 19,482,573,296 B |
| 共同联合SLO | 376/500=75.2% |
| TTFT达成 / 条件TPOT达成 | 377/500 / 495/500 |
| TTFT均值 / Type-1 P95 | 12,101.914 / 89,344.939 ms |
| TPOT均值 / Type-1 P95 | 16.635 / 22.261 ms |
| E2E均值 / Type-1 P95 | 14,518.237 / 94,910.515 ms |
| 物理GPU-s | 2,552.304434614256 |
| 时间分解 / TPOT重算误差 | 0 / 0 ms |

GPU-s互斥分项：pre-arrival60.06152611039579、arrival2442.4698638916016、
drain25.806846657767892、cleanup23.96619795449078，和与U一致。平均提交后
dispatch等待10,352.600ms，其中router队列均9,934.702ms；service TTFT均
1,739.089ms。覆盖视图每ID只访问一次，不把这些数与完整W0/PrimeLoRA直接排名，
也不将相对D756不同人口的变化归因为某一机制。

服务峰值77,310,423,040B，memory.high198次、max/OOM/OOM-kill/swap均0；
主机可用最低49,218,744,320B、磁盘最低142,338,260,992B，无安全中止。
内存回收/节流属于共同资源条件，不隐去、不因其存在删除本轮。离线核验域
另有high377、峰值3,221,733,376B，无OOM/swap，发生于GPU释放后，不混入
服务性能解释。service/launcher/analysis/result已按身份关闭；loader恢复，
本轮远端三个服务停止。145投稿保护项与用户修改未变。

当前结论仅为完整500-ID原生功能覆盖通过；`n_correct=null`、独立数值证明
未知、正式性能资格false继续保留。audit SHA
`072f4fd7881155a13191efd198a24e5177524f533af8d2d0d44690b4a658817b`，
curated位于`paper_results/ieee_tc/baseline_audit/d758_3b_native_coverage500/`。
不因75.2%继续基线调参；完成安全存储回收和结果备份后，直接进入缺少的
原1000请求兼容检查，再做完整W0/W1/W2。

### 已完成的存储收尾

仅回收本轮已关闭的`/tmp/tc-d758-3br/remote_artifacts/objects`：4,000个文件、
载荷19,482,573,296B；当前下载与保留原池逐文件SHA、大小和独立inode一致，
无软链/嵌套挂载、可读同UID活动引用或未来配置/脚本引用，receipt/journal/
run_contract保留且与证据副本一致。实测回收19,497,320,448B，余量
161,646,469,120B；下轮仍需实时准入，不以该时点数值代替。
沿用D532已验证清理器，未删除原工件、原始成功/失败记录、编译缓存或用户
修改。三个同UID不可读proc（2628、3360275、3361057）作为检查范围限制保留，
不声称特权级全机检查。维护空域已按身份关闭，145保护项未变。

## 输入与机制边界

- 复用D649/D753已经验证的`adapter_coverage_v1`索引：原trace前500条
  prompt、request ID、到达与目标输出不变，按静态清单排序各执行一个逻辑ID。
  500请求、500逻辑ID、68,925目标tokens；不生成新trace或权重。
- view SHA `4a9f63b08a600c40af7f5acd8066fb518aa21ed48ee8d5c75d7b3e2e8a51088e`，
  source SHA `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`。
  500逻辑ID不等于500组独立训练权重，不替代独立数值正确性证据。
- 原生vLLM0.10.2/store加载/RR/scaler配置与D756完全一致：min1/max4、
  target8、keep10、seq4/token1024/LoRA4/CPU4/rank64、图执行和prefix保留。
  D755稳定物理GPU编译目录继续复用，不清除编译缓存、不预热GPU。
- 真实远端D78/D80只读发布缓存；请求内不打包；私有下载目录
  `/tmp/tc-d758-3br`，服务实际持卡及60秒准备完整计量。
- 原114项原生测试、89项3B输入测试、17项编译所有权测试复用；四个实际
  source文件SHA与D756一致。没有生产修改，不另跑重复组件测试。

## 执行和分析

复用D756启动、guarded-launch、原生资格、remote管理、loader恢复、精确
空域关闭和既有D402分析器；仅机械替换campaign路径和既有coverage人口。
分析显式指定`--population adapter_coverage_v1 --model-key 3b`。
原始目录为`results/ieee_tc/serverless_qualification/d758_20261023`。

启动即时检查102GiB可用内存、150GiB磁盘和32GiB预计新增峰值；服务
72/80GiB、swap2GiB，辅助3/4GiB、swap0，单个GPU实验。不得与扫描、
压缩、构建、备份或远端配置修改并行。成功/失败都保留；低SLO不触发调参。
结束顺序：实际GPU释放→空launcher关闭→本轮loader恢复→按身份停止远端
服务→原分析器验证→封存/备份→必要存储维护→原1000兼容与完整主场景。

准备阶段第一次在解析时发现多余括号，未执行任务动作；第二次误把
FrozenReplayEntry中的adapter视为直接属性，在配置/模型视图创建前失败。
已保留原recipe及失败/释放记录；finish_prepare复用D753的source_json读取，
仅修包装取值。无模型启动、无推理重跑、无输入或测试断言放宽。

执行代码检查点MAIN `a49cd3f75d633bc5843f4542c136cf5e12ceb733`已备份，BASE
`2bdf53cb25040b98e84e587f421bd772528c96b9`已同步origin/main。D758机械
运行配方在launch_gate保存原文和SHA，下个正常结果检查点一起备份；不声称
这些新配方已包含在旧commit。用户修改与145投稿保护项保持不变。

共享k3s不动；HydraServe已完成隔离及7B结果复用。3B其余基线、PrimeLoRA
W1/W2/G1/G2、dLoRA、7B最终同版本主表/Full、A1–A5、S1–S13、CE/Pareto、
审稿文档和最终图表均保留。完整目标仍active。
