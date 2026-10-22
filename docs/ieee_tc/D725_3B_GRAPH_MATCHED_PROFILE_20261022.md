# D725：图覆盖候选的同规格性能参数测量

## 主线与依据

D724完成16请求的原生图执行资格，不是完整性能胜点。D719最新完整3B为
joint91.60%、U13508.300779 GPU-s；当前问题仍是共同服务要求下的资源效率。
D723发现原生图覆盖边界，D724实际PIECEWISE重放通过，因此只继续这一候选。
不恢复已拒绝的LoRA元数据、HOST元数据、GC或控制串行候选，不扫描图上限。

继续遵循D724记录的vLLM v0.30.0官方图管理器和设计说明。图捕获改变启动、
空间、padding和执行成本；不能因请求长度相同就复用旧延迟。使用
serving-llms-vllm技能核查原生配置/监控路径，其通用吞吐示例不作为本机证据，
也不采用其中与冻结协议不符的500ms或GPU利用率评价门槛。

## 可复用与必须重测的边界

- 完整复用D701的24个既有代表、76波次、423请求引用、各层级/并发/长度
  类别与顺序；不生成负载、权重或整池副本，不新增类或按性能换代表。
- 复用D642已完成请求的原生prompt身份和固定输出数量，仅用以初始化
  admission的历史平均完成长度。固定输出数量由共同生成合同确定，图执行
  不改变该量；它不是时延、质量或数值正确性的代理。
- 对这一长度复用新增显式graph-only binding：只允许compilation_config
  中合法捕获尺寸/上限变化；源SHA、原生完成计数、prompt/backend/生成/容量
  等其余合同严格不变。自然停止、不完整计数、未声明及额外配置变化均拒绝。
- 新model_backend_id与profile_id从完整目标配置重新推导；编译字段不从
  FrozenServiceProfiles身份中排除，旧D701 service/preparation时间全部不复用。
- 重新测量原160 service类/96 preparation类，按原严格exporter和loader
  验证实际覆盖。未测出前profile_export_authorized=false，不用0/近邻补类。

## 实现与执行

小幅扩展既有prepare_d701_3b_same_spec_profile.py、run_d298_capacity_source.sh
和export_d701_3b_profile.py的显式路径/身份入口，复用既有native_source_matrix，
不建立新实验框架。只改资格统计绑定，不改九公式、Full算法或共同SLO。
新目录d725_20261022与d725_3b_graph_profile，所有旧D701/D724产物保留。

使用D534共同工具链合同和既有600秒启动保护；复用D78/D80真实远端只读gzip
发布。请求期打包必须为0；真实读取、传输、竞争和本地物化保留。启动前
检查原生1Gbps链路状态、共同远端服务配置和健康；运行中不管理远端、不
压缩扫描/备份或另启重型任务。结束后先验证GPU释放，再按InvocationID
关闭本次远端服务和monitor。不触碰非项目k3s。

CPU资格、物理72/80GiB服务与3/4GiB辅助限制、CPU分区、swap、watchdog、
150/100GiB磁盘和144旧投稿/用户manifest保护保持。使用run-experiment与
monitor-experiment的现有tmux/gated入口；不套用技能内旧conda环境示例。

## 验收与下一项

本次是开发profile测量，不是M1/M2或独立数值正确性证明。必须验证423原生
请求/传输、所有保留类、input/source/config/runtime SHA、物理释放及资源
事件，然后才导出新profile。按用户2026-10-22新增的“不增加无用门禁”要求，
本次已有D724执行资格与D725并发测量覆盖后，直接采用D719 Full策略的同一
图配置进行完整4000 W0，不再另加重复Full前缀。仍核对配置与profile绑定；
不以局部样本宣布达95%或更低GPU-s。

首轮选择了过广的launcher测试集，其无关测试启动多个原生import子进程，
受限CPU测试组memory.high触发54785次、峰值3434647552B，max/OOM/swap0。
已按身份停止全部8个所属进程；没有GPU实验或远端服务启动。保留原日志和
关闭回执，不扩大资源限额。改为本次直接相关的62项检查全部通过，峰值
763990016B、high/max/OOM/swap0。静态重绑定成功，24代表/76波次/423请求
与D701完全相同，completed-length means不变，新的model_backend_id为
native-model/d9d6ede9672526cf67b6570d78207ca90fc023e83b1439550e916dd40b440981。

W1/W2、其余3B baseline与dLoRA、A1–A5/S1–S13、CE、两轮审稿文档及最终
图表全部保留；接近目标前不画图。D724已备份至7beb43b0fe49af7b3d9e6da70f826795041b1961，
backup/d724-graph-coverage-20261022远端核验通过；V2既存分歧不强推。

## 完成结果

一次原生真实远端采样完成423/423请求；160/160 service类、96 preparation类，
415/332条对应观测。全部原生生成、内容、profile身份和loader检查通过，
未缺类、未插值、未复用旧时延。423次真实远端交付均为已发布gzip、打包0；
当前实测链路1000Mb/s/full。原始source SHA为
19f91a2302fa1aae3c33c01d1005983f43103ff8eb3ae1b5bf28812a51dc2fc6。

引擎启动97.698451秒；服务峰值5717614592B、分析780734464B，high/max/OOM/
swap均0。服务与原生GPU上下文已释放，远端两个artifact服务及monitor按
本次InvocationID关闭，服务端传输与监控日志已保存。144旧投稿、用户manifest
与冻结指标均不变。此处不将采样启动差异或profile均值解释为Full效果改善。

新service/preparation profile分别为b0d6d57a…/9080ca04…，严格绑定新图模型
身份；完整产物见export/和completion/。下一D726直接完整4000 W0，所有
非图控制逻辑与D719相同，不新增100请求资格轮。
