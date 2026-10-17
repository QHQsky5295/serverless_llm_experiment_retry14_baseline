# D632：最小双副本的 7B 固定资源工作点

2026-10-17。阶段：CPU资格通过，待备份后启动；尚无新性能结果。前置为已备份的 D631
`6b5a57ce9971b7e3ff5febdae828aec59f552de1`。D628 已精确撤回；D606/D610
文件可见性及 D618 合法设备域修复保留。不是再做 HydraServe 平台适配。

## 问题、假设与边界

D621 的 W2 联合达成 3798/4000=94.95%，未满足共同95%要求；D629
私有校验候选退化至3727/4000，同时GPU时间增加。D630发现其新增首token
延迟主要落在许可申请至恢复阶段，并与真实启动区间高度重叠；这不证明
全部等待均由容量不足引起。既有 D342 已发现同类问题，不重复称新发现。

可证伪假设：在不改九个公式和全部机制的条件下，将7B最小副本数1改为2，
可减少缩到单副本后再次启动期间的等待；是否提高共同SLO并守住GPU预算
必须由完整回放决定。代价包括初始并行启动、额外驻留以及后续控制轨迹变化。
不能用旧轨迹约113秒单卡时间估计新反事实资源。尤其 D619 的W1预算余量
只有约236.675 GPU-s，双副本下限可能使其预算不可行。

这是明确的按模型固定资源配置，不包装成自动学习公式或新颖算法。保持
max_instances=3、物理四卡可见、TP=1、14请求容量、批token1024、38秒回收
冷却、45秒TTFT窗口及所有profile不变。不按场景名、阶段或未来请求切换。

## 原始来源与实现语义

- [Knative scale bounds](https://knative.dev/docs/serving/autoscaling/scale-bounds/)
  区分持续下限与初始化规模；它说明配置语义，不证明本项目应取2。
- [ServerlessLLM 官方历史实现](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)
  对desired实例应用min/max边界，并区分starting/ready。这里只借鉴边界与
  状态所有权核查，不移植其控制公式、不改基线配置。
- 本项目 `IEEEReplicaControl` 保持原max-score与all-low回收式；只有实例
  下限参数变化。初始owner同时预留两个不同设备，第一ready可以服务，另一
  initial仍计pending。不能声称两副本都在业务开始前ready，也不能把晚ready
  的initial改名为natural_scaleout。回收不得低于2。
- 逻辑ready+pending上限3不等于瞬时最多3个物理持卡者；draining重叠仍
  按实际四卡设备域及物理区间并集计入。D618修复不撤销。

## 最小资格与完整回放

复用D621输入/环境/profile/保护核查、D615实际controller枚举，以及现有
`IEEEControlContract`、`IEEEActualControl`、`IEEEInitialObservedControl`、
`IEEEInitialDeployment` 测试。新检查覆盖双initial、第一ready不等待另一
initial、pending不算ready、兄弟/晚失败、取消所有权、min2回收边界。
无改动的D631 458项回归只核验SHA并复用，不反复执行。

资格通过、实际CPU域关闭并checked备份后，使用现有受限runner执行唯一
新键 `d632_20261017/7b_min2_full_w2` 的完整4000请求。复用W2视图与工件，
不新生成数据，不开详细observer。真实远端复用D78/D80只读发布缓存，
请求中不打包，读取/传输/竞争/本地准备保留在延迟里。

性能仍按冻结V1+单轮V2：共同TTFT/TPOT、至少3800联合达标、真实GPU时间；
W2预算12086.669343876 GPU-s。保留全部失败、数值身份未知与同后端输出变化。
N_correct未获独立证据前保持null，不以原生token和adapter名称代替数值证明。
本次为已知seed42回放上的开发观测，不改名独立留出或统计显著性。

若W2达标，再用同一min2/max3配置完成W1/W0；不能拼接旧min1配置的结果。
若未达标或资源代价不合理，保留结果并回到有界阶段诊断，不重跑同键择优。
不增加无新假设的锁微测，也不提前把单场景改善写成跨场景最优。

## 保护、备份与后续

实际资格：32项测试/192控制状态通过，复用458项SHA绑定的回归；未启动GPU。
第一次入口未保护multiprocessing导入，导致重复创建资格身份和进程池失败；
原入口/日志/失败说明保留。按确切InvocationID中止后递归空关闭，peak
1691213824B/events0/swap0。第二入口只修正主模块导入边界、使用独立证据名，
不改配置/测试断言/生产。成功InvocationID cbe20344c90a4db0b690843f5ee9bd2d，
inode26883701，递归空并关闭，peak1975480320B/events0/swap0。
这些是实验入口与正确性资格，不是新的GPU性能实验或系统收益。

只新增配置、run-key入口与小型资格/管理包装，不改生产算法；所有本地编辑
采用apply_patch。原始结果在gitignored新目录；小型证据进入
`paper_results/ieee_tc/p2_backend/d632_min2_full_w2/`。
启动前147项历史SHA和用户manifest、CPU内存/磁盘gate、真实空闲GPU、旧域
关闭均须通过。单重任务；运行中不扫描/压缩/构建/推送/维护远端。
安全域沿用服务72/80GiB、辅助3/4GiB和冻结CPU集合；原磁盘150/100GiB不变。

所有7B跨场景/正确性、3B、dLoRA限定对照、A1–A5、S1–S13、CE/成本、
两轮审稿文档和最终学术图表仍未整体完成。接近目标前不画图，不缩减计划。
