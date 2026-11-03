# D1045：按真实预留事件累计忙runtime时间

2026-11-03。唯一3B PrimeLoRA候选，父源码5e67916c77b16f2bd35116d58c582d7a71aab880。
上一D1044为PROGRESS：完整离线证据及两个仓库备份已完成，不是重复性能运行。
本文件接续用户dirty的EXECUTION_STATUS，后者不修改/暂存。

依据D1044：当前D1043在2→1缩容时，采样忙时间41.098417实例秒低于已证
下界48.291097，随后同83条请求的联合达标从78降至24，主要增加准入等待。
这是支持观测修正的具体新前提，不是完整因果证明。D997/D1037此前暂缓
事件积分的原因及本次重新检验依据见D1044，不恢复已拒缓存容量、预取或
控制调用延后方向。D1033仍为94.30%参考，D1043不晋升性能参考。

已核查be5ba93c忙实例服务率、67cbd79d预测释放、88178f60预测扩容及
b24dd5a1高压仲裁历史。再次联网核查
[Knative事件统计](https://raw.githubusercontent.com/knative/networking/main/pkg/http/stats/request.go)：
计数改变前先累计上一状态的时间。只借鉴观测方法，不复制参数/控制策略，
不从该系统结果推断PrimeLoRA收益。

候选只把完成数/忙实例秒的分母改为事件积分。实际reserve、release及退出
running集合同步更新；同runtime group内重叠请求只算忙状态并集。遵循已有
get_runtime_groups分组，draining不再提供可路由容量，但物理GPU仍计费。
不改变完成数、历史窗口端点、EWMA、预测/IEEE公式、cooldown、初始实测
profile、1–3副本、输入生成、远端交付和冻结评价标准。

线上在控制器创建时启用事件观测，之后不允许切换或静默回退。旧独立控制器
API保留明确标记的采样观测以兼容历史调用；生产runner固定使用事件观测。
记录累计忙时间/事件数及实际观测身份，防止将新代码旧结果混用。

先扩展现有IEEE控制测试，保留父版失败，再实现并验证：跨采样短请求、重叠、
同runtime别名、独立runtime、窗口裁切、非法时序、漏事件拒绝、drain、取消
释放失败和重复释放。沿用3/4GiB、swap0、CPU2/3/26/27、CUDA隐藏；不重跑
会启动原生子进程的整个旧ActualControl类，不增加无关资格或短Full回放。
语义通过后保存来源与测试，进入一个新键完整4000请求W2；没有达到目标则
保留负结果，不能换阈值、加常驻副本或重跑同键寻找优值。

保持完整范围：两模型最终同版本W0/W1/W2/G1/G2、独立数值正确性、最终
baseline比较、A1–A5、S1–S13、CE/Pareto/价格、审稿/B1/B2/C1/C2/C3文档、
IEEE图表及FaaSLoRA实际V2集成。目标active，不重开已完成HydraServe适配。

## 实现及CPU验证完成，准备唯一完整W2

实际新增observe_busy_instances，在所有权变化时先累计上一状态，再记录新
runtime group数量。控制历史保留相同端点，分母为两端累计值之差；旧EWMA
和零忙时间不产生新服务率样本的规则不变。七个withdraw入口统一调用同一
同步方法；预留/释放各在真实计数变更后、任何await之前通知。不记录模拟
handoff投影。GPU allocation/release路径与计费未修改。

父版新9项检查保留失败，其中真实reserve/release别名fixture应为2实例秒，
原采样结果为0。修正后65项全部通过（9项新事件检查、原IEEE控制和已有
控制分析），含短请求、重叠、窗口、别名、独立owner、异常时序、快照与事件
状态不一致时拒绝（不声称可识别一对同时遗漏的开始/结束事件）、
withdraw、取消失败和重复释放。AST检查覆盖七个实际withdraw入口；测试中
最初列举的两个方法名按实际源码修正，未改预期计量结果。首次误用本LLM
环境未安装的pytest，在执行测试前退出；保留失败回执，改用既有unittest
包装器，不安装包或改变环境。CPU语义通过不作为GPU性能收益证据。

最终测试峰值932847616字节，high/max/OOM/kill及swap均0，scope已退出。
来源/测试/保护/配置证据放在
paper_results/ieee_tc/p2_backend/d1045_event_busy_time/prelaunch/。
配置20261103_d1045_3b_event_busy_time_w2.yaml只换三个独占输出/缓存路径，
其余逐字段等同D1043，不新增调参项。复用D1000入口、D747 fresh descriptor、
32 CPU LoRA槽、c8/4096 token、1–3副本、真实远端已发布缓存、同4000请求
和500逻辑adapter、60秒通知，关闭详细profiler与prefix诊断。

运行身份：RUN_ID=d1045_20261103；RUN_TAG=d1045_3b_event_busy_time_w2_full。
不重新生成权重/负载、不改远端、不启动额外短Full。完整评价仍需至少3800
联合时延达标，G2预算12083.063870 GPU-s，另有独立数值正确性；实际结果
待测。原D1033/D1043数据和负结果全部保留，不将本轮改动挂接到旧结果。

提交首次检查仅在父版unittest失败输出第7行发现尾部空格，提交尚未发生。
原始证据不作格式化；仅允许这一确切日志告警，其他diff/凭据/来源和保护
检查继续。失败回执保留于raw/backup_attempt1.json，不重跑测试或改已封存SHA。
