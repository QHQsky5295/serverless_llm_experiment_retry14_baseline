# D668：3B当前HOST清点与核心等待的单次定位回放

2026-10-18。承接D667，3B W0优化主线。此前完整D665的联合延迟达成88.75%，
尚未达到冻结要求；这次是详细观测，不是新性能候选或G1/G2结果。

## 唯一问题和停止边界

当前完整HOST物理图清点究竟在原生核心线程占用了多少时间，分别发生于哪些
控制操作？将函数内部wall/thread-CPU与其外部排队分开，避免把pending到
acquired一整段等待都归因于清点。D667已完整记录历史失败候选、官方vLLM/
PyTorch源码和仪器设计，本轮再次核对原始源码，不引入新的调度策略。

只做一次原W0前1000条诊断，不按SLO胜负重跑，不从时延中扣除观察器开销，
不以prefix成本外推4000请求。若HOST清点不是主要开销，转向已测其他段；
若是，再验证保持物理状态正确性的新设计，不能用epoch/TTL缓存跳过检查。

## 冻结执行条件

- 后端、profile、8 slots、32 HOST adapters、资源包络、准备窗口、generation、
  arrival、admission和scaling全部来自D665；无公式或算法变化。
- 原3B W0 trace SHA为`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`。
  仅原序前1000条，不重新生成负载或工件。视图SHA写入preparation/qualification。
- YAML仅三项输出/cache路径改变，total_requests保持4000。runner通过既有
  diagnostic_prefix_v1=1000选择原序前缀，保持明确的诊断身份。
- 沿用D78/D80真实远端发布缓存；请求中不打包，真实传输及竞争仍计入观察。
  不修改远端服务配置，不启动新工件发布，不清整机page cache。
- `host_spans_v1`加`fresh_inventory_v1`，不启用逐tensor、Python栈或V2 step
  components额外仪器。每原生core上限200000 spans/64MiB，检查truncation。
- 服务72/80GiB、swap2GiB；外置3/4GiB、swap0和既定CPU分区；一项重型
  作业。推理机150/100GiB及内存保护不变；远端独立门槛不变。

## 复用与入口校验

重用D665配置组装与profile loader检查、D663生产事务资格655次既有CPU
执行证据，以及D667的259项诊断/回归检查；按源SHA复核，不重复跑大套件。
本轮只重查wrapper静态合同、bash/AST、配置精确差异和当前安全/历史保护。
原有诊断入口新增一个注册身份，去掉该新增分支后与旧脚本逐字一致；observer
wrapper在非D668分支明确关闭新开关，防止环境继承意外启用。

长任务用既有tmux/受限launcher。启动前推送检查点，确认远端、源结果、
工件和profile身份；开始后不构建/压缩/清理/改代码或第二次启动。
观察超时只查看同一会话，不视为运行失败或重启依据。

首次CPU入口核验尝试将YAML population改为1000，组装函数明确只支持已经
测量的100/4000合同，因此拒绝，未启动GPU或远端。保留该配置快照、原核验
脚本/日志与关闭回执（peak610328576B、事件/swap0）。修正仅让YAML保持
原4000，像D351一样由已存在的外部回放合同选择1000前缀；不改组装函数、
不加兜底或放宽生产资格。第二次核验另写qualification2及独立资源回执。

## 收尾与证据

先确认真实GPU释放、精确服务/外置资源域递归空并关闭，再停止本轮确认为自己
启动的远端服务。保留失败/partial、原生token/绑定/时间线、资源事件和完整
仪器日志；所有分析放新目录，不覆盖prelaunch清单和旧结果。

分析复用D351的core span方法，新增两个whole-function边界的聚合：
scope、缓存对象/分配数量、操作分类、wall/thread CPU、均值/P95、父子包含、
短decode窗口内交叠。嵌套span不可重复加和，CPU时间不冒称GPU kernel时间。
是否完整覆盖、是否截断和观察器开销单列；不足以归因时如实保留未知。
保持n_correct未知/不参加排名的边界，不开额外全HF oracle支线。

## 变更清单

| 文件 | 本轮作用 |
|---|---|
| `configs/ieee_tc/20261018_d668_3b_full_w0_prefix1.yaml` | 原配置的唯一诊断路径和前缀 |
| `scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh` | 注册D668、选择原3B工件/trace和已有限制 |
| `scripts/ieee_tc/profile_d328_controller_python.sh` | 仅D668启用D667整函数计时 |
| raw `d668_20261018/`下辅助文件 | 复用资格、备份、启动及精确关闭流程 |

`system-profile`用于控制观测范围，`run-experiment`用于复用执行入口；不改
生产gpu_monitor/residency_manager或新建实验框架。图表继续按用户要求延期。

3B W1/W2/其他系统/dLoRA、同版本7B、A1–A5/S1–S13、成本/CE、审稿文档和
最终图表完整保留；原k3s及已完成HydraServe适配不动。
