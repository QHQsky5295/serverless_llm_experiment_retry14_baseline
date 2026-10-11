# D451：ElasticLocality 全工件池功能覆盖

2026-10-12，接续 D450 已完成的真实远程100请求验证。此项回答能否执行全部
500个既有逻辑 adapter，而非判断完整 W0、G1/G2 或进行基线参数搜索。

## 已有证据与唯一必要适配

D450的100/100原生输出数量与来源绑定通过，但联合SLO只有17/100；该结果
保留，不因效果不好重跑或调整此对照。原完整W0仅出现132个逻辑ID，不能代替
全池功能覆盖。D433/D443已有的 adapter_coverage_v1 将源trace前500条请求的
adapter映射为冻结池排序ID，arrival/prompt/target均不变。

原HTTP执行器已支持此视图，Unix回放和主执行器只支持普通前缀。D451将相同
纯索引规则接入已有 FrozenReplayPlan、受控启动入口和输入加载器；不写新trace，
不复制权重。直接与未修改的 baseline load_http_replay_plan 逐条比对派生行。
新增视图显式标 ADAPTER_COVERAGE_V1，保留原始/派生行SHA与pool SHA，拒绝
formal运行、非500数量、速率变换、缺池/池SHA改变。发布器、服务和GPU生命周期
必须消费同一视图。普通W0/W1、控制策略和IEEE公式保持不变。

冻结视图SHA：286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d；
500请求、500个逻辑ID、68,925目标tokens。逻辑ID覆盖不等于500种独立训练权重，
不把来源绑定当作独立数值证明。

官方 [vLLM benchmark说明](https://docs.vllm.ai/en/latest/cli/bench/serve/)
区分请求速率与并发限制；本项目仍复用已冻结的open-loop实际到达索引，不用
重新采样的同均值速率替换它。D451是输入兼容适配，不是新的性能优化方案。

## 冻结执行条件

- 完全沿用D450的ElasticLocality：slack1、普通GPU局部性/负载回退、普通制品LRU；
  无handoff/hierarchy/主动admission，保留原生按需加载和物理安全。
- vLLM0.30.0、7B FP16/TP1、并发/slot14、context1024、四GPU上限、同autoscaler，
  非路径配置逐字段相等。无需重新执行D450，未动主结果配置。
- 共同60秒通知；fixed_length_greedy_v1、原生token计时、1800秒资格保护，
  D394共同SLO。未ready排队不扣除、不后移业务起点。
- D78/D80只读gzip交付复用；不重新发布、不请求内打包。读取、传输、竞争和
  本地物化仍计入时间；测量期间不修改或清理远端。
- 一次重型任务。服务72/80GiB、swap2GiB；辅助3/4GiB、零swap；CPU分离、
  外部watchdog和实际GPU退出证明不变。磁盘启动150GiB/运行100GiB保持。

## 记录、验收与下一步

raw：原项目results/ieee_tc/elastic_locality_qualification/d451_20261012；
新配置：configs/ieee_tc/20261012_d451_7b_elastic_locality_500.yaml。
复用D450管理/收尾与analyzer，只改campaign身份和明确覆盖视图核验。
500/500 offered/submitted/terminal、完整原生数量、adapter/source/lease绑定，
真实远端UUID/SHA/字节和完整物理GPU释放均需验收。

CPU失败保留：首次组合测试中四项依赖pidfd而环境缺接口，一项500消息夹具
误用50ms到齐断言；系统Python又缺五项离线测试依赖的numpy，native环境也无
pidfd。最终按实际职责分组：四项PID-handle测试使用生产watchdog的系统Python，
其余测试使用既有LLM环境；不安装包、不弱化安全检查、不修改生产等待策略。
500消息夹具改为有超时的实际接收完成条件。
一次stdin测试选择器不能用于multiprocessing spawn，已按准确InvocationID停止，
退出143；改为带main guard的文件入口。生产代码没有为测试增加降级路径。

最终268项控制/回放/输入测试通过54.029s，4项PID-handle安全测试通过0.005s。
此前失败记录不删除；未增加GPU性能重复。

输入适配和CPU核验已通过：新旧500行逐条相同，视图SHA一致；非路径配置与
D450相同。主runner仅输入加载定义改变，另596个定义AST不变。147个历史保护项
和三个用户修改文件SHA不变。六个CPU资源域均在核对身份及递归实际空进程树后
停止，high/max/OOM/swap均0。以下实际GPU覆盖是在该备份后运行的一次覆盖，
接续原1000兼容→一次完整4000W0。其余已封存五系统W0不重跑；W1/W2/G2、
必要PrimeLoRA优化、3B、全部消融/机制/敏感性仍在主线，整个目标未完成。

07:11:38实际500-ID运行已启动，tmux tc-d451-locality500。代码检查点
6cb9bf6e3d29984de24967f9d9f8c36b8a115501已推送且远端SHA一致。

## 实际覆盖结果与解释

07:27运行终态：500/500请求成功，68,925原生输出tokens，全部目标数量、
adapter/source/lease绑定通过。与D433保留的500条请求合同独立核对：canonical
prompt SHA、原生prompt token-ID SHA、目标长度、派生源行与覆盖索引全部一致。
这不要求跨后端输出token逐字相同，也不把控制/来源绑定冒称独立数值证明。
原始n_correct=null保留。新产物为
`paper_results/ieee_tc/elastic_locality/d451_actual_coverage500.json`。

| 指标 | 本次观测 |
|---|---:|
| 共同联合SLO | 186/500（37.2%） |
| 平均TTFT | 58,534.608611 ms |
| Type-1 P95 TTFT | 168,610.997153 ms |
| 平均dispatch/admission wait | 16,565.516369 ms |
| 平均service TTFT | 41,969.092242 ms |
| P95 TPOT | 36.373022 ms |
| 实际物理GPU生命周期 | 2,579.828998 GPU-s |
| 真实HTTP获取 | 500次，420,740,410线上bytes |
| activation | initial 1，natural_scaleout 3 |

本视图每个逻辑adapter只出现一次，500条dispatch前均为remote，未触发GPU
局部性命中；693次路由尝试并非693个请求。主动准备为0、文件LRU替换为0。
它验证完整工件池功能，不隔离locality、缓存驱逐或hierarchy的性能贡献。
共同阈值下明显不满足95%，不因该结果调整对照，也不把它当完整W0排名。
初始activation43.483525秒，与D450不同视图/运行的启动差异不作因果解释。

两端transfer UUID、内容SHA与字节逐条相符，按请求打包次数为0。再次收到用户
一次性压缩交付缓存授权后，核查并复用D78/D80，不重新发布、不新增解压池。
真实读取、传输、竞争、本地物化保留在请求时间中，没有事后扣减span。

931次资源采样：峰值28,073,377,792 bytes，high/max/OOM/swap均0，无安全中止。
服务/watchdog退出0，GPU实际释放和服务资源域移除后，才停止准确归属的远端
3B/7B/监控服务；端口关闭，日志传回SHA匹配。启动/回放辅助域及收尾9–14域
均核对身份、实际递归空进程树后停止。当前没有遗留本轮作业。
147个历史保护项、冻结协议和用户修改SHA不变；私有NVMe工作目录已由正常
运行收尾释放为空，磁盘约151.19GiB可用，本轮无需额外删除。

离线分析失败也保留：首次解释器路径写错，exit127且分析未运行；额外输入
见证首次引用不存在的source_item_sha256属性，exit1；另建v2脚本使用已有
source_sha256后500条通过。没有改变实验数据、重跑GPU或加入运行时兜底。
新一轮必须重新通过资源门槛，下一任务仍为原1000兼容，再一次完整4000W0。
