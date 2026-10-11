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
停止，high/max/OOM/swap均0。尚未宣称GPU覆盖完成，备份后只运行一次此覆盖，
再原1000兼容→一次完整4000W0。其余已封存五系统W0不重跑；W1/W2/G2、
必要PrimeLoRA优化、3B、全部消融/机制/敏感性仍在主线，整个目标未完成。
