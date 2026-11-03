# D1036：预算推导 CPU adapter 容量的完整 W2 验证

2026-11-03。当前目标active，主线3B PrimeLoRA；**已准备，尚未启动**。
本文件是最新交接，不改用户dirty的EXECUTION_STATUS.md。D1035画像已完整
结束并导出，不得重复，也不额外添加短Full。HydraServe隔离适配和D810基线
结果已存在，不接管共享k3s，不重开基础适配。

## 唯一问题与父点

D1033为开发父点：4000原生完成、3772联合达标（94.30%），物理GPU时间
8155.283735964913。它未达到3800，也未因GPU时间小幅减少被接受为改进。
D1034在完整日志中发现CPU容量32与GPU槽32相同、CPU缓存3868次达到上限、
实际原生CPU-only来源0次；现有2GiB子预算未充分利用。公式倒推得到110条，
保留两份工作区与观测非注册占用，仍执行实际字节准入；不作容量网格扫描。
历史、一级来源、推导和限制详见D1034。重复加载可能减少，状态检查可能变贵，
最终以完整结果判断，不预设能补齐28条联合违约。

## 冻结配置与输入

`configs/ieee_tc/20261103_d1036_3b_native_host_capacity_w2.yaml`
SHA f5435ebc10ca2e77cd14c3cb193a0e0afc0d6ce7c6260876a041d132e9cc746e。
相对D1033仅14处结构变化：CPU上限32→110、两个完成长度身份、两组新实测
画像路径/SHA/上下文及三个独占输出/缓存路径。严格运行配置装配后160服务类、
96准备类均能载入；保持启动政策、GPU32槽、并发8、token预算4096、1–3
副本、16GiB HOST、2GiB原生张量预算及IEEE九公式。实际物理分配日志新增
startup_memory仅为观测字段，不更改决策或计费，也不补写旧结果。

D1035画像manifest：c33175472fcc6739193808cdc7ffdbda4930693e6b297631efb28f3af9e409a7。
service：b1fca5b24c7ef91087851a744fa5b3cae038194df016a3a080ef0e20c184ebb1；
preparation：9849e90fd1b1a160baec3ee2e9be582d3b0aecb499f65787524e7f14083a5348。
24工件画像不能证明110条填满，Full须核对CPU-only复用、加载次数和相关
准备/控制开销，不把dispatch HOST误当原生CPU命中。

既有seed42开发trace，4000请求/500逻辑adapter；W2 rotation100、相同arrival/
prompt/target和fixed_length_greedy_v1，提前60秒部署，准备及释放持卡全计。
trace SHA 4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d；
map SHA 28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23。
真实远端使用已发布只读对象，实际读取/传输/竞争/本地物化计入，不打包或
扣除真实传输。远端快照不冒称连续监控，保留共同限制。

## 执行与判定

继续复用 `scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh`。
外部变量：FAASLORA_D1000_RUN_ID=d1036_20261103；
FAASLORA_D1000_RUN_TAG=d1036_3b_native_host_capacity_w2_full；
FAASLORA_D1000_CONFIG=上面配置路径。runner已固定W2，历史文件名不改。
显式取消三个TC详细timing、D1000_NATIVE_PROFILE、D1000_PREFIX_COUNT、
TC_DIAGNOSTIC_PREFIX_COUNT及VLLM_ENABLE_STARTUP_PLAN，不注入额外画像。
只启动一次完整键，run目录独占创建；不复制缓存、不强制命中或重测冷/热版本。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
冻结V1/单轮V2及D648阈值不变：G1全部正确且至少3800联合达标后比GPU-s；
W2 G2还要求U≤12083.063870并比较P95 TTFT和观测联合差值。不用控制台
5000ms作最终判定，不作CI。独立数值正确性仍未知，不补造n_correct。

启动前PLAN、实际资源和既有147项保护检查；服务72/80GiB、swap2，辅助
3/4GiB、swap0，原CPU集合与独立watchdog不变。启动102GiB内存/150GiB
磁盘与预计增长，运行16GiB/100GiB底线；不得在测量时构建、压缩、推送或
远端变更。失败/退化保留，不扫描其他容量或重复求胜；先关闭本候选再决定
下一瓶颈。仅保存数据，达标前不为展示画图。

完整剩余范围：两模型同最终版本W0/W1/W2/G1/G2、独立数值正确性、外部
基线最终比较、A1–A5与S1–S13、CE/Pareto、B1/B2/C1/C2/C3及两轮审稿文档、
IEEE图表和最终V2集成。检查点备份不等于历史分歧FaaSLoRA V2已最终集成。
