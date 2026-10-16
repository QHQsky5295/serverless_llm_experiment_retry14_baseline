# D591：补齐 S-LoRA 的完整 7B W2 对照

2026-10-16。主线缺项：S-LoRA高频adapter轮换W2，完整4000请求一次开发观测。
不调整其性能参数，不为低SLO追加搜索；原生S-LoRA调度、round-robin、四个TP1
常驻副本、顺序启动、packed HOST表示、swap和统一显存池保留。
native配置除计量源码view路径/SHA外与D542 W1完全相同。

## 输入、资格与来源

- 沿用所有W2对照的既有view，SHA
  `5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7`；
  原seed42、development41、formal=false。4000请求、497实际ID、458224目标token。
  500逻辑工件池不变，没有生成新权重、prompt、到达或输出长度负载。
- 固定greedy/ignore-EOS，逐请求原生token等于target；从计划到达测TTFT。
  相同60秒准备通知、真实远端D78/D80只读已发布gzip；请求中不打包。
- 原生环境`slora_official_cu118`复用；源接入共同helper包含后续其他系统改动，
  因此刷新计量view身份。相对D542执行时baseline提交`bde2154`的16个S-LoRA
  函数AST一致，生成的原生测量文件SHA与D542一致，未改变S-LoRA机制。
- 当前通过51项接入/生命周期测试、60项分析/输入测试；W2字段和完整source
  衍生合同通过，不重复GPU smoke或性能资格。资格资源域实际关闭，峰值
  622231552字节，high/max/OOM/kill和swap均0。
- D433既有500-ID原生绑定/计时/数量全部通过，68925原生token，audit SHA
  `7cc9f8e220e0b5cf761cd5c416e6d3126d22f42cb77f46494af10341feedf3f7`；
  D434原1000兼容以及D435/W0、D542/W1继续复用，不重复这些实验。
- D591 native配置SHA
  `53d249d2ec23d3d8ede25a3aa01c183288514efc40b0ff88f58e4266d4a313ba`；
  新计量view manifest SHA
  `6f377417a35860757cb0627d7b71d008278b04afd21fea4d0f90db4e34c20fba`。

## 执行与判定

先提交备份，再核查实际空闲GPU、内存≥102GiB、磁盘≥max(150GiB,
100GiB+1.5×20GiB)及祖先资源边界。服务72/80GiB、swap2GiB、指定40逻辑CPU；
回放/监控3/4GiB、swap0和独立CPU。独立watchdog不与被测服务共存于同一限制域。
远端仅管理已核实归属的两个制品服务和本轮监控，不访问或改动共享k3s。
本轮仍保存既有Triton磁盘缓存身份，不把它冒称完全冷缓存。

使用既有native-http-service、外置open-loop回放与严格slora_resident_rr_v1、
full4000_w2分析器。观察结束后：实际GPU/服务释放→确切远端关闭→保存双端
证据→严格合同/原生计时/物理GPU积分与共同D394 SLO→输入join→结果检查点。
不在运行期间测试、构建、压缩清理、推送或调整远端；失败完整保留。

判优仍按冻结V1/单轮V2。独立LoRA数值正确性未知、498零权重/2非零和4种独立
权重SHA限制保留；4000原生绑定成功不自动证明独立数值正确。
不承诺本点达到95%，也不因低SLO排除这条已发表基线；不得将一个原生冻结
工作点写成S-LoRA的全局最优。单轮无CI、无补跑筛赢家。

原始：`results/ieee_tc/slora_qualification/d591_20261016`。
汇总：`paper_results/ieee_tc/main_comparison/d591_slora_w2/`。
启动前尚无性能结果。D590安全清理已备份`a76ed14159edff7ebed7db29804dd61ab503f2ab`；
baseline代码仍`80e08b4c6e7c2874b1b237500a00811ec06cd8c5`，用户修改不stage。

本点完成后汇总全部可用7B对照，针对PrimeLoRA在W2/G2的实际差距提出一个
有历史日志、源码和原始文献依据的可证伪优化假设。3B、dLoRA限定比较、A1–A5、
S1–S13、成本/CE、审稿文档和最终图表仍完整保留。依最新规则暂不绘图。
