# D642：当前3B分层来源测量完成

2026-10-17。执行版本`a24c655201cfeef1173176a36d3ae8d539e3977a`，
真实远端、已发布gzip、原368case和既有3B池。不是4,000请求Full主性能点。
采用analyze-results区分观察/解释/边界；单轮协议不计算运行间CI，不提前绘图。

## 实际结果

368/368原生请求完成，actual/target/native token数全部相等；360条测量、8条
保留预热。40/40可达服务类及96准备类有覆盖，缺失内容来源组合为0。
五条来源表示各72测量条目；HOST文件与HOST原生张量分开，不虚称五个硬件层。

| 请求前来源 | 平均获取阶段ms | admitted→首token均值ms | 原生TPOT均值ms |
|---|---:|---:|---:|
| 真实远端 | 1512.502 | 1788.983 | 17.175 |
| NVMe文件 | 121.469 | 365.546 | 19.509 |
| HOST文件 | 118.762 | 366.086 | 19.402 |
| HOST原生张量 | 53.700 | 392.775 | 18.988 |
| GPU已就绪 | 0.000 | 164.332 | 19.497 |

获取为实际acquired−admitted，GPU命中保护时二者相等，零值不代表GPU请求无
服务开销。admitted→首token不是从计划业务到达计算的主TTFT。组内有并发和
不同完成长度；这些阶段不是独立相加的CPU/GPU瓶颈剖析，也不等于完整负载收益。
NVMe访问可能命中page cache，未进行整机drop cache，不声称纯物理盘读取。

实测启动93.584810秒，单卡物理生命周期505.402549 GPU-s（包含整个测量的
受控层级设置、初始化和释放），不是主实验GPU-s/request。368次远端交付含
between-wave setup，线上854130750字节，全部双端ID/内容/字节对应，请求内
打包与临时打包文件均0。不能把这些受控fetch数当成W0 miss率。

服务内存峰值5735419904字节，high/max/OOM/OOMkill均0、服务swap为0。
实际GPU清空、服务路径消失；launcher递归空后按原身份关闭。两个远端artifact
及专属monitor已inactive，双端日志按SHA下载。147投稿保护和用户manifest未变。

## 可支持与不支持的结论

- 本机当前3B配置下，远端到可执行状态存在可测准备代价，GPU已就绪可避免该
  阶段；这为分层驻留动机提供来源证据，不自动证明调度/传播/准入的独立贡献。
- NVMe与HOST文件的均值很接近；原生HOST获取较短，但admitted→首token未
  更低。不能宣称每次提升层级都严格改善端到端延迟；后续按实际D/T/O使用测量。
- D641为业务前失败，D642成功并不单独识别失败根因。编译合同及先前留下的
  编译/page cache状态不同；本次93.6秒也小于旧180秒上限，不把“延长超时”
  写成因果性能改善，不把V2 Model Runner无对照地定为问题来源。
- 这是source-only路径，无完整pending-KV admission记录。首次离线工具调用
  admission-enabled exporter因此被拒绝；失败保留，未弱化检查、未补造字段，
  未重跑GPU。随后复用source类覆盖检查和真实原生时钟/token检查完成离线审计。
- 原生完成/adapter来源绑定不冒充独立数值正确性证明，n_correct保持未知；
  不计算正式联合SLO，不把已有3B旧阈值或7B warm阈值套入，不宣称Full达标。

## 紧接的主线

复用本轮完成请求的原生prompt/输出长度，按现有completed-length审计及
measured_admission_initializer建立新3B初始化身份；不能给旧D89时延换标签。
随后以同模型参数的admission-enabled source实际测量/检查必要profile，补
当前共同3B warm和Resident，再完成3B Full/W0–W2和全部baseline/dLoRA。
不返回7B重复HF资格，不恢复无限baseline优化；A1–A5、S1–S13、成本、审稿
文档和最终图表全部保留。后续选择配置前完整读取冻结V1/单轮V2。
