# D424：7B W0 同合同比较与 Resident 参考冻结

2026-10-11。本次仅重用封存日志做离线分析，没有新 GPU 实验、参数选择、远端
操作或历史结果覆盖。按冻结 V1 和单轮 V2，仅报告已知 W0 的单轮观测。

## 结论与边界

三系统的 4,000 条请求、adapter、原始到达、目标输出、canonical prompt 及
原生 prompt token 身份全部匹配；均生成 458,224 个原生 token，实际出现
132 个逻辑 adapter，完整可访问池为相同 500 ID。共同 SLO 均超过 95%。

| 系统 / 封存运行 | 联合 SLO | 物理 GPU-s/request | 平均 TTFT ms | P95 TTFT ms | P95 TPOT ms |
|---|---:|---:|---:|---:|---:|
| PrimeLoRA D371/D397 |97.700%|3.620631|1150.798|2837.896|51.741|
| ServerlessLLM D414 |98.125%|4.022441|973.687|1052.246|32.929|
| vLLM D423，四卡 Resident |99.475%|4.028844|358.019|1200.507|27.177|

在这三个已测、共同 SLO 达标的工作点中，PrimeLoRA 的物理 GPU 时间最低：
相对 ServerlessLLM 减少 **9.9892%**，相对 Resident-vLLM 减少 **10.1323%**。
但联合 SLO 分别低 **0.425 / 1.775 个百分点**；P95 TTFT 分别高
**169.699% / 136.391%**。资源优势与延迟代价必须同时报告，不能写成全面优胜。

这不是完整 M1 的跨 W0/W1/W2 `resource_opt` 选择，更不是相同预算下的 G2。
Resident-vLLM 是预登记四卡参考，不是经过 1/2/3/4 卡资源最优选择的 vLLM。
ServerlessLLM 的原生四卡 store pool 同样是当前已验证部署，不代表其所有配置。
因此不得写成“PrimeLoRA 已优于优化后的全部基线”。

## 比较合同核查

复用现有 `check_d299_full_context.py`，新增显式 `--matched-w0-output` 入口。
输出目录必须不存在。既有严格审计通过 SHA 绑定后，再逐请求关联原始记录；
大结果使用流式读取，CPU 2/3/26/27、内存上限 4 GiB、零 swap，GPU 隐藏。

1. 三份封存审计及 D414/D423 证据清单 SHA 核对；引用的请求、物理账本、
   allocation/release、终态账本和源 trace 再核验。已回收的下载缓存不重新下载。
2. 4,000/4,000 请求集合、source-item SHA、adapter、target、canonical prompt
   SHA、native prompt-token SHA/数量、内容 SHA 全匹配。到达只容许单调时钟
   相减产生的 1 微秒偏差，不允许负载重排/缩放。
3. PrimeLoRA 历史 subset ID 集合等于冻结内容索引 500 ID 集合。源 trace
   路径不同但字节 SHA 和完整 view SHA 相同。
4. 三系统采用业务前 60 秒通知、独立 open-loop、相同 CPU/high/max/swap
   包络、原生首末 token 计时和 D394 两个长度组阈值。重算与封存值一致。
5. 物理 GPU 区间有真实释放，按每卡不重叠区间积分；四个互斥窗口之和等于
   总量，未用逻辑副本数或 idle 折扣替代。
6. 本轮冷 LoRA 工作缓存、同一已发布 gzip 交付协议，无请求中打包、无预启动
   GPU。后端编译磁盘缓存分别保留并披露，OS page cache 未全局清空。

匹配合同摘要 SHA：
`384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399`。
source refs、原始数值与 source-order 500 请求分段计数见
`paper_results/ieee_tc/main_comparison/d424_7b_w0/comparison.json`。
分段是描述性诊断，不是八次独立运行，也不取代包含启动的完整结果。

原始 `n_correct=null`、数值资格未知/false、`formal_run=false` 不改写。
“合同有效”要求已封存的 source/content/native/worker/physical 绑定，不是
仅 HTTP 200 或生成数量，仍不等于全部请求的独立内核算术证明。既有
D385/D397/D407/D409/D423 有界独立证据按原范围保留。

## 输出变化、计时边界与机制解释

输出 digest 比较：PrimeLoRA–ServerlessLLM 有 203 条不同，PrimeLoRA–vLLM
有 231 条不同，ServerlessLLM–vLLM 有 137 条不同。不同请求 ID 完整保存；
不假定均为舍入误差、不删除差异请求、不用逐 token 一致作额外准入门槛。
PrimeLoRA/vLLM 同为 0.30.0，但批处理、调度和 native 配置不同；要解释数值
差异的原因，必须另外关联已有独立 witness，不能由摘要推断。

PrimeLoRA 的 E2E 完成点是 controller completion，基线是 external HTTP
completion。原平均 E2E 分别为 4825.670 / 4000.734 / 3182.814 ms，保留定义。
另给共同“计划到达至末原生 token”：4487.361 / 3977.746 / 3176.705 ms；
只作补充，不改名为 E2E，不抹去通知尾部成本。

为了避免不同 dispatch 位置误导，以下统一按 **native engine entry** 分解：

| 平均阶段 ms | PrimeLoRA | ServerlessLLM | vLLM |
|---|---:|---:|---:|
| 计划到达→提交 |2.558|2.848|2.655|
| 提交→原生引擎入口 |834.060|559.782|18.754|
| 原生引擎入口→首 token |314.180|411.056|336.610|
| 首→末 token |3336.563|3004.060|2818.686|
| 末 token→各自完成通知 |338.309|22.988|6.109|

前三阶段均值之和等于平均 TTFT；不能相加阶段 P95。PrimeLoRA 的前端等待/
准备与完成尾部值得后续定位，但这张表不能确定排队、控制计算、跨进程传输
各占多少，更不能直接归因于九个公式。先完成剩余 7B 基线合同，避免局部优化失焦。

三系统本轮均为 **132 次 HTTP / 131,480,060 wire bytes**，所以本 W0 没有
“减少远端下载”的观测优势；资源节约不能据此归因。PrimeLoRA 131/132 次
请求级远端阶段关联，缺一条阶段归属，但总远端计数已知。大部分权重为零或
重复的限制仍在，500 逻辑 ID 不等于 500 独立训练权重。

## 7B W0 Resident 数值参考

D423 运行前登记为参考候选，是此路径首次完整计量合格 W0，未从重复中择优。
单轮 V2 下冻结：

- `U_ref = 16115.377162149176 GPU-s`。
- `Q_ref = 1200.5068596918136 ms`，Type-1 P95 TTFT。
- `B_W0 = 0.75 U_ref = 12086.532871611882 GPU-s`。

数值文件 `resident_reference.json`，SHA
`737c84804023c9f98677fb5f19c8240b4b05cabdb52f6ecdbbfd4527964c3269`。
仅用于 7B/W0，不修改 D394 SLO 文件的历史 resource-reference-open 状态。
配套 `reference_provenance.json` 记录 generation、runtime/evaluation SLO、
warm/resource-reference 的哈希及执行来源。执行键由已核验的启动前身份/实际
启动与 worker 配置、编译缓存和执行 commit 事后派生，明确不是新预登记或新运行。
W1/W2 各自参考与 G2 仍开放，不能用 W0 预算代替。PrimeLoRA 当前 W0 占用
超过此 W0 75% 预算，仅是描述性事实，不是对未执行的 W1/W2 作达标判定。

## 验证、交付与下一项

- 四项新增 CPU 测试加原有 27 项：**31 项通过，0.284 秒**；该次测试是工具
  终端记录，未声称存在独立原始测试日志。实际完整离线关联一次通过。
- 前置只读检查曾因环境缺少 ijson、使用错误历史路径失败，没有改输入或生成
  结果；改用已存在环境和实际路径。这些不是性能运行或性能重试。
- `comparison.json` SHA：
  `0a4df226457f09451e447ea0d520789ed6b8fe8b40689361bc4336e9ba8acd97`。
- `metrics.csv` SHA：
  `2ddc86883b9934b8fe31367ef1e844539a8ea42e8beb344309e02dab276ea30e`。
- 不画图、不改后端或论文、不重复远端发布、不生成新数据/权重。
- CSV 为标准库默认 CRLF 行结束；Git 默认 whitespace 检查将 CR 报为行尾
  空白。使用仅本次命令的 `core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol`
  识别合法 CRLF，同时保留其他空白检查，不改结果字节或仓库全局配置。

下一项：按顺序完成 S-LoRA 历史日志/官方实现/原生计量合同核查，复用既有固定
输出与 FIFO 修复，只补缺失证据。不直接运行会预加载整池、等 ready 后回放或
按端口盲杀的旧 wrapper。其后仍有其他 7B 基线、必要比较后优化、W1/W2/G2、
3B、核心消融、机制、敏感性、CE 与文档/图表交付。完整目标继续进行。
