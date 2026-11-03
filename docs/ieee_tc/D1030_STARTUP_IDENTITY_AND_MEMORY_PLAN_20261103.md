# D1030：启动缓存身份核查与下一项有界优化

2026-11-03。状态：离线身份核查和 CPU 测试完成；本轮没有 GPU 回放，
没有修改生产策略、已安装后端、远端服务或论文。完整目标 active。
本文承接 D1029；不改用户 dirty 的 EXECUTION_STATUS.md。

## 主线和现有结果

已完整重读目标、PLAN、冻结 V1/单轮 V2，并核查 D1023、D1028、D1029 和
相关代码历史。采用 analyze-results、serving-llms-vllm 的源码/证据分析流程；
技能中的通用时延或吞吐建议不替代本项目冻结评价。备份使用 github-sync。

- PLAN SHA：d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
- V1 SHA：5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22。
- V2 SHA：193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
- D1029 全部 4000 原生合同完成，联合 SLO 3774/4000（94.35%），
  8167.260452 GPU-s，P95 TTFT 1890.549688 ms。差 26 条才到 3800；
  资源低于预算不是替代性通过条件。
- D1027 的 3784/4000、8316.260704 GPU-s 以及 D1022 的 3770/4000、
  8062.415716 GPU-s 全部保留。D1029 不重跑暖缓存，不删启动阶段。
- n_correct 仍未知，ranking_eligible=false；原生数量/binding 合格不冒充
  独立数值正确性。D810 基线观测保留；HydraServe 隔离适配已经完成，
  不接管共享 k3s 或重开已结束的基础适配。

## 1. 重复编译的实际证据，不凭加载器差异推断

复查两个已关闭 worker 的 payload、日志及后端自身保存的
cache_key_factors.json，而不是只比较父级 YAML。

| 项目 | D1028 画像 | D1029 Full 初始副本 |
|---|---|---|
| 原生 worker PID | 3665170 | 3730298 |
| AOT key 前缀 | 9bc8f69a03d6 | 68ad0d2bf035 |
| 编译时间 | 30.86 s | 31.08 s |
| loader | auto / prefetch | fastsafetensors_affinity_preserving_v1 |
| 原生 config_hash | 56a30633c7 | 56a30633c7 |
| 原生 compiler_hash | 67f58750f5 | 8e33489fd9 |
| 已记录环境项 | 237，逐项相同 | 237，逐项相同 |

两份 code_hash 同为
fe4ac84c7d0766cdfcc1f902d784260e22693d5d5149a0a3465ea162bc7d801d。
实际 worker payload 中的 admission profile、capture policy、slot、token 轴
一致，只有 loader 等启动路径记录不同。相等短哈希不等于全部执行合同相同。

安装源码显示：LoadConfig.compute_hash 不包含 loader；ParallelConfig 排除
worker extension；SchedulerConfig 的相关图形状因子是 token/sequence 上限。
因此，目前证据不支持“换成相同 loader/worker extension 即可命中缓存”。

官方路径把环境、配置与编译器因子组合进缓存身份。编译器因子包含系统信息、
PyTorch 源码身份、Inductor/Functorch 配置；当前历史文件只存组合哈希，
**尚未确定其中哪一个子因子导致差异**。不删除哈希因子、强制缓存命中、
拷贝编译产物或编造根因。

两次独立、CUDA 不可见的 CPU 导入探针分别不导入/导入 bulk loader，
得到相同编译器组合哈希 7e70d33753，四个组件哈希逐项一致；CUDA 未初始化。
这只排除“导入 loader 本身必然改变该 CPU 状态”的解释，不模拟 GPU 启动，
不证明两次真实编译器状态相同。探针的 env_factor_hash 是诊断用 str 哈希，
不是原生规范化环境哈希，不据它覆盖上面的真实原生记录。

**有界结论：**缓存身份核查到此保存；两次最小 CPU 探针后不继续无目标地扫描
编译器配置、不重复 GPU 来找热缓存优值。若后续必要启动验证产生新 worker，
可在相同启动边界额外保存四个编译器因子的身份，补齐具体来源。

## 2. 返回实际扩容瓶颈：复用显存画像，而非绕过编译校验

D1029 后续三个 worker 的编译已为 0.27/0.30/0.29 s，仍需
27.59/28.81/27.57 s 初始化；这说明自然扩容仍有与冷编译不同的重复工作。
D1028/D1029 已证实内存估计捕获与实际服务捕获分别发生。
不把这两次图捕获对象直接复用：临时图被释放，真实 KV 建立后地址不同。

联网核查发现所安装 vLLM 已提供 VLLM_ENABLE_STARTUP_PLAN。它在完全画像
后保存显存计划，后续匹配并且当前空闲显存不低于记录值时跳过重复内存估计；
仍执行 profile_run 所需编译及真正服务图捕获。当前默认关闭，startup_plan
缓存目录不存在。本轮没有创建计划或打开开关。

需要区别两个事实：

1. 原生计划采用 compute_hash，但这个**图哈希**直接忽略 gpu_memory_utilization、
   kv_cache_memory_bytes 和 loader。当前 PrimeLoRA 的额外 model_backend_id
   间接绑定部分模型配置，不能把这个间接关系当作所有内存依赖均已覆盖。
2. 原生保存的是按实际图用量与原生安全余量重新推导的 KV 建议值，而非必然
   等于此次已分配的 KV。D1029 初始日志建议 7654083544 B，当前 KV 记录
   7.54 GiB；直接启用可能改变 KV 容量，不能声称只改变启动时延。

因此，不直接把历史建议数字写成新的手工 KV 常量，也不从两位小数的空闲显存
日志伪造精确缓存记录。这不是要求增加整个新性能矩阵，而是避免错误复用。

### 下一唯一候选与执行边界

可证伪假设：**在同一实际模型、加载器、显存预算、执行图与运行环境下复用
已经完成的显存画像，可减少自然扩容的重复估计工作，改善完整 W2 的联合 SLO
或资源占用，而不改变 IEEE 公式、路由、准备目标和 admission 语义。**

下一任务沿现有 worker_cls/原生 startup-plan 入口实现明确 opt-in 合同：

- 内存计划身份须显式覆盖完整内存相关配置、loader、执行图策略、后端/工具链
  与实际硬件；不削弱原生编译器缓存校验。采用固定配置或实测身份，不按 W2
  标签或已知差 26 条设计阈值。
- 仅用真实启动后测得的记录；缓存缺失/身份不同/空闲显存不足时，按原生正常
  画像路径测量，不编造容量、不隐瞒缓存 miss，也不重试另一组性能参数。
- 复用现有最小原生启动/生命周期入口验证缓存产生和命中、KV 实际容量、
  原生请求合同及释放；只验证此次改变的路径，不重新跑 500-adapter 基础资格。
- 若 KV 容量改变，登记新身份并用现有 423 请求画像入口重测相应 service/
  preparation；不把旧 timings 改标签。若相同已有画像已覆盖 changed path，
  不额外插入短 Full。
- 冻结后只跑一次新执行键的完整 4000 请求 W2。记录全部 worker 的初始/自然
  扩容、缓存冷/热及阶段耗时，按相同 D648 SLO 与物理 GPU-s 评价。
- 缓存身份缺口无法在现有接口内清楚解决时，停止此候选，回到现有稳态失约
  证据；不无限围绕缓存兼容性扩建框架。最多两轮最小验证，保留反例。

该候选是下一步，不是本轮已经实现或验证的加速，也不保证能恢复 26 个请求。
它与 D724 token 轴、D1028 active-LoRA 轴的形状裁剪不同，不再裁形状。
D368 的 startup-plan 文档是历史请求预处理屏障，非此后端显存画像缓存。

## 3. 源码依据、产物和保护

2026-11-03 原始来源：
[vLLM 编译缓存因子](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/compilation/caching.py)、
[vLLM 配置哈希](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/vllm.py)、
[原生启动计划](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/startup_plan.py)、
[worker 实际调用位置](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu_worker.py)。
实际安装身份另存 manifest，不假定在线标签与本机逐字相同。

复用 audit_d703_3b_full.py 增加只读 compare-native-compile-factors 模式；
不载入 pickle、不执行编译缓存、不改原生运行。现有/新增测试共 6 passed，
6 项既有配置/弃用警告。输出为
paper_results/ieee_tc/p2_backend/d1030_startup_identity/compile_factors.json。
raw 位于 results/ieee_tc/p2_backend_qualification/d1030_20261103/，包括两次
CPU 探针、脚本与 identity_sources.tar.gz。归档仅含两份因子 JSON 和两个
worker payload，SHA 9677e84f650d6fecf1b89bbc03e2b3611d3fab6a15db6abd7d2bea6c5ec7690a；
没有归档或复制模型/工件/负载。

CPU 检查均限制在 2/3/26/27、3/4 GiB、swap0、CUDA 不可见，串行完成。
探针/测试/分析进程均退出 0；未完整保留退出前峰值，不虚构全量资源零事件。
11:54 可用磁盘 161185902592 B（约 150.12 GiB），无 GPU compute 进程。
没有删除/压缩唯一数据、没有远端操作或共享服务变更；两个用户 dirty SHA 不变。
本次按既有 github-sync 检查保存可回退节点，不 force push 分歧 V2 分支。

完整待办仍保留：两模型同版本 W0/W1/W2/G1/G2、独立数值正确性、
与现有基线的最终对比、A1–A5、S1–S13、CE/Pareto、两轮审稿与 B1/B2/C1/C2/C3
文档、达标后的图表及最终 V2 集成。本轮没有把 CPU 核查写成新性能实验。
