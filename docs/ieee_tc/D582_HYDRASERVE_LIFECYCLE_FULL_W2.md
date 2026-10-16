# D582：HydraServe 同配置完整 W2 churn 回放

2026-10-16。准备完成，尚未启动；本文不是性能结果。

## 要回答的问题

在固定 D573/D574/D576/D579 已资格化的原生配置下，观察更频繁的 adapter
轮换对联合 SLO、实际 GPU 生命周期和尾延迟的影响。保留完整 W0/W1 的
不达标结果，不根据其胜负改变 HydraServe 调度或评价标准。本轮是尚未执行
的 W2，不重复已有实验，不以新的运行编号冒充独立测试负载。

## 冻结合同

- Llama-2-7B，已有500逻辑adapter、4000请求、源seed42、开发身份41。
  复用 D553 的 rotation100 映射，arrival/prompt/token 工作量不变；逐项与
  D557 的共同 W2 视图核对4000个request ID、时间、源内容及adapter。
  实际497个adapter，目标458224原生输出token，不重新生成工件或trace。
- view SHA `5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7`；
  map SHA `48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`。
- 原生batch8/use_cache0/slow_expr0、idle retirement30秒；共同通知60秒、
  请求保护1800秒、fixed_length_greedy_v1。16个原生适配文件与D574相同；
  共享执行源码与D576/D579相同，不搜索新配置。D553既有386项测试复用，
  此次新增的是实际输入对应和来源核查，不冒称再次运行386项测试。
- BASE `80e08b4c6e7c2874b1b237500a00811ec06cd8c5`；准备时MAIN
  `cef1218e4d8b4e3dfae5c59d82af85b0310d0a48`。独立rootless Podman，
  不接入或改动其他项目的k3s。单物理推理机四卡，不冒称物理双机。
- backbone本地；LoRA沿真实远端既有只读gzip发布获取。每运行下载缓存初始
  为空，允许原生合法复用。请求中不打包，真实传输/竞争/本地物化保留。

## 验收与安全

仅一次完整观测。启动前重新核对GPU空闲、预计新增20GiB的磁盘门槛、102GiB
空服务内存准入、147个保护文件、三份用户修改SHA、远端状态与执行源码。
服务72/80GiB、swap2GiB；回放及检查3/4GiB、swap0，既定CPU隔离。
运行期间不安装、构建、测试、推送、压缩或维护远端，也不并发其他重型任务。

完成后先实际释放，再核对全部4000请求、原生adapter/token/时钟、远端传输和
物理GPU区间并集。按冻结V1/单轮V2及D394共同SLO报告结果；失败和不达标
均保留。W2 adapter与W0不同，不能将D502 W0输出当成相同输入的数值对照。
独立数值正确性仍未知，原始n_correct保留null；原生合同完整性单独报告。
pipeline/consolidation只报告实际触发，出现pp>1须先补计量审计，不套pp1结果。

## 来源与下一步

RAW：`results/ieee_tc/hydraserve_qualification/d582_20261016`（历史实验根目录）。
准备证据：`paper_results/ieee_tc/hydraserve/d582_lifecycle_full_w2_prelaunch/`。
PLAN SHA `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`；
V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。

随后完成Loquetier新适配完整W0/W1/W2及其余7B基线，再依据完整比较优化
PrimeLoRA；G2/跨场景配置、3B、dLoRA、A1–A5、S1–S13、成本/CE、审稿文档
和最终图表全部保留。接近目标前不画图，不把适配成功写成性能达标。
