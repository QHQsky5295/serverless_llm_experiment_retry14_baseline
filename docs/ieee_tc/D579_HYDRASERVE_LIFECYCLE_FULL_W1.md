# D579：HydraServe 同配置完整 W1 扩容回放

2026-10-16。准备通过，尚未启动；此文件不是性能结果。

## 问题与既有证据

D573 修复隔离平台不同 worker 之间额外的生命周期互斥，不改变官方调度、
加载、pipeline 或 consolidation 策略。D574 100 请求资格及 D576 完整 W0
已完成；不重复这些运行。D576 联合 SLO 为 68.225%，不能由适配正确推断性能
改善。现在只补受影响的完整 W1；保留旧 D546 及其全部失败/慢请求证据。

## 冻结执行身份

- 模型 Llama-2-7B，500 逻辑 adapter 可访问；原有 4,000 请求、源 seed42。
  发展期身份41，不冒称独立留出或正式数值正确性资格。
- W1 沿用 D546 配置：8×500，阶段内原到达间隔减半，阶段间30秒；不等待
  drain、不强制制造八次扩容。输入配置逐字节相同，SHA
  `22e533264ed007c56777cd1da3c9ff7b3d7a45a6c2628a9b54b9e77216be19f7`。
- 原生 batch8/use_cache0/slow_expr0、idle retirement30秒；共同部署通知60秒、
  保护超时1800秒、fixed_length_greedy_v1不变。预期458224原生输出token、
  132个实际出现adapter，不通过改目标长度改善指标。
- BASE `80e08b4c6e7c2874b1b237500a00811ec06cd8c5`；准备时 MAIN
  `c1cd00e3d2c2d30fc515a1734c288a40410907b4`。共享执行源码与 D574/D576
  一致，16个原生适配文件与D574逐字节一致。独立rootless Podman；无关k3s不动。
- 真实远端复用既有只读gzip发布，cold per-run下载缓存；请求路径不打包。
  backbone本地，仅LoRA真实远端。一个物理推理节点、四卡，不冒称物理双机。
- 一次完整回放，不取多轮最佳、不改变共同阈值。W1不是重跑同一执行键；
  D546使用旧平台实现，D579使用已登记的D573适配。

## 验证和安全

准备阶段受3/4GiB、swap0及指定CPU限制，峰值138293248B，所有内存事件0；
实际进程树为空且资源域消失。准备不是新GPU实验。

启动前重新检查：推理机150GiB磁盘门槛、预计新增20GiB、空服务102GiB内存
准入、GPU空闲、受保护147文件和三份用户修改SHA、源码快照与远端内容身份。
服务72/80GiB、swap2GiB、指定CPU；外置监控/回放3/4GiB、swap0。仅一个重型
任务；回放期间不构建、测试、推送、压缩或维护远端。按实际所有权收尾所有
worker及远端服务；不能由进程退出推造GPU释放时间。

结束后：完整原生token/输入/adapter/时钟绑定与物理生命周期审计；使用冻结
V1+单轮V2、D394共同SLO，报告联合达成、GPU-s、P95/P99 TTFT及阶段分解。
与D546逐请求核对输入与原生输出变化；独立数值正确性尚未知，n_correct保留null。
pipeline/consolidation只报告实际事件，未触发不归因。保存所有失败与不达标。

## 来源与后续

RAW：`results/ieee_tc/hydraserve_qualification/d579_20261016`（历史实验根目录）。
准备清单、preregistration、source manifest与归档位于
`paper_results/ieee_tc/hydraserve/d579_lifecycle_full_w1_prelaunch/`。
PLAN SHA `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`；
V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。

完成并保存本轮后，继续HydraServe W2、Loquetier新适配完整矩阵及其余7B基线；
之后按完整对照优化PrimeLoRA。G2/跨场景配置、3B、dLoRA、A1–A5、S1–S13、
成本/CE、审稿文档和最终图表均未删除；接近目标前不画图。
