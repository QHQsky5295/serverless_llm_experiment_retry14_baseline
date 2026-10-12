# D477：HydraServe 独立 GPU 容器资格进展

2026-10-12。当前主线遵循最新用户要求：HydraServe 必要适配和实际结果 → 完整
可用基线对照 → PrimeLoRA 优化。本次没有新增 PrimeLoRA 候选或重复已封存回放。

基线仓库已备份至 `504cd97ac4a50d338fde3ab5c7ed82eddd312ba5`，远端 HEAD 独立核验
一致。详细记录见基线仓库 `docs/ieee_tc/D477_ISOLATED_GPU_CONTAINER_QUALIFICATION.md`；
100 项来源 SHA、各次失败和闭合证据见 `paper_results/ieee_tc/hydraserve/d477/`。

实际通过：独立、无特权 GPU 容器只访问指定 UUID；NVML 命令成功退出；worker
实际父组、128 MiB 子组内存上限、swap 0、CPU 绑定和释放经过核验。外层 taskset
曾被 runtime 重置为 48 核，这次失败保留；最终在容器入口、执行检查之前绑定。
镜像只获取一次并复用；无新模型、LoRA、trace、原生后端替换或基线性能调参。

尚未通过：整个 HydraServe 服务的资源资格、Kubernetes、原生修改版 vLLM、
流水线/合并时 LoRA 数值和完整性能回放。Podman pause 位于另一资源域，已按
准确身份释放，但不能因此宣称它先前受共同父组限制。用户级 cpu/cpuset 委派
仍缺失；原有 k3s 完全未动。下一步核查忠实的项目独立编排/平台接口路径，
必要时明确提出最小管理员协调，不以伪接口、模拟 ready 或裸 vLLM 冒名。

本次平台证据不是新的论文性能行，不用于宣称 HydraServe 快慢，也不修改冻结
G1/G2、联合 SLO、单轮规则或 IEEE 公式。W1/W2/G2、3B 和完整后续矩阵仍开放。

复用既有归档器无损压缩三份关闭的失败日志，释放 188,305,408 B；两次解压 SHA
一致，147 项保护清单零变化。回执：
`paper_results/ieee_tc/storage/20261012_d477_closed_outcomes_lossless.json`。
磁盘仍接近 150 GiB 启动底线；下一构建需重新核验真实新增峰值，不能降低门槛。

原始记录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d477_20261012/`。
