# D779：dLoRA 实际入口的环境缺项及最小修复

本轮遵循完整 PLAN 和基线有界适配规则，已从纯接口检查转入真实远程、完整
500-adapter 池及原 100-request prefix 的共同服务资格。仍采用历史成功的
3B DP2/TP1、migration_type=3 参数，不调整基线算法或搜索性能配置。

## 实际结果及证据边界

完整池已准备，随后原生 Python3.9 进程因缺少 `aiohttp` 在导入 HTTP 接口时
退出。Ray、模型和 GPU worker 均未初始化；不是 dLoRA 性能差的证据。
这次 qualification 失败永久保留，不报告有效 TTFT、TPOT 或 GPU-s 排名。
已写入的物理 acquire 没有合格 native worker lifetime，原物理汇总仍为
incomplete；实际清理证据不能反向补造一段有效性能测量。

前轮 CPU 检查未覆盖这一原生 HTTP 导入，因此不能用其通过来宣称完整入口
已经合格。本轮不再新增无变化的广泛回归，只验证已确认缺项。

## 处理和验证

依据现有 Loquetier D441 的同类适配历史，复用八个有 SHA 的 Python3.9 HTTP
依赖，使用共同接口既有 aiohttp3.13.3。其[官方版本元数据](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/pyproject.toml)
支持 Python>=3.9。安装前 dry-run，安装时禁止解析额外依赖、源码构建和
无 hash 工件。实际前后包清单确认所有原有版本不变，仅新增八个缺失包。

实际原生环境导入 D778 EngineManager/共同 handler 并完成一次 CPU HTTP
往返，之后关闭端点；PyTorch2.1.2+cu121、Ray2.51.2 保持原样，CUDA/Ray
均未初始化。此处不宣称完成 LoRA/GQA 数值正确性或实际推理资格。
新接口依赖作为下一执行键的环境差异记录，不能在旧键上改名重跑。

## 资源与保护

- 服务峰值 20,533,035,008 bytes，high/max/OOM/swap 事件为零。
- 实际服务域和启动辅助域已关闭，无遗留 GPU compute；所属远端服务均关闭。
- 500 个 adapter 的 4,000 个重复缓存文件逐项与原池核对，保留 resolver 和
  服务端传输证据后删除 objects；payload 19,482,573,296 bytes，实测回收
  19,497,287,680 bytes。没有删除原始工件、输入、失败日志或唯一证据。
- 环境修复辅助域峰值 347,373,568 bytes，无内存保护事件，已结束并关闭。
- 145 项旧投稿与三项用户修改 SHA 保持；无共享 k3s 操作。

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d779_20261024`。
小型归档目录：`paper_results/ieee_tc/baseline_audit/d779_dlora_entry_failure/`。
baseline 对应说明：`docs/ieee_tc/D779_DLORA_HTTP_ENVIRONMENT_20261024.md`。

## 主线不变

下一使用修复后的原生环境、D778 source view、原工件及负载，做实际加载和
必要 LoRA/GQA/共同 HTTP 验证，随后只做既定 3B W1/W2。Loquetier/HydraServe
3B、PrimeLoRA 完整对照与优化、同版本 Full、A1–A5、S1–S13、CE/Pareto、
两轮审稿文档与最终图表全部保留。整体目标尚未完成，不从本轮环境修复推称
PrimeLoRA 已最优；不为提高基线 SLO 开性能搜索。
