# D450：完整 ElasticLocality 对照的联合资格

日期：2026-10-12。阶段：剩余 7B 内部对照资格，不是新一轮参数搜索。
完整 PLAN、当前状态、2026-10-11 基线规则、冻结 V1 和单轮 V2 已读取。
用户再次批准一次性只读压缩交付缓存；复用已完成的 D78/D80（两模型归档
合计 1,581,215,261 B），不重建、不新增解压池，不改权重与负载。
离线发布在部署通知前；必要读取、线上传输、竞争与本地物化仍计入请求路径。

## 问题、依据与非目标保持

D448/D449 分别完成二元 GPU-locality 路由和普通制品 LRU，但不能把两个
孤立检查当成完整对照。尤其 no_handoff 单独仍允许 ready 后的稳态准备。
本轮检验：将它们组合后，是否仍通过真实 IEEE 生命周期、扩缩容、按需 native
加载与物理引用路径执行，同时没有主动收益规划、handoff 或有效容量策略混入。

历史依据为 D371/D397 封存 Full、D446 五系统 W0 以及 D448/D449 的实现记录。
2026-10-12 再查 [vLLM 官方 LRU](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/utils/cache.py)
与 [Production Stack 负载感知路由](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html)。
前者分离普通访问顺序与 pinned 保护；后者明确缓存亲和与当前负载需要权衡。
它们不是本项目的完整 LoRA 对照实现；本轮不移植新评分公式、不称复现这两个系统。

新配置 `configs/ieee_tc/20261012_d450_7b_elastic_locality_100.yaml`：

- `ieee_system_variant=elastic_locality_v1`，输出身份 ElasticLocality。
- `ieee_selection_strategy=gpu_locality`，slack 固定为 1，不搜索。
- `file_cache_policy=demand_lru`，handoff、hierarchy、coordination 和旧主动转发显式关闭。
- 与 D371 相同 backend/model/FP16/TP1/并发14、实例1–4、原生cache预算、
  autoscaler/profile/trace/生成/真实远端/资源包络；完整配置差异由测试逐字段核对。
- 仍用 `faaslora_full` 内部容器路径来复用真实所有权，不表示测试对象是 Full。
  场景必须为 `tc_ieee_elastic_locality`，不能混入 Full 主结果。

没有改变 Full 九个公式、实际扩缩容决策、native demand 或文件 LRU 函数体。
没有改旧消融 non-feature hash 的含义，避免追改历史执行身份。

## 实现与验收

既有 runner 增加完整机制合同和 actual-owner 启动核对，不给缺失开关补默认值。
拒绝 inherited proactive diagnostic；直接调用诊断也在任何准备前拒绝。
输出配置合同不自动证明机制触发，`mechanism_execution_qualified=false`。

既有数据分析入口增加 ElasticLocality 专用机制检查：

1. launch、终态和逐请求路由策略一致，有真实选择尝试；重试数不是请求数。
2. activation 存在且所有 handoff plan 为空、preparation disabled。
3. residency/file/GPU/native-HOST proactive journals 显式为空，无 planning CPU work。
4. 实际文件策略为普通 LRU；替换若发生须有成功容量事务证据。
5. 缺字段拒绝，不把缺日志解释为“没有触发”。

这些检查叠加已有原生 token、时间分解、来源/引用、请求集合、实际 GPU 释放、
真实双端 HTTP 与资源审计；不单凭机制检查声称数值正确、SLO 或系统优越性。
GPU 命中/驱逐在短前缀中未发生时报告实际次数，不注入压力强造命中或淘汰。

## 预登记真实资格与未完成项

复用 `scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh d450`，只增加冻结
campaign 分支；100 请求来自原 W0 前缀，共同部署通知60秒、1800秒保护、
真实远端已发布工件、同固定输出。独占输出目录，禁止覆盖或相同键重试求胜。
运行前源代码须提交备份、GPU idle、磁盘/内存及实际 worker 保护通过。
本文件登记时尚未启动 GPU 运行，CPU 检查结果与真实回放结果分别补记。

随后按受影响合同推进 500-ID 功能覆盖、原1000和单轮4000，不重跑已封存
Full/五系统W0。短前缀不能代替4000排名，也不以它优化外部系统。
HydraServe 平台访问问题另待确认；W1/W2参考/G2、必要PrimeLoRA优化、3B、
A1–A5/S1–S13、CE补充、两轮审稿文档及后续图表仍未完成。

## CPU 联合资格完成

最终 621 项回归通过（58.387 秒），分析器 38 项通过（0.20 秒）。聚焦检查
与回归重叠，不作独立重复或性能样本。实际控制路径测试证明关闭主动准备时
仍可扩容；原生 runtime 为合成夹具，不冒称模型 GPU 验证。
AST 对比上一提交：runner 只改构造/合同核对/观测/诊断禁用四个函数，新增
一个纯配置核对函数；另592个既有定义不变，控制/公式/需求事务未改。

失败过程完整保留：LLM 解释器没有 pytest，未执行用例；改用 native 环境
进行整个控制测试时，依赖自动构建的子进程使辅助域发生38275次high事件，
峰值3441676288B，按精确身份SIGTERM停止，max/OOM/swap均0。没有扩大包络。
轻量环境首轮621项中一项新夹具缺少coordinator metrics字段，补齐测试夹具
后12项聚焦及完整621项通过，未在生产路径加fallback。源码打包审计还曾
误读保护检查不存在的missing字段，改为其现有pass/changed合同后通过。

九个辅助域均已核对实际递归进程为空及InvocationID后停止。其余域high/max/
OOM/swap均0。147项保护条目、三个用户文件、D371配置和冻结协议SHA不变。
完整来源、日志、失败、资源身份见 `paper_results/ieee_tc/elastic_locality/d450_combined_qualification.json`。
使用 run-experiment 的既有入口、github-sync 的明确文件暂存与备份检查。
远端管理/回收脚本仅从D371复用并替换新目录、运行名及资格来源，尚未执行。
