# D1031：复用实际成功分配的启动显存画像

2026-11-03。承接 D1030；完整目标 active，主线仍为 3B PrimeLoRA。
本记录替代当前交接位置，不修改用户 dirty 的 EXECUTION_STATUS.md。

## 依据与边界

已完整核对 PLAN、目标和冻结 V1/单轮 V2。D1029 的 3774/4000 联合达标、
8167.260452 GPU-s 保留，未达 3800；独立数值正确性仍未知，不能进入排名。
不重跑 D1029 来获得暖缓存。D1030 的编译器子因子根因未确定，不继续该支线。
D1029 自然副本即使编译已热，仍有约 28 秒初始化，包含重复显存估计。

本轮使用 run-experiment、serving-llms-vllm，复用原启动、画像和资源入口；
github-sync 用于测试后可回退检查点。技能中的通用吞吐/时延常数不替代
冻结的本项目指标，不升级后端、切换服务公式或修改宿主驱动。

唯一假设：相同实际配置/硬件下复用已成功启动的 KV 容量，减少后续副本
重复的内存估计及临时图捕获，可以改善完整 workload 的联合 SLO 或 GPU-s。
这不是只跑启动微测即可接受的结论，也不保证弥补 26 条失约请求。

## 对 D1030 候选的必要细化

官方 vLLM 0.30.0 的 startup_plan 保存建议值，而非必然等于已使用容量。
直接开原生开关会混入容量变化。因此本候选借鉴其匹配和空闲显存规则，
通过现有公共 worker_cls 复用**实际 determine_available_memory 返回值**，
仅在原生 KV 初始化、实际服务图捕获全部成功后发布。不开启另一个原生
建议容量缓存，不改安装源码、不复制编译缓存、不放宽编译哈希。

- 默认关闭，仅 ieee_startup_memory_policy=exact_native_startup_memory_v1。
- 身份包含全部实际 AsyncEngineArgs、原生硬件/配置指纹、驱动、torch/CUDA、
  编译环境及四个编译器分项哈希、已安装关键源码；loader 和 GPU budget
  显式保留，不只依赖 graph hash。既有封存模型由路径和文件元数据绑定，
  不每副本重新扫描 GB 级权重；它不是新的权重内容校验声明。
- 不排除 admission profile ID；因此画像/Full 或 loader 不同可能分别冷启动，
  不强制共享。等价物理卡仅不把 placement 当内存依赖。
- 记录缺失、失配、损坏、空闲显存低于原生记录基线，仍执行原生完整画像。
  不改参数重试、不缩 KV 求通过。首次成功记录不被后续 run 覆盖。
- 命中时仍执行原生 profile_run 编译、KV 初始化和真正服务图捕获；只省去
  重复内存估计及临时图捕获。失败捕获不发布缓存。
- JSON 位于既有 VLLM_CACHE_ROOT/exact_native_startup_memory_v1；不含模型
  权重或请求，不创建另一套数据/实验框架。日志保存 miss/hit、精确容量、
  原生 free-memory 基线及 producer，不把未命中隐藏为成功优化。
- IEEE 九公式、router/planner/admission、槽、批大小、token 轴、SLO 不改。

原始来源（2026-11-03 核查）：
[vLLM startup_plan](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/startup_plan.py)、
[原生 worker 调用路径](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu_worker.py)。
本机源码 SHA 仍按 D1030；后端建议值与实用值的区别来自本机实际调用点，
不是额外人为指定 KV 容量。

## 有界验证和后续顺序

1. CPU 测试：精确容量、首次发布、空闲显存不足、失配、非法字段、捕获失败、
   原生调用保留及与 bulk loader 共存。最初 36 passed；补充真实 native-hit
   分支测试后记录最终数量，不用测试替代 GPU 数值正确性。
2. 复用 D701 的 24 adapter、76 波、423 请求，生成新 D1031 service/preparation
   画像身份。旧 timings 不改标签，固定输出长度初始化复用；不生成负载/工件。
3. 真实命中路径需要一次最小原生验证：必须与该记录同实际配置，核对原生
   token/binding、KV 容量一致、估计跳过、真实图仍捕获和完整释放。不为性能
   重复 Full。若必要画像已覆盖命中，则不加冗余短 replay。
4. 验证后冻结，一次新键完整 4000 请求 W2，全部 initial/natural、cold/hit
   和资源开销计入；D648 SLO 与 12083.063870 GPU-s 预算不变，无 CI。
5. 最多两轮针对性验证；无法清楚复用或完整收益不支持则保存并返回稳态瓶颈，
   不追加缓存调参支线。未达性能与数值正确性要求不称已完成 G1/G2。

配置：configs/ieee_tc/20261103_d1031_3b_startup_memory_source.yaml。
画像准备：paper_results/ieee_tc/p2_backend/d1031_3b_startup_memory_profile/。
原始：results/ieee_tc/p2_backend_qualification/d1031_20261103/。
已完成 CPU 准备：423 请求、76 波、24 工件，timing_samples_reused=false，
model_backend_id=native-model/1f20d3c08a031104c453596febee0a24064a4dc95e60d0ecb2dd42f1eddc1ba8。

## 未删减的主线

测前完成：39 项 CPU 测试通过（19 既有警告，11.72 秒），包括实际原生
determine_available_memory 的命中分支（profile_run 恰一次，临时图估计零次）
以及真实导入的身份采集函数。静态 shell/diff 检查通过，147 旧结果保护项
零变化。两个用户 dirty SHA 与 D1030 相同。测试/准备辅助域均已退出；
未保存全程峰值，不虚构资源全程零事件。

测量前无损压缩已关闭 D1029 main_outcome.json；双次解压和源 SHA 验证后
回收 439906304 B，原始 result、请求 CSV 和所有失败证据保留。回执为
paper_results/ieee_tc/storage/d1031_closed_d1029_outcome_20261103.json；
可用磁盘 161621721088 B。未进行整盘重新扫描或删除其他项目数据。

当前下一步：备份检查点后，只执行一次
`bash scripts/ieee_tc/run_d298_capacity_source.sh d1031`。
本节不是已经完成 GPU 画像的声明。

## source1 启动错误保留与接口修正

12:16:06 在 tmux tc-d1031-3b-startup-memory 启动；代码
c54c3ddd0bca10bab1e7e2fa129ce48e6f32ccad 已推送并核验 origin 的 V2 分支和
faaslora_origin/backup/d1031-startup-memory-20261103。备份不冒称分歧 V2
最终集成。远端 PID1984520 未变，两端 1000/full，无远端配置变更。

source1 在模型加载后、内存画像前失败，完成 0 条请求：新增身份收集错误地
对包含 bytes 的 torch_key 使用 JSON hash_factors。原生编译器实际使用
str(factors) 编码及 safe_hash；修正只对齐这个接口，不删除因子或静默重试。
原异常被旧 physical lease 的 lifetime-unqualified 收尾错误遮蔽；以 worker
原始 traceback 确认真正原因，不修改旧测量/保护代码掩盖失败。

独立 watchdog 完成收尾：service_path_removed=true、native GPU context
release confirmed=true；55 样本，service peak=5722959872 B，high/max/oom/
oom_kill 均 0。未形成显存缓存，原始请求分母/失败状态不改。worker
/tmp/faaslora_worker_fik_9l0k（native PID4030375）已归档到 raw 的
source1_worker_logs.tar.gz，SHA
d4f6c1ca94ddbd5b0153f81e858e8d1b9a0165e1750ea4cfc38e993aa5d117da。

补充真实 bytes 类型及 native 编码测试后，发现测试自身错误地把 SHA256 前缀
与 safe_hash 的 MD5 前缀比较（39 pass/1 fail），已按实际源代码修正。
这也限定 D1030 CPU 探针的 7e70d33753 是其诊断 SHA256 前缀，不是原生
compiler_hash；D1030 两份真实 cache_key_factors 的不同比较仍然成立。
不得用本次记录抹掉历史 failed attempt 或宣称性能已被否决。

修正后的第二次尝试使用 FAASLORA_D1031_ATTEMPT=source2，保持相同配置、
画像请求、冻结口径及内存保护，输出 source2*.json/log；source1 全部不覆盖。
此为具体启动错误的修复，不是更换参数或重新抽样寻找胜出结果。
修正后 40 passed、19 既有警告，13.56 秒；实际 compiler interface 的编码测试
通过。原生开关仍关闭，未生成建议容量记录，也未修改底层库。

两模型同版本 W0/W1/W2/G1/G2、独立数值正确性、与保留基线的最终对比、
A1–A5、S1–S13、CE/Pareto、两轮审稿及 B1/B2/C1/C2/C3 文档、达标后图表和
最终 V2 集成仍待完成。D810 的基线和 HydraServe 隔离适配已完成部分保留，
不接管共享 k3s，不重开历史适配。旧结果、两个用户 dirty 文件均保护。
