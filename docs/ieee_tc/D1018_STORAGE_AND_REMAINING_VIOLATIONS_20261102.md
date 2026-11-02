# D1018：闭合证据无损归档与 3B 剩余违约分析

2026-11-02，接续已闭合并备份的 D1017。上一 goal turn 为 PROGRESS：
完整 4000 请求、阶段/真实远端/资源审计和 Git 备份全部完成；不重复回放。
本轮完整重读目标、PLAN 与存储规则，当前主线仍为 3B PrimeLoRA 优化，
不删减最终两模型主比较、A1–A5、S1–S13、正确性与论文交付。

## 先恢复必要存储余量

当前 GPU compute 为空，D1017 已释放，文件系统可用 160259006464 B，
略低于新重型运行的 150 GiB。复用 D848 全盘审计：其他项目/共享目录的
权限和用途缺口仍保留，不重新做无边界扫描、不操作共享 k3s 或其他项目。

只选择两个已完成并拒绝的控制候选的 `main_outcome.json`：

- D1006：`d1006_full_20261102/d1006_3b_no_forecast_release_w2_full`，
  原日志 1066550465 B；关闭预测释放导致资源增加而联合 SLO 下降。
- D1007：`d1007_full_20261102/d1007_3b_min2_forecast_release_w2_full`，
  原日志 573079657 B；最小两实例仍未通过联合 SLO，历史权衡保留。

两者实际 launch 均为 pass=true、service_path_removed=true、
native_gpu_context_release_confirmed=true，目前无压缩件或压缩回执。
未因结果负面而删除数据：复用既有 `compress_closed_main_outcomes.py`，
原文件 SHA、gzip SHA、两次解压 SHA、文件身份和无活动引用全部通过后，
仅移除未压缩副本。每个原始字节仍在同目录 gzip 中，成功/失败/开发身份
不变。其完整请求结果、launch、allocation、curated 审计不压缩；当前
D1017/D1000 的原始分析输入不动。以后重做旧 outcome 投影须先按回执
核验并恢复原路径，不能把压缩后路径缺失误判为证据丢失。

任务使用原 CPU 2/3/26/27、MemoryHigh/Max=3/4 GiB、swap0、低 IO
优先级；不与 GPU 测量或构建并行。不改阈值、不触碰模型/LoRA/trace，
不覆盖旧投稿数据或用户两个 dirty 文件。实际回收量以完成回执为准。

## 同时推进的性能问题

D1017 已有完整阶段表：3738/4000 联合达标，尚差 62；剩余 156 个仅
TTFT、93 个仅 TPOT、13 个二者违约。106 个 TPOT 违约中 103 个属于
1–64 token；169 个 TTFT 违约中 114 个 dispatch 前为 Remote。这是
条件诊断，不足以据此改变扩缩容、路由或删除任何状态确认。

本轮先读现有 native 控制事务、旧已否决优化与 vLLM 官方边界，寻找一条
新的可证伪解释；不重复 D1005 最大四实例、D1006 禁止释放、D1007 最小
两实例，不再次调 D747 描述库/GC/整图、解压/线程或已关闭的提前路由方向。
`system-profile` 用于资源与已保存窗口的瓶颈核查；当前不插入新 observer，
生产代码、模型配置和实验框架未改，不将维护当作性能实验。

## 完成回执：存储不是当前阻塞

无损压缩正常退出，共释放 **1435938816 B（约 1.337 GiB）**；可用空间由
160257937408 B 增至 161693720576 B（约 150.59 GiB）。精确路径、原文件
SHA、gzip SHA 和两次解压校验记录在
`paper_results/ieee_tc/storage/d1018_closed_outcomes_20261102.json`。
两个原始副本对应的回收量分别为 934526976 B、501411840 B。两次解压
SHA 均与压缩前原文件一致；`audit_sha256=null` 不是新来源资格通过，
此维护只保证当前字节可恢复，不把旧候选升级为可直接复用的主结果。

压缩 scope 已 inactive/dead。配置了原定辅助内存/CPU 上限，但没有保存其
结束前 memory.peak/events 快照，因此不填写虚构的峰值或零事件。后续
分析/测试也使用独立的 3/4 GiB、swap0、CPU2/3/26/27 辅助 scope，全部
退出；没有与 GPU 性能测量重叠。147 项历史保护检查通过，两个用户 dirty
文件的 SHA 与 D1017 保持一致。没有触碰共享服务、其他项目或唯一工件。

## 本次排除：重复启用异步，或无依据增加控制延后次数

核对本机 vLLM 0.30.0 的 EngineCore、已有 D548/D549 记录，以及当前
`core_timing.attach_step_components` 和 `lora_metadata_publication` 的原生
边界：本项目既有路径已经使用 async scheduling 和两批异步流水线。
不能把“打开异步调度”重新登记成优化。版本对应的
[官方调度配置](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/scheduler.py)
将 async 与请求/LoRA 容量分开定义；
[官方 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
仍在引擎循环中处理输入和 utility。异步执行不意味着任意 CPU 控制事务都
已经从关键路径消失。

当前 `atomic_control_yield_v1` 已在完整 PrimeLoRA 控制事务后返回原生循环；
不再假定仍是无界连续清空控制队列。D998 的只读事务延后已经得到完整负面
结果，不能凭 D1016 的控制交叠占比再次增加延后次数或换一个配额常量。
[vLLM 旧版性能文章](https://vllm.ai/blog/2024-09-05-perf-update)支持检查 CPU
调度/执行重叠，但旧 V0 多步参数和 H100 收益不能直接移植到当前 V1/3090。
本轮不实现 duration-credit、额外多步或重新开放已关闭的线程/整图方向。

## 新的只读证据：请求并发容量与 LoRA 驻留容量是两个问题

当前 3B 配置 `max_num_seqs=8`、`runtime_concurrency_cap=8`、
`max_loras=8`、`max_cpu_loras=32`。本机及
[同版本官方管理器](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
表明 `max_loras` 同时决定 GPU 原生 active cache/slot 数；它不是只影响
单批请求数。HOST registry 容量另由 `max_cpu_loras` 决定。
[LoRA 配置源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py)
也表明 slot 变化会改变 buffer/编译身份，不能沿用旧延迟画像而不说明。

已回查 D343、D710/D711 和当前 3B 配置历史：D343 是 **7B 并发与 slot
同时增加** 的试验，并不是当前 3B 保持并发不变的驻留容量对照；D710
后来的旧排程已被 D711 撤回，不据它重开 7B。当前检查范围内未找到已闭合
的 3B slot-only 对照。这不是声称所有历史都不可能存在该试验；下一候选
登记前仍须核对其精确配置来源。

复用 D1017 已封存、全部 4000 请求的阶段表，只扩展现有 D703 analyzer
一个离线模式，不再读取/复制完整原始结果。按同一个 native owner 内的
admission 时间排序，记录距上次相同 adapter 请求之间的不同 adapter 数；
副本重建后的 owner 分开，不能跨 epoch 合并缓存历史。8/32 分界来自当前
GPU/HOST 槽数，不是按观测胜负选择的新配置。

| admission-ordered demand 类别 | 请求数 | 实际调用 native load | 联合违约 | load 且联合违约 |
|---|---:|---:|---:|---:|
| 本 owner 首次请求 | 954 | 878 | 148 | 146 |
| 重复，间隔 0–7 个不同 adapter | 1755 | 224 | 57 | 5 |
| 重复，间隔 8–31 个不同 adapter | 1091 | 969 | 52 | 48 |
| 重复，间隔至少 32 个不同 adapter | 200 | 167 | 5 | 4 |
| 合计 | 4000 | 2238 | 262 | 203 |

新产物：`paper_results/ieee_tc/p2_backend/d1018_demand_reuse/summary.json`。
它绑定阶段 audit、请求表和本次 analyzer SHA。六个 owner、4000 个唯一
request/ordinal、原达标计数全部核验。4 项单元测试通过，覆盖实际排序、
distinct 而非请求数、epoch 隔离、7/8/31/32 边界、原 load 计数保留及坏输入。

**解释边界：**969 次 load 中 834 个来源为 HOST、133 个为 file、2 个为
staging。这支持核查“较小 GPU 驻留窗口是否反复产生准备事务”，但不证明
969 次均能通过增大缓存消除。数据未重建主动准备、真实 LRU touch、pinning、
驱逐原因或并发 acquisition 顺序；first-in-owner 甚至有 76 个实际 GPU-hit，
而短距离重复也可能 load，恰说明 demand 历史不能冒充 native cache 状态。
不能从本表减去时间、推算新命中率或声称能补足尚差的 62 个联合达标请求。

## 下一主线与本轮边界

下一步只检验一条新假设：**保持并发8、原模型/请求/九公式和控制规律不变，
GPU 驻留槽数与并发数解除相同数值绑定，能否减少反复加载及其干扰？**
先复用已有原生 pool/KV/storage 回执，核对容量与画像依赖；仅在它与历史
对照确实不同且物理可行时，登记一个 slot-only 候选。32 是当前 HOST 槽数
给出的可核查候选上限，不是本轮已经冻结/启用的最佳值。不能同时提高并发、
换 autoscaler、放宽预算或换 SLO，也不无限扫描多个 slot 数。

若候选成立，沿用既有原生 source/metadata/copy 资格和画像入口，仅补因
slot/buffer 改变而必要的检查，然后一次完整4000请求；不能用微测代替
主比较或给 profile 换标签。若容量/历史证据否定该方向，则保留否定结论，
不以一个没有新依据的回放填充进度。本轮尚未改变模型配置或启动新 GPU 运行。

最终保留 D1017 的 3738/4000、8084.812476 GPU-s 原观测，`n_correct=null`、
非 G1/G2 排名资格不变。7B/3B 最终同版本 W0/W1/W2、G1/G2、baseline/dLoRA
剩余点、A1–A5、S1–S13、正确性范围、审稿文档/图表与最终 V2 集成全部开放。
HydraServe D570/D810 的隔离适配和已有结果不重开，共享 k3s 不操作。

### 分析修改清单与复现入口

- `scripts/ieee_tc/audit_d703_3b_full.py`：新增纯离线 demand-distance 汇总模式；
  原有分析模式及生产服务路径不变。
- `tests/test_ieee_tc_demand_reuse.py`：4 项小型语义/拒绝坏输入测试。
- 本文与两个 D1018 JSON：存储和分析回执；D1017 文档只补发布后的交接。
- 未添加 observer、GPU instrumentation、运行期策略或新工件/负载；不绘图。

分析入口（使用上述辅助资源限制）：

```bash
python -B scripts/ieee_tc/audit_d703_3b_full.py \
  --diagnose-demand-reuse paper_results/ieee_tc/p2_backend/d1018_demand_reuse/summary.json \
  --stage-audit paper_results/ieee_tc/p2_backend/d1017_restore_descriptor/completion/stages/diagnosis.json
```

该命令已成功执行一次；输出采用 exclusive-create，复核时使用新分析路径，
不覆盖封存文件。使用 `analyze-results` 分开观察与推断，`serving-llms-vllm`
用于版本相关配置核对，`system-profile` 要求的修改清单如上。冻结协议和
用户指令优先于技能中的通用阈值、画图或统计建议。
