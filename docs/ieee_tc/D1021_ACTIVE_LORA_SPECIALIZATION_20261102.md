# D1021：驻留容量与活跃 LoRA 计算网格分离

2026-11-02。当前状态：唯一 GPU 画像已完成，23:34 严格导出通过，资源已释放。
下一项为 D1022 完整 W2；下方准备和运行中记录保留原时点，不再表示仍运行。
本文承接 D1020，不覆盖用户修改的 EXECUTION_STATUS.md。完整目标保持 active。

## 主线、已有结果和本轮边界

已重新完整读取目标、PLAN、冻结 V1 与单轮 V2。PLAN SHA 为
`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`；
V1/V2 SHA 分别为
`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`、
`193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。

D1020 完整 3B W2 为 3746/4000 联合达标、8135.629131 GPU-s、P95 TTFT
1940.053078 ms。仍比冻结 3800 门槛少 54 条，虽低于 W2 预算
12083.063870 GPU-s，却不是 G1/G2 可行点。n_correct 仍未知；原生绑定、
token 与计量通过不替代独立数值正确性。不能与旧版本 W0/W1 拼接。

HydraServe 已完成隔离适配与完整回放。用户明确原 k3s 并非本项目专用，
继续沿用已验证的 rootless Podman 路径，不获取管理员 kubeconfig、不改
共享集群、不重复基础适配。D810 的全部 3B baseline 观测与限制继续保留。

本轮先核对 D96/D111 的查询合并、D197 的消息表示、D691 的控制交错、
D864–D866 的原生 HOST 图、D998 控制延后、D1014 文件优先与 D1016 诊断。
这些都不是尚未实现的优化。本轮不修改状态确认、HOST 图或预算检查。
特别注意：D697 的 single-flight 是扩容提交序列，不是查询合并；两者不能
因名称相同而混为一个反例。D96/D111 的查询合并已保留在生产实现。

## 新证据与可证伪假设

D1019/D1020 首次把 3B GPU 驻留槽增加到 32，而请求并发仍为 8。
加载次数减少 1217，联合达标仅净增 8；这是观察，不是内核瓶颈的证明。
本轮直接核对官方与实际安装的 vLLM 0.30.0，发现：

- `specialize_active_lora` 原生默认 false，现有 runner 没有传递此选项。
- 默认捕获计数为 `[max_loras + 1]`，32 槽对应 33；该值包括原生元数据
  约定，不应写成 33 个真实活跃 adapter。
- 原生 `LoRAKernelMeta.meta_args` 在 false 时使用容量相关的默认计数，
  true 时使用当前活跃数对应的元数据。驻留 32 个不意味着每批都用 32 个。
- true 的原生捕获集合为 `[1,2,4,8,16,32,33]`，另外保留无 LoRA 分支；
  它会增加图捕获和内存成本，**不是减少图捕获的启动优化**。初始考虑的
  “当前已捕获多组无用 adapter 图”不成立，因为当前 false 并未捕获这些组。

官方依据：[配置定义](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py)、
[捕获计数规则](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/utils.py)、
[Punica 调用路径](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/punica_wrapper/punica_gpu.py)、
[图分派](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/cudagraph_dispatcher.py)。
访问日期 2026-11-02。本机 config/lora.py、lora/utils.py、lora_kernel_metadata.py
SHA 分别为
`ed6cf6575f5950f079a62f904caeae93d1a4745dd8c21788b74367883bcd4b08`、
`fc593bd7935f8f40c186b2221bf1bf3e244b6aacd8624727905648dd2753110c`、
`7669eeeb491e89df7ef4ff050a79d3934ce7a63b146fe93239f081d176c68881`。
官方语义与本机身份分别记录，不声称全部字节一致。

唯一假设：保留 32 槽的缓存收益，按当前活跃 LoRA 数选择原生计算网格，
是否能降低推理代价，并在包含新增启动、图内存和扩缩容影响的完整工作负载
中改善联合 SLO 与资源–尾延迟关系？目前没有测得 GPU kernel 时间或收益，
不能把源码推论写成加速结果。此方向不同于 D724 的 token 形状覆盖，也不
重新扫描槽数、token budget、扩缩容阈值或重复同键。

复用 D1020 请求表的小型离线核查还发现：152 条 TTFT 违约中前 25 条全违约，
其余在序号 25–99、100–499、500–3999 中分别为 2、1、124 条；TTFT 主导
阶段为 arrival→admission 的 66 条、admission→acquired 的 86 条。前 25 条
都有显著初始 dispatch 等待，不能只优化稳态后忽略启动。条件分类不是因果
时延归因，不删初始请求或从时延扣除启动。输入为已封存 requests.csv，SHA
`ed531383de5dc0cf1ea6253eb2b94b5f736b6537f261bffcf650440995a6ec5d`。

## 实现与必要验证

只在既有 runner 透传原生布尔选项；未配置保持原默认，不修改 vLLM 安装、
模型/LoRA 权重、CUDA kernel、调度或 IEEE 九公式。原生参数不支持时仍报错，
不删选项重试。严格 profile identity 保留此字段，旧时延不能复用。

复用 prepare_d701、run_d298 和既有完成长度初始化器。新长度绑定仅复用已完成
固定输出的长度统计；明确声明图、槽及 specialization 与历史源的差异，不
复用旧时延或数值资格。实际候选与直接父配置 D1019 只有一个选项不同。

已完成检查：

- 7 项 engine 参数/图配置检查通过，包括 true/false、默认、错误类型、未启用
  LoRA 时拒绝、原生拒绝不重试，以及容量 32/并发 8 不变。
- 15 项既有长度 transfer 检查和 2 项新增 specialization 检查通过；包括旧
  绑定不能接受新字段、少生成 token、未声明差异与其他参数变化拒绝。
- 静态准备通过：24 个原 adapter、76 个原波次、423 个原请求引用，无新工件/
  trace、无旧时延移植。新 model_backend_id 为
  `native-model/eabe83c26c456fea25273dbe53db6d286bfd0d00fc3aa5fffde8fa2b76dcc8f6`。
- 新配置 `configs/ieee_tc/20261102_d1021_3b_active_lora_source.yaml`，SHA
  `833cfdda2273255b5ad809236f30f3a5170f602f6ae9d30d821bc7593aaf191e`。
- shell 语法和 diff 空白检查通过。测试保留 19 项既有警告，没有测试失败。

上述四次 CPU scope 均使用 3/4 GiB、swap0、辅助 CPU 2/3/26/27、隐藏 GPU，
结束已 inactive。未保存终态前峰值，不能据此宣称完整资源事件为零。没有新
GPU 运行、远端服务操作、删除、压缩或绘图；用户两个 dirty 文件 SHA 未变。

依照 system-profile 的插桩记录：**本轮无新增/修改插桩**；只复用历史日志。
变更为 runner 原生参数接入、preflight 长度绑定、现有准备/启动入口、两个
测试文件、新候选 YAML 与派生 spec/contract/prepare JSON。通用技能中的
默认吞吐/时延示例不替代本项目冻结目标。

## 下一步与有界判定

1. 测量前备份此接入，核查即时磁盘/内存、GPU 空闲及旧结果保护；当前磁盘
   约 150.6 GiB，不能把此前通过当作新启动保证。复用健康远端，不操作共享 k3s。
2. 执行一次 `bash scripts/ieee_tc/run_d298_capacity_source.sh d1021`。这是新
   计算网格的必要画像与原生路径验证，不另加短 Full 或重复已有整个资格矩阵。
   记录实际选项/网格、原生 token/adapter、pool/KV、启动时间、内存及释放。
3. 若原生路径和画像覆盖通过，则新建 Full 身份进行一次完整 3B W2，保持
   D1020 的控制/输入/远端合同；重新测时延画像，冻结 D648 SLO 与预算不变。
   启动和图内存代价全部计入，不以单个画像均值预判整体输赢。
4. 若运行错误、语义/资源不成立，保留失败并针对具体缺陷处理；若完整结果
   退化则拒绝，不追加同键求胜。不顺便修改捕获集合或其他控制政策补救。

D1020/D1017 和所有负向候选继续保留。目标仍是同一版本跨 W0/W1/W2 的
G1/G2；两模型最终比较、A1–A5、S1–S13、独立数值正确性范围、两轮审稿
文档、后期图表与最终 V2 集成全部未删除。尚未达标，不宣布系统最优。

## 实际启动交接：23:21 后运行中，禁止重复启动

接入提交 `cdd2ed5a99fbb2e95b5be0577cb369bf3c8bd0e7` 已在测量前推送并经
ls-remote 核验：origin 的 retry14_continuous_queue_v2，以及 faaslora_origin
的 backup/d1021-active-lora-entry-20261102。后者是可回退备份，不冒充有
分歧的 FaaSLoRA V2 最终集成，没有强推或暂存用户 dirty 文件。

再次核查 PLAN SHA 未变、原147保护项通过；GPU compute 为空、约150.6 GiB
磁盘和104 GiB可用内存，实际 gated-launch 按8 GiB预计增长及原内存规则
放行。远端认证 health 为已发布gzip/artifact_timing_v2，原clock_id不变；
PID1984520仍是10月31日的既有服务。两端链路当前1000/full，远端可用
132879314944 B；没有重启、改资源或打包。没有连续远端CPU监控，保留此限制。

23:21:05，tmux `tc-d1021-3b-active-lora-profile` 执行既有 run_d298 的 d1021
入口，显式取消三项详细 core timing 环境变量。实际资源域：

- aux：`primelora-tc-aux-903174ec2cbb486da153865a4f296c78.scope`；
- service：`primelora-tc-svc-9a97a0603583498f81c60c2274241403.scope`；
- service InvocationID：`5faf340deb444913bc6deba17e98986d`。

样本84时原生 GPU worker PID2269373 已持有GPU0（9246 MiB），服务进入
engine_initialization，尚无 source1.json/launch 终态。service峰值
5699629056 B、主机可用107731865600 B、磁盘161700741120 B；high/max/OOM/
swap/foreign compute/escaped 均无观察事件，无保护告警。这是运行中样本，
不等于整轮资源或原生计数验收。

发现历史 D998 aux scope 仍显示 active，但 TasksCurrent=0，仅31.5 MB剩余
记账；它不是正在执行的旧GPU实验。测量期间不清理该空scope，不停止共享
服务、不压缩、不构建其他项目或推送。当前轻量交接在本轮结束后再备份。

继续监测同一 raw `results/ieee_tc/p2_backend_qualification/d1021_20261102/`
的 source1_launch.launch/service.log、watchdog.jsonl 与最终 source1.json。
完成后先实际GPU释放，再复用D1019导出流程核查423请求、160服务类与准备
画像、原生选项/捕获记录、真实远端交付及资源；通过才进入一次新FullW2。
不要再次 prepare 已存在的输出、重启该profile或在运行中叠加捕获集合修改。

## 完成：画像通过，进入完整工作负载

23:31 前 source1 与 gated-launch 均成功结束；423/423 请求满足原生计数、
终态和 reservation 释放检查。23:34 使用现有 export_d701 导出，160/160
服务类、96 准备类、415 服务样本和332准备样本全部来自此次测量。
423次HTTP传输、981702550线上字节逐条匹配远端既有journal的ID、归档SHA、
字节和sent终态，无请求时打包。没有重复source1或重新生成输入。

| 画像观测（不是Full性能结果） | D1019 | D1021 |
|---|---:|---:|
| 初始化时间（秒） | 100.196043 | 116.424165 |
| 画像物理GPU-s | 572.980089 | 577.996995 |
| GPU LoRA槽 / 池字节 | 32 / 1825046528 | 32 / 1825046528 |
| 空闲KV block | 4618 | 4269 |
| KV池字节 | 8475901952 | 7835484160 |

初始化增加16.228秒，KV减少349个block，均保留为候选代价；不从后续TTFT或
GPU-s扣除。当前初始化模型回执包含specialize_active_lora=true，参数透传
测试已通过；未记录直接kernel网格/图数量，因此不将官方捕获集合冒称运行中
实测数量，也不凭上述表宣布decode加速。完整4000请求决定端到端收益。

586个资源样本：服务峰值5776695296B、主机最低可用105648058368B、磁盘最低
161315610624B；high/max/OOM/oom_kill/swap、外部compute、本服务逃逸、告警和
中止均无记录。实际allocation release与原生上下文释放成立，两测量scope
均inactive。源画像清理仍有既有未await的HOST预算协程警告；引擎已退出、
工作目录删除和物理释放有独立证据，但不声称该逻辑协程执行成功，不隐藏警告。

远端15:34:03Z只读快照仍为PID1984520、原clock/cgroup、CPU0–39、无限制
memory high/max/swap以及1000/full。资源前快照复用D1019的12:37:47Z历史
记录，覆盖本次运行但不是即时pre-run/连续监控；D1021测前note确认PID/clock/
链路。export上下文明确此局限，不沿用虚构2/4GiB限额，不改健康服务。

完成证据：paper_results/ieee_tc/p2_backend/d1021_3b_active_lora_profile/
completion.json、artifact_resource_context.json及export/。source SHA
14550e0a06401c12cd267beca8cc68da07c8ddb076ebaf88422df160fd70b5d1；
manifest SHA 1fa53f43ffec59d472fbe6fdc070b02ca63d60f9e0564405708cefa2df7281a2。
独立数值正确性仍未知、非主比较性能结果。147历史保护项与用户两个dirty文件
SHA未变，不修改EXECUTION_STATUS.md。下一步严格按D1022登记进行一次完整W2，
不加短Full、不修改捕获集合，不重开基线适配或局部参数扫描。
