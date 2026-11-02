# D1019：固定并发下的 GPU LoRA 驻留容量候选

2026-11-02。接续已闭合的 D1017/D1018；不是重做基线适配，也不是新的
评价协议。当前目标文件为
`/home/qhq/.codex/attachments/7816a1ea-77dc-4e47-bb42-a6aef482c7ac/goal-objective.md`。
本轮完整读取 PLAN、冻结 V1/单轮 V2；PLAN SHA 保持
`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。

## 主线与停止条件

D810 已有 3B 五个外部主基线 W0/W1/W2 和 dLoRA W1/W2 的完整观测；
不因为低 SLO 重开基线参数搜索。用户再次确认共享 k3s 非本项目专用，
沿用 D570 的 rootless Podman 隔离适配，不读取管理员 kubeconfig、不修改
共享集群。HydraServe 已有完整结果，不能将它重新列为基础适配阻塞。

此前超过 95% 的 D752 是 W0，不能替代当前版本的 churn/W2。
当前 D1017 W2 为 3738/4000=93.45%、8084.812476 GPU-s，仍差 62 个
联合达标请求。全量独立数值正确性仍未知，`n_correct=null`；原生生成/
绑定/计量检查通过不扩写为独立数值证明。G1/G2、95% 和75%均不改变。
联合 SLO 是必须展示的结果，不是拒收低分基线的规则；仅按 SLO 排名又会
遗漏资源效率贡献。当前继续同一个 3B 主线，而非再切回 7B 自我比较。

本候选仅一次必要画像测量，若完成则进入一次完整4000请求W2；不另加
重复Full前缀、不扫描8/16/24/32网格、不按胜负重跑同键。完整结果决定
接受或否决，并明确同版本其他场景与外部对照缺口；达到目标后进入完整
比较和机制实验，不把局部改善当作无限优化理由。

## 新证据与版本依据

D1018 在 D1017 的6个原生owner中发现：1091个重复请求的 demand reuse
distance 为8–31，其中969次实际 native load、48个load且联合违约。
该统计没有主动准备、pinning和实际LRU顺序，不能当作缓存命中模拟，也
不能推算足够补齐62个请求。它只支持核查容量与并发不必数值相同的假设。

既有 D343 是7B并发和槽数一起增加的对照；D710已被D711撤回。本轮搜索
3B配置未发现16/24/32/64槽的既有候选。不重复D998延后、D1005–D1007
扩缩容、D1014文件优先、ISA-L、全图元数据或GC等已关闭方向。

[vLLM 0.30.0管理器](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
分别管理CPU注册容量与GPU active槽；后者由max_loras控制，并不要求与
max_num_seqs相等。
[同版本配置](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py)
将max_loras纳入计算图身份，因此增加槽数也可能增加图/内核开销或挤占KV。
本机已适配源码SHA分别为
`6695bb7d6373d8f29bb13d6ae31e2ef2a98f8c23a6f37f9147d67cafca5f3302`、
`ed6cf6575f5950f079a62f904caeae93d1a4745dd8c21788b74367883bcd4b08`；
在线官方源码用于语义核对，不宣称与本机字节完全相同。

## 物理容量依据与唯一候选

复用D725完整原生测量 `source1.json`，SHA
`19f91a2302fa1aae3c33c01d1005983f43103ff8eb3ae1b5bf28812a51dc2fc6`。
启动前后均观测到：

| 量 | 原生回执 |
|---|---:|
| GPU LoRA槽 | 8 |
| pool_allocated_bytes | 456261632 |
| uniform_slot_layout | true |
| 每槽容量 | 57032704 B（54.390625 MiB） |
| 空闲KV block | 5002 |
| KV block bytes / tokens | 1835008 B / 16 |
| KV池字节 | 9180545024 |

唯一候选：max_loras=32，等于已经存在的max_cpu_loras=32上限。保持
max_num_seqs/runtime_concurrency_cap=8、TP1、FP16、rank16、1024上下文、
4096 iteration tokens、0.72显存比例、原图覆盖、HOST预算及控制规律。
没有提高请求并发，没有改九公式或共同SLO，没有使用未来热点。

均匀布局的静态估算为32槽1825046528B，比原来增加1368784896B；若只从
KV减去这部分，相当于746个block、剩4256，仍高于8条最大上下文需要的
512个block。这只是启动可行性的估算，不是新配置的实测空闲量或保证。
新运行必须保留实际pool/KV/图配置和资源回执，不能把这个估算填作实测。

## 实现、复用与必要检查

只扩展既有prepare_d701、run_d298和完成长度初始化的显式入口；核心路由、
驻留选择、admission、扩缩容及服务实现未改。新配置与D724只有max_loras
差异；后续Full以D1017控制为父点。保留D747逐view新鲜描述库和stdlib gzip，
不误用D877分支中的ISA-L。原生版本仍为既有vLLM0.30.0，不升级环境。

24个原adapter、76个波次、423个请求引用与D725一致，不新增权重/负载。
仅复用旧原生完成的固定输出长度初始化；不能转移旧时延、SLO、数值资格。
新增graph-slot完成长度绑定拒绝改变并发、CPU容量、生成合同、未声明差异
及不完整原生计数；原有graph-only规则仍拒绝槽变化。新model_backend_id：
`native-model/9e75aa074c3f0d3a81dca0d8ffd22b321e6dbc03ef13d1c340abe59f1a0e75d3`。

20项既有/新增完成长度测试通过；bash语法、diff检查、147项历史保护通过。
准备入口实际通过24身份/76波次/423请求检查。源画像文件已独占创建，
未修改D725旧文件。两个用户dirty文件的SHA不变、不加入提交。
CPU测试与静态准备运行于3/4GiB、swap0、CPU2/3/26/27；结束已退出，未保存
结束前峰值快照，故不虚构峰值或零资源事件。只读检查曾误用系统Python的
hashlib.file_digest和缺numpy客户端，改用兼容读法/既有环境后通过；不计作
GPU候选失败，也没有据此更改任何测量结果。

输入产物：`paper_results/ieee_tc/p2_backend/d1019_3b_gpu_slot_profile/`。
config SHA：`65e43028b746f4c1ad9a188a92c9f2a5a5b1d1df62aadc7a4c6979a4b7a2ddb1`；
source_spec SHA：`976d8dad5018d28944e4a2c50ff2e76294480d40844786f030ca559d238c0de6`。

## 原生测量入口与保护

在既有tmux中执行 `bash scripts/ieee_tc/run_d298_capacity_source.sh d1019`。
该入口仍使用原gated-launch、服务72/80GiB、辅助3/4GiB、外置watchdog和
原CPU分区；新建 `results/ieee_tc/p2_backend_qualification/d1019_20261102/`，
旧输出存在则拒绝覆盖。测量期间不压缩、推送、构建其他项目或操作远端。
完成后严格exporter检查原生423请求、160服务类、准备类、真实交付和释放。
若通过，不再加一个无问题目标的短Full，直接准备完整W2的新执行键。

准备时GPU compute为空，推理机可用约150.58GiB，MemAvailable约104GiB；
原150/100GiB及防OOM门槛保持。远端3B认证health通过，已发布gzip、
artifact_timing_v2、原clock_id不变；PID1984520仍健康，用户unit inactive
不意味着HTTP服务停止，因此不重启。远端eno1为1000/full、可用约123.69GiB；
采用既有制品节点独立门槛，不打包或复制工件。没有新远端CPU监控，不能
声称整个测量期间不存在外部干扰。共享k3s和7B服务均不操作。

进一步只读核查发现该远端进程位于`session-8529.scope`，affinity为0–39，
可读祖先/叶组memory.high/max/swap.max均为max；这不是历史exporter中
硬编码的2/4GiB、推理机CPU分区。不能把旧资源声明继续写成这次实测。
远端源码当前磁盘SHA为
`a365072244512f4880432d7f4198cf3e45897ff19063a2b25eb141c8ca4e2a02`，
不冒充已验证进程加载字节；既有日志为`tc/d80_20260927/3b/server.log`。
本轮不为修正记录而移动/重启健康服务。新画像导出前须用明确的实际资源
回执替代旧exporter的历史默认声明，记录监控缺口；旧profile原字节保留。

本文件登记候选与静态检查，不宣称画像或Full测量已经完成。后续结果追加，
所有失败保留。7B/3B最终同版本主比较、G1/G2、A1–A5、S1–S13、数值正确性
范围、审稿文档、图表和最终V2集成全部仍开放；接近目标前继续不绘图。
