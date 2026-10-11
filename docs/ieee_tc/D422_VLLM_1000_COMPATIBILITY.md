# D422：vLLM 7B 原始 1,000 请求兼容检查（预登记）

2026-10-11。16:42:25 已启动，尚未终态。唯一原始结果目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d422_20261011`。

目的：完成计划 §6.7 中原始负载的 1,000 请求兼容检查，然后推进一次完整
4,000 请求 W0。复用 D419 原生路径、D421 全 500-ID 功能证据；不重复已通过的
100/500 检查，不进行参数搜索，不使用功能视图性能选择配置。

复用既有 seed42 trace 的前 1,000 行（`trace_prefix_v1`），不改 adapter 映射，
60 个实际 ID、122,510 目标 token；视图 SHA
`0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981`。
源 trace SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
pool index SHA `e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c`。
不生成新 trace/权重，不复制完整池。

原生配置继续为 vLLM 0.30.0，DP4/TP1、原序号 round-robin、native LoRARequest/LRU，
FP16、max-seqs2、max-loras4、CPU-loras16、rank64、1024 context/token budget，
graph、chunked prefill、无 prefix cache、seed0。配置 SHA
`361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5`。
baseline 当前备份 `d9971a14e18eca3391665d9da1244594e69b6041`；本次不改运行代码。

共同 60 秒部署通知、独立 pre-ready open-loop 到达、1,800 秒资格保护；
服务 72/80 GiB、swap2 GiB，外置回放/监控 4 GiB、swap0，同冻结 CPU 包络。
预计新增峰值 8 GiB，启动至少 150 GiB，运行底线 100 GiB。
per-run LoRA cache 为空；现有原生编译磁盘缓存单独登记 SHA，不预启动 GPU 引擎。
复用 D78/D80 只读发布缓存；不重新发布、按请求打包、业务期间操作远端服务。

raw 中 `run7b.sh` 复用既有 gated-launch/native-http-service；远端启动/关闭、
输入冻结和证据保存只作 campaign/视图参数替换。实际执行 commit 由启动前的
`input_identity.json` 固定，不把后续结果备份 commit 冒充运行版本。

验收：1,000 offered/submitted/terminal/native/source/prompt/adapter/content/timing
合同完整、无 fallback、逐请求 actual==target；原生时间重算误差≤1ms；四 worker
来源/资源读回、真实 HTTP 收据与内容相符、物理 GPU 租约完整关闭、旧保护清单零变化。
使用现有 analyzer：`--scheduler-contract resident_rr_v1 --population prefix1000`。
共同 D394 SLO 仅作此短兼容视图描述；不是完整 W0 排名、U_ref 或跨场景最优。
重复/零权重和独立数值证据的限制保留，原始未知正确性字段不补写。

终态顺序：实际 GPU 释放 → 按本次身份关闭远端 → 保存证据 → 严格分析 → 备份。
任何失败保留；不因短视图联合 SLO 较低而重复。之后一次完整 W0，PrimeLoRA D371
和 ServerlessLLM D414 不重跑；U_ref/W1/W2/G2、3B、其他基线和后续完整矩阵仍开放。

实际 main 执行 commit `9f334caab8a2c305f7e29909532a65f7cc8db1d1` 已推送并核对远端；
baseline commit 如上。本次服务 InvocationID `9d9487f825874614acb5d7a94e0f29a7`，
tmux `tc-d422-7b`。628 文件/54387041 字节原生编译磁盘缓存的当前身份已冻结，
与冷 LoRA 缓存分开。启动前 27 项已有分析 CPU 测试通过；没有新后端代码改动。
