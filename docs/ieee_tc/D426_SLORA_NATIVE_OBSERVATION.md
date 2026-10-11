# D426：补齐 S-LoRA 公平计时的观测组件

2026-10-11，完成 CPU/source-view 里程碑；没有新 GPU 性能实验。

D425 证明既有 S-LoRA 完整结果缺少新协议必需的原生首末 token 与完整物理
GPU 占用。D426 在已有 baseline 计量组件中补原生观测，不改变其推理算法：

1. 实际批次注册前保存不可变 prompt 和 adapter 绑定，避免 input list 被
   native decode 追加后污染原始输入证据。
2. 在原采样 token vector 已返回 CPU、文本/HTTP 之前记录时间；不新增 GPU
   同步，不把 SSE 收到时间当 token 产生时间，也不把完成通知尾部算进 TPOT。
3. 通过原生 detokenizer/FIFO 和 API 传递请求级事件；拒绝缺失、乱序、重复、
   adapter/时钟/输入/长度不一致。单 token TPOT 保持 N/A。
4. 原 S-LoRA RPC 注册并非 vLLM queue entry，使用独立 schema，不伪造同名
   边界。原数值资格未知状态不由这些字段自动提升。

source-view adapter 校验 13 份代码 SHA，只在两份复制文件中加观测；其余
使用已有源码。原嵌套仓库及 10 份兼容修改不变。移除精确观测语句后 AST
与原源码完全一致。统一池的权重副本与 staging tensor 分开核对，避免合法
释放 staging 被错误判为 adapter 未加载。

验证结果：原 Python 3.9.25 环境 15 项新检查通过（0.198s）；共用启动、
计量、router、vLLM 和固定输出回归共 142 项通过（8.177s）。实际 native
batch init/forward/filter/merge、detokenizer/FIFO/API 方法在 CPU fixtures
下执行；不是 CUDA 数值证明。原环境 RPC/API 真实导入也通过，CUDA 未初始化。
第一次 discovery 的测试命名空间导入错误及全部成功日志均保留。

全部检查受 4 GiB、零 swap、CPU 2,3,26,27 保护；没有模型加载、远端操作、
新 trace/pool、重复发布交付缓存或性能重跑。147 项历史保护核对零变化，
三个用户修改 SHA、完整计划及 V1/V2 协议 SHA 不变。遵循用户最新图表节奏，
此阶段保存数据与结论，不制作性能图。

详细源码、原始链接和验收边界见 baseline 仓库：
`docs/ieee_tc/D426_SLORA_NATIVE_OBSERVATION.md`。
原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d426_20261011`。
source-view manifest SHA：
`bf373f58e52538d20884100798d0a5e3439d5a73cbfe2bfabe6bd6638996fd18`。
142 项成功日志 SHA：
`b968411d50a1defc12e614d30a7e12692d67aa80435bc90bf1838486ed1bd960`。
日志（含失败的 discovery）、source-view manifest、两份观测源码及辅助域
退出回执已封装为 `paper_results/ieee_tc/baseline_audit/d426_slora_native_observation.tar.gz`，
SHA `8d6e49e310935d347233ed7acf34f451e5d5ce5d15f290178ec6be6ef9334c6e`。
7 个本轮辅助域均核对实际身份、4 GiB/零 swap 限制和全部空进程列表后关闭；
未停止其他进程，未删除任何工件/日志。所有域 high/max/OOM 事件为零。

Baseline 测试代码及报告备份：`b694915d920bd61200b32614f7e60d88a23a5c08`，
已推送 `origin/main` 并核对远端 SHA。这是组件代码检查点，不是性能运行 commit。

下一项是将观测接入共同业务到达和真实资源计量，保留 S-LoRA 原全池 HOST
准备，但把准备放回共同部署通知及资源窗口。随后才进入必要 GPU 路径资格
和完整 4,000 请求对比。现在没有新的 S-LoRA 性能排名或 500-ID 执行证明。
已封存 PrimeLoRA/ServerlessLLM/vLLM 点不重跑；其余 7B 基线、W1/W2/G2、3B、
消融、机制与敏感性、成本补充及最终图文仍未完成，完整目标继续执行。
