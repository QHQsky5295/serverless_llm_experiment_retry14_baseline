# D430：S-LoRA 共同部署与原生进程生命周期接入

2026-10-11。CPU 接入完成，**不是 GPU 性能或完整 S-LoRA 资格结果**。
继续 D425 历史复用、D428 表示适配和 D429 同口径请求计量；没有改变 IEEE 公式。

## 实现边界与依据

[原生固定版本 API](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/api_server.py)
创建 router 与 detokenizer 两个子进程；
[原生 RPC](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/model_rpc.py)
在 TP1 中直接运行于 router，模型与全部 adapter 初始化完成后返回。
因此 HTTP 父进程就绪/退出不是实际 GPU worker 就绪/释放的充分证据。
本轮联网重查官方固定 RPC，并复核 D425–D429、旧启动器和本机兼容源码；
API 的已核验原始 SHA 未变，不把本地 NCCL/模型等历史适配冒充未修改上游。

复用 `replay_vllm_engine_trace.py` 中的共同 service，而不是另建调度框架。
增加显式 `ieee_tc_slora_native_service_v1` 和 `native-slora-worker`：
outer pre-ready/open-loop HTTP 仍在 Python3.12；原生 API、router、detokenizer
使用已有 Python3.9.25 / PyTorch2.0.1+cu118 环境，不升级或覆盖环境。

原生顺序保持为：共同通知 → ingress 可接收请求 → 完整静态 500 池串行物化
→ 顺序启动各副本并执行其完整 HOST 加载。业务按固定通知后 60 秒到达，
不等全部 ready，不使用未来请求决定预加载集合。CPU-only 物化阶段不虚构
GPU-s；各卡在分配给对应 worker 前开始计量。D428 的 packed HOST 表示显式
启用，既有 paging、batching、kernel、adapter 选择和 round-robin 不变。
原生公共参数显式记录，当前接入支持原有 TP1、swap、非 BMM/slora 路径；
尚未据此宣布冻结性能工作点或支持所有模型配置。

## 原生证据与释放

source-view 新增可选 `--worker-lifecycle`，默认关闭：
原 RPC 在 NCCL 初始化前重新核对实际资源域/源码/环境；在原有全池初始化和
random seed 设置完成后记录实际 worker。证据包括 PID/出生身份、API 父进程、
可见设备、模型、完整 500-ID→native-index 映射、实际 InputParams、源码/
配置 SHA 和原生环境。没有额外 warm inference 或 CUDA 同步。

父服务核对这些字段、实际 NVML compute PID 和物理 owner，然后才开放选中
worker 的队列；health 只是另一必要条件，不能单独放行。请求的 native PID
还必须匹配该 worker 的 receipt。声明配置不代替实际进程证据。

原 API 的子进程启动到 uvicorn 返回范围，显式包在 owned-handle finally
中；退出只 terminate/join 已创建的 router/detokenizer，必要时 kill 同一
持有句柄，不按端口/进程名清理。仅 TP1，不能宣称覆盖其 TP 多级树。
**子进程退出仍不是 GPU release**：复用外层 pidfd 和新鲜 native census，
缺证据继续保留不完整/右删失。正常服务路径的原语句顺序未改变。

## 验证结果及限制

- 最终相关全套 185 项 CPU 检查通过，9.769s；原生 Python3.9 的 38 项
  通过，1.877s。D429 主仓库 236 项证据复用，主仓库计量代码本轮未改。
- 执行实际 API main 的 AST，验证原正常返回、启动异常和服务异常都会清理
  原句柄；另用两个真实 CPU 子进程验证 terminate/join。没有实际 GPU 子进程。
- bootstrap 生成的 argv 交给原生 argparse，逐字段匹配，含全部 500 路径；
  检查源码/环境/设备不匹配时拒绝启动，没有无约束 fallback。
- 真正执行共同 outer service、loopback HTTP、pre-ready 和原生 SSE fixture，
  验证“先接收到达，再完成全池准备，再分配 GPU/启动/确认/退出/释放”的顺序。
  该测试的 GPU/cgroup/owner 明确为合成，不能替代真实 containment/lease 验证。
- 最终 source view 的 API/RPC、实际 guard 在 native3.9 环境可导入，
  CUDA 隐藏且未初始化；没有实例化模型、测 pinned RSS 或运行 inference。
- 首次 49 项有一项 fixture 错误：MagicMock 无法作为源码文件身份被 inspect。
  改为真实 callable fixture，生产身份检查不变；失败日志保留。
- 九个辅助 scope 顺序执行，最大观测 peak 506,540,032 B，high/max/OOM/swap
  全为零；逐一核对身份及空进程后停止。147 旧结果保护项零变化，三个用户
  dirty 文件和原生十文件历史补丁未改。新鲜 GPU census 为空；
  磁盘可用 163,363,241,984 B，不把该值当作后续启动保证。

## 证据与下一步

Raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d430_20261011/`。

- source-view manifest：
  `51c6f1da21cc5070a7d71e567c10564d3066605546602705c0b1c5d8b9277004`
- 185 项最终日志：
  `38cf8760338c1f77e164e0dd1eb40466a3e2fb32b6a9a711d641f38faeda28e6`
- native3.9 最终日志：
  `376f77ca032212db8ef49eb8b78e0ac1ed38fcc6c97aeb14a53eeb48c2161e15`

下一步：备份 → 给现有 `audit_d402_serverlessllm_path.py` 增加明确 S-LoRA
来源/全池传输/native/physical join → 重新读取计划及保护/资源/输入门槛 →
冻结合法原生公共配置 → 实际 containment/release、受影响 100 请求 →
缺失 500-ID、1,000 请求 → 一次完整 W0。不能用本轮 CPU 完成替代 GPU 资格。

已封存 PrimeLoRA/ServerlessLLM/vLLM W0 不重跑，不跳 3B、不更改评价尺子。
其他 7B baseline、W1/W2/G2、3B、A1–A5、S1–S13、CE、文档和最终图表仍 OPEN。
没有新工件/负载、清理、远端操作/缓存重发布、安装、图表或论文修改。
