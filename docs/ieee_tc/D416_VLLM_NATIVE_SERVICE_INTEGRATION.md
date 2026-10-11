# D416：vLLM 共同测量入口已接入，尚待 GPU 资格

2026-10-11。当前是实现与 CPU 检查完成，不是新的性能点或 Resident 预算参考。
完整说明在 baseline 仓库 `docs/ieee_tc/D416_VLLM_NATIVE_SERVICE_INTEGRATION.md`。

## 本轮实际进展

既有 `replay_vllm_engine_trace.py` 新增显式原生服务模式，复用 D415 原生
token 快照、共同 HTTP 请求发生器、发布工件解析器、准备期入口、外置安全
监控与物理 GPU 账本。没有复制 PrimeLoRA 调度器，也没有新建负载或工件池。

原请求按固定时刻到达，目标按请求序号轮询；选定实例未就绪就排队，不因
其他实例已就绪而偷换目标。真实远端读取、传输、本地物化和生成均归服务
资源域。主计时使用 native token 事件，完成通知尾部不进入 TPOT。
每卡独立物理所有权在进程创建前取得，进程与 GPU 上下文真正退出后才返回。

旧 D395 工作点核对为 DP4/TP1、maxseqs2、GPU LoRA4、CPU LoRA16、FP16、
GPU fraction0.7、token/context1024、chunked prefill、非 eager、无 prefix cache。
新部署直接使用原生 LoRARequest/缓存，不继续使用旧外置 API 注册 LRU；
这是公开的接入差异，不将新旧部署冒称完全相同。待 GPU 验证，不宣称已复现成功。
旧用户修改的 replay、relayserve 文件未修改；D395/D396 全部结果保留。

## 证据与限制

- 新增 10 项服务合同检查与已有测量/启动/路由/SSE 合计 122 项 CPU 检查通过。
  覆盖准备期排队、轮询、取消、失败不回退、原生时间及实际 loopback HTTP 校验。
- 共同物理 GPU 生命周期已有 32 项测试通过；真实四卡释放尚未验证。
- 实际 vLLM0.30.0 环境在隐藏 GPU 的条件下接受同一 AsyncEngineArgs 和
  cumulative SamplingParams；没有构造 engine 或加载模型。
- 已联网核查 [固定版本 AsyncLLM 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)
  和 [LoRA 文档](https://docs.vllm.ai/en/v0.30.0/features/lora/)，没有把通用
  技能中的阈值或性能承诺写入冻结协议。

这些检查不是完整 workload、数值 LoRA 正确性、GPU 速度或资源节省的证据。
D394 SLO、单轮规则、95%/75% 和共同 60 秒准备保持原样；U_ref 仍未关闭。

下一步：新执行键上的原始 100 请求受影响路径资格，检查实际原生事件、
远端 bytes/content、四卡物理释放与 CPU/内存归属；审计可复用后端资格后
进入一轮完整 W0。PrimeLoRA D371 和 ServerlessLLM D414 已封存，不重跑；
不跳到 3B 或展开局部参数搜索。D78/D80 一次性交付缓存直接复用，不重建。
