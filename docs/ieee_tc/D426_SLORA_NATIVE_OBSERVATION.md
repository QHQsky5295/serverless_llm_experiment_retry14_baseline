# D426：S-LoRA 原生 token 观测组件

2026-10-11。CPU/source-view 里程碑，不是新 GPU 性能结果或完整资格。
接续 D425 的 25 份历史回放审计，不重复已封存的其他系统 W0。

## 实现范围

复用 `ieee_tc_serverless_measurement.py` 和现有 source-view adapter，新增显式
`prepare-slora-measurement-view`。校验当前 13 份原生源码 SHA，仅复制并插入
`model_rpc.py`、`api_server.py` 两文件的观测语句；其余包文件使用符号链接。
原嵌套仓库、10 份历史兼容修改、已安装环境、用户 replay/relayserve 均不修改。
现有 ServerlessLLM/vLLM 路径不启用 S-LoRA 观测。

在 native RPC 注册时保存原始 prompt IDs、目标长度和 adapter 对象索引；
`InferBatch` 会复用并追加原 input list，所以不能从完成后的 list 反推 prompt。
首次真实 forward 前记录调度边界、实际 LoRA engine 和选中 adapter。
沿用原采样结果 `.detach().cpu().numpy()` 后记录本批 token vector 可用时刻；
不新增 CUDA 同步，不冒称逐 kernel 结束时刻。首 token 携带初始绑定，其余
token 只携带请求/adapter/index/native token ID/time，避免逐 token 重复 prompt。

元数据由原生 request 对象持有，filter/merge/free 决定其生命周期；没有新的
全局请求缓存。原 detokenizer/FIFO 传递元数据，API 仅增加一个 SSE 字段。
consumer 拒绝缺失、乱序、重复、错误 adapter/clock/prompt/token、过早结束和
未完成结果；完成通知尾部不混入 TPOT，单 token TPOT 为 N/A。

S-LoRA 的 RPC 注册边界不是 vLLM queue entry。因此使用独立
`ieee_tc_slora_native_v1` schema，不伪造 vLLM queue 时间或物理资源证据。
支持范围限定 TP=1、真实 `slora` 调度、固定长度 greedy；dummy/base-only/
no-kernel/no-LoRA-compute/其他 scheduler 均拒绝。BMM 与统一池分别核对
原生存储归属，不以 staging adapter 的 `is_on_gpu()` 误判统一池内容。
本组件不是独立数值正确性证明；原 `lora_numerical_correctness_qualified=false`
语义保留。实际池、源码、进程和 GPU 生命周期仍需下一阶段联结。

## 原始依据及检查

已读 D425 来源审计和共用组件 D415/D416 修改史，并联网核查固定官方版本的
[model RPC](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/model_rpc.py)、
[批次对象](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/infer_batch.py)、
[统一池加载](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/server/router/model_infer/infer_adapter.py)。
加载到共享池后释放 staging tensors 是原实现行为，不是本轮策略变化。

- 新增 15 项 CPU 检查；S-LoRA 原 Python 3.9.25 环境全部通过，0.198s。
- 执行实际 native init/forward/filter/merge 方法，只有 tensor/CUDA 边界使用
  明确的 CPU fixtures；验证 prompt alias、分批合并、请求独立计数。
- 执行实际 detokenizer、FIFO、SSE Python 方法，socket/tokenizer 使用 fixtures；
  验证生产者批量输出后慢消费者不丢原生计时或 token。
- 去掉精确插入的观测语句后，两份文件 AST 与当前原源码完全一致。
- 共用组件、启动、router、vLLM、固定长度/FIFO 回归共 142 项通过，8.177s。
- 第一次广义 discover 运行 131 项，其中一个既有测试导入失败：`tests` 命名空间
  未绑定到 baseline 包。按项目模块全名启动后通过；没有改代码规避该失败。
  初次 14 项组件检查、失败 discovery 及后续成功日志全部保留。
- 创建独占实际 source view 后，在原 Python 3.9.25/PyTorch 2.0.1+cu118 下导入
  native RPC/API 和观测 helper，实际路径匹配且 CUDA 未初始化。没有运行模型。
- 所有检查限制 4 GiB、零 swap、CPU 2,3,26,27，GPU 隐藏；没有重型实验、
  远端操作、新负载/工件池、安装或论文修改。

原始证据目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d426_20261011`。
source-view manifest SHA：
`bf373f58e52538d20884100798d0a5e3439d5a73cbfe2bfabe6bd6638996fd18`。
成功 142 项日志 SHA：
`b968411d50a1defc12e614d30a7e12692d67aa80435bc90bf1838486ed1bd960`。
原 Python 3.9 检查日志 SHA：
`1bc36e853f120afa3cf4c23afa32848a604a135acd0e77fdb453ac1fdd582fd7`。

## 下一项主线，不提前宣称资格

将已测观测组件接到既有共同 pre-ready/open-loop 入口和 physical-owner
计量中，补明确的 S-LoRA validator 分支；保留原全池 HOST 准备、普通既定
路由、60 秒部署通知和真实远程发布工件。全池准备需从通知后计量，不能
硬套按请求 fetch span 或把原 wrapper ready 后回放当共同准备协议。
Python 3.9 的 native 进程不直接调用共用 helper 中 Python 3.11 的
`asyncio.timeout` 入口；外层已有 Python 3.12 计量服务可复用。

先验证服务进程归属、准备资源与内存增长，再进行原 100 请求受影响路径、
缺失 500-ID/1000 资格和一次完整 W0。尚无新 S-LoRA G1/G2 点，不调阈值或
加入 PrimeLoRA 策略。PrimeLoRA/ServerlessLLM/vLLM 封存结果保持不变。
