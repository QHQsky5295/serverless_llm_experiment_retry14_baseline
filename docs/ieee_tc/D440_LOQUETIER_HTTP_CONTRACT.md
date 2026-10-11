# D440：Loquetier 公共请求与物理计量合同

日期：2026-10-12。状态：HTTP 接口和计量边界的 CPU 检查通过；实际原生
worker、服务启动及 GPU 请求资格仍未完成。不是新的性能点，也不是完整复现。
接续 D437 静态分片、D438 容量和 D439 在线原生边界，不重复已通过的容量实验。

## 依据和范围

本轮复查完整计划、执行状态、冻结指标 V1、单轮 V2、最新基线保真规则及
D439 修改历史，并核对 [Loquetier 官方仓库](https://github.com/NJUDeepEngine/Loquetier)。
官方核心仍为虚拟模块及 SMLM；本轮只接入共同实验请求和计量协议，不改
原生计算、精度、批上限、四分片映射或缓存策略，不进行性能参数搜索。
网络页面用于核查系统定位；具体原生代码依据此前固定本地 checkout，
不声称本轮网络页面验证了每个本地源文件的字节。

用户再次确认一次性只读交付缓存授权。继续复用 D78/D80，未重新发布，
未复制解压池。正式请求中的必要读取、传输、竞争和本地物化仍需计量，
离线打包不计入服务路径；本轮没有访问或调整远端服务。

## 必要适配

复用 `replay_vllm_engine_trace.py`、`prepare_ieee_tc_serverless_stack.py`
和既有 `PhysicalGPUDeployment`，没有新建实验框架。

1. `NativeLoquetierRequestHandler` 将共同请求送到 D439 原生在线队列。
   逐条检查静态分片、已准备 adapter 路径/索引、模型、原生输入 ID、
   greedy 固定输出合同、进程和 ready 收据。返回数量必须是原生整数计数，
   布尔值不能冒充索引或单 token 的计数。失败/取消不生成成功、不重试。
2. 原生注册、调度、首末 token、worker 完成和客户端完成保留各自边界。
   `ieee_tc_loquetier_native_v1` 和 `loquetier_native_integer_token_id`
   是明确的新配对，不把 Loquetier 包装成 vLLM 或 S-LoRA。
3. 共同 HTTP 发布器增加显式 Loquetier schema、静态分片生命周期和
   pre-ready 入口合同。首末 token 用原生整数可见事件，HTTP 通知尾部只
   进入 E2E，不算进 TPOT；单 token 的 TPOT 为 N/A。
4. 复用既有整池顺序 materialization 函数（保留原函数名，新增显式 backend）。
   Loquetier 使用独立 `loquetier-full-pool:` 身份；完整 500 个静态 ID，
   必须在共同通知后、原生 ready 前准备，不读取未来热点。
   S-LoRA 的默认行为不变。每请求记录准备引用，不重复计作一次传输。
5. 公共物理计量只增加上述合法 token/schema 配对，原 allocation/release
   并集算法不变。未知的 `n_correct` 和数值正确性状态不补造成已通过。

## 验证及其证据边界

- 185 项接口/服务/公共回归通过，10.093 秒：11 项新 Loquetier HTTP 检查、
  11 项既有在线服务检查、163 项 ServerlessLLM/vLLM/S-LoRA 公共回归。
  HTTP 使用真实 loopback 传输，但 token producer 明确为 synthetic；
  加入通知延后，核验其不会改变原生 TPOT。不是 GPU 模型算术证据。
- 既有 Loquetier 环境中 26 项原生 CPU/在线边界检查通过，2.363 秒，无 skip。
  沿用此前实际原生 KV/停止边界检查，不是新增 GPU 吞吐测量。
- 物理生命周期 36 项检查通过，1.417 秒，包括 Loquetier 配对和错误跨后端
  配对拒绝。测试中的合成 lease 不是本次真实 GPU 释放证据。
- 本轮较早 21 项检查通过（0.441 秒）；补整数类型负面用例后运行上述
  完整回归。均为开发单元检查，不是重复正式性能运行或择优。
- 所有检查 GPU 隐藏，辅助 CPU 2,3,26,27，high/max=3/4 GiB、swap=0。
  最终公共/原生/生命周期测试峰值分别为 379109376、254976000、28749824 B；
  high/max/OOM 为 0。各资源域实际进程树为空并按 InvocationID 收尾。
- 三个用户修改文件、Loquetier 三个和 S-LoRA 十个历史补丁 SHA 保持不变。
  旧 147 项保护清单核查另存 main 仓库，不修改旧封存证据。

## 下一步和未完成项

接入真实原生 worker 的受限初始化、原生请求构造、启动 registry/进程收据、
致命错误退出，以及共同服务的静态分片启动、健康核验和真实 GPU 所有权。
目前仅新 HTTP handler/发布器/校验器就绪，**CLI 和外层服务尚不接受 Loquetier
部署**。不得凭本轮 CPU 检查启动完整性能回放或声称已经端到端接通。

新启动须重新生成小型 source-view manifest，引用本轮 source adapter 身份；
旧 D438 manifest 不用于新源码，也不被覆盖。然后依既定顺序进行 100 请求、
完整 500-ID 功能覆盖、原 1000 兼容、单轮 4000 回放，并取得自身数值证据。

其他 7B 基线、HydraServe、ElasticLocality、W1/W2/G2、必要 PrimeLoRA 优化、
3B、A1–A5、S1–S13、两轮审稿文档和后续图表仍 OPEN；不缩减完整计划。
