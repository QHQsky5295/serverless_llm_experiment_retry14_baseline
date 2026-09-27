# P2：共同 vLLM 后端资格（进行中）

已通过两模型各 100 请求的顺序合同及普通并发检查；取消分支见以下独立证据。
完整模型/池资格与性能资格仍未完成。
旧环境与旧结果不覆盖。

## 2026-09-26：7B 真实 GPU 所有权和退出资格（D27）

在原四请求前缀上，独立 runtime 已接入物理 UUID 分配、真实 worker 核验、
退出事件及资源归还。三次检查均为4/4目标输出、551原生 tokens，逐请求的
adapter/prompt/output SHA 在三次间一致；不代表全池或独立数值正确性已通过。

| 检查 | 物理占用 | 最后 token 后仍持卡 | 结果 |
|---|---:|---:|---|
| 第一次 | 截断，未确认归还 | N/A | 退出时仍有 GPU 上下文，失败保留 |
| 第二次 | 102.164 GPU-s | 48.769s | 计量闭合，但最终强制退出 |
| 第三次 | 60.738 GPU-s | 7.642s | 正常退出，原生上下文清除，归还确认 |

修正来自实际退出依赖：控制连接必须先关闭，才等待 Python3.12 服务结束；
同时等待按出生身份固定的原生 worker 退出事件。没有调短等待参数、打 sleep
补丁或用请求完成时刻假冒资源释放。60秒清理预算和安全保护保持不变。

该结果只资格化单卡7B独立进程边界，不是 M1/M2，也未验证 Full 多 runtime、
共享引擎、TP、其他 baseline 或远程500池。完整记录和状态表见
`PHYSICAL_GPU_MEASUREMENT.md`，数据见
`paper_results/ieee_tc/p2_backend/20260926_7b_physical_lifecycle.{json,csv}`。
666项功能回归、56项安全/回放检查通过；旧投稿数据不变。

## 2026-09-26：7B 实际 source admission / D+T+O 资格

复用原前 32 请求；单 worker 顺序执行，调用 D25 的实际接纳/准备方法与原生
token observer。没有构造默认延迟或假 profile；这次不运行 Full 路由/规划。
首次原生 miss 先 load/release，16 条 priming 的回执均保留且位于测量接纳前，
因此不能把这一诊断说成真实 remote 主实验或端到端成本收益。

| 检查 | 结果 | 结论边界 |
|---|---|---|
| 固定输出 | 32/32，5,967 native tokens | 机械合同通过，不替代数值正确性 |
| 接纳源 | GPU 28，HOST 4 | HOST 由原生 LRU 自然产生，无强制驱逐 |
| GPU D | 28 次全部 0；可执行引用先于接纳 | 不是在 resolve 后追认命中 |
| HOST D | 93.910 / 103.719 / 108.989 / 118.300 ms | 含实际控制/加载路径，不称纯 H2D 时间或稳定均值 |
| 原生事件、分解、TPOT | 32 条一致，恒等式和重算最大误差均为 0 ms | 同一实际 monotonic 时间域 |
| 原始 prompt / native input IDs | 32/32 与历史前缀一致 | 无重新生成输入 |
| 历史 native output SHA | 31/32 一致；req_00005 不同 | 原样保留，原因未确立，语义/数值资格仍开放 |
| 引用与退出 | 每条 GPU/HOST 引用归还、cache 清空、GPU contexts 清空、scope 移除 | 实际整卡生命周期计量仍另行接入 |
| 资源 | 256 次采样，峰值 4,743,290,880 bytes；high/max/OOM 全零 | 不外推完整负载峰值 |

req_00005 为既有 legal_lora，同 prompt、目标和原生 input IDs；当前输出 SHA
`73983ecc61e7dbd6bcb38eeca91430ceeca2fe5f4fbe2a3c1c67d937662d7dd3`，
历史为 `a00a9aad882e6d3c22387dcafe7d7fa10d2df7bf54541bb5507cb5ce630d8272`。
不能仅因长度正确而宣称语义验证全部通过，也不能未证明就归咎于某个数值 kernel。
保留已有全池/数值资格问题，不重复没有判别力的零权重同提示对照。

本次初始 JIT 与串行 admitted=1、仅 GPU/HOST 的覆盖均记录。当前原始区间
不是完整冻结生产 profile，不据此设 SLO，不生成系统优越性图。按计划 11.2
交付状态表及逐请求 CSV/JSON：`paper_results/ieee_tc/p2_backend/20260926_7b_source32.*`。
原始 SHA `28e01325a1aefd272615cdb47993eaa27bf1c3c87fb601b12c6d461989382614`。
新增三项测试后的完整功能回归 653 项通过（22.819 s），安全/census/replay
56 项通过（.648 s），无跳过或失败。

下一步沿主线接通物理容量/生命周期并补齐真实服务类初始化和集成资格，不把
这次测量包装为 Full、远端 174、M1/M2 或新一轮性能胜负证据。

## 2026-09-26：实际跨进程取消（7B）

假设：旧 proxy 把单连接取消标为整个 engine dead，会阻止已知请求的
retirement 确认，并可能使其他同批请求所在副本被提前回收。
官方 [AsyncLLM 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)
区分 frontend abort 与 core 请求生命周期，不能由 TCP 关闭推断已释放 GPU 工作。

改动：控制交换使用独立短连接；丢失回复的 mutation 按 owner/lease 留存，
不重发生成，不标记进程已死。存在未知 mutation 时拒绝新的 generation；
只有匹配 owner/lease 的原生 retirement 才清除相应 generation/retirement
不确定记录。其他未知 load/acquire/release 不被顺带清除。已在执行的其他
请求继续完成。真正进程退出/原生 fatal 仍按失败处理。

| 实际 7B 检查 | 结果 |
|---|---|
| 既有前四请求、两组并发 | shared finance；writing/finance |
| 取消时 native batch/KV | 两组均观测到共同 decode |
| 取消请求 | 2，未计为正确 fixed-work 请求 |
| 保留请求 | 2，59/217 native tokens，输出与旧对照一致 |
| 取消后资源归属 | 只剩 survivor 的一个 lease；最终全部归还 |
| 结束后的 proxy 状态 | 无未知 mutation，engine 未被误标死亡 |
| E2E / TPOT 重算误差 | 0 / 0 ms |
| 服务内存峰值 / high/max/OOM | 5,683,507,200 bytes / 全零 |
| 外置采样 / 最终释放 | 62 次；GPU contexts 清空，scope 移除 |

数据：`paper_results/ieee_tc/p2_backend/20260926_7b_cancelrpc4.{json,csv}`。
raw SHA `fc4c9cec7603fe2bb58d17f4377e50a5fe428e5af2d71d343a0b58b394ef8989`。
这是实际 dedicated worker/TCP/native core 的资格诊断，不是完整 controller
工作流、真实远端性能、G1/G2 或非零 deferred-KV fence 的证据。20 ms 诊断
采样与独立控制连接开销不能当作正式低扰动监控成本。

本次正常 shutdown 按旧路径删除了 worker 私有文字日志；结构化 native
观测、请求结果和外置 launch/watchdog 原始证据均保留。后续资格启动启用
现有 keep-worker-logs 开关，不为补文字日志重跑已完成的正确检查。
3B 输出差异仍须 stock-native reference 归因；不因 7B 通过而宣布两模型通过。

## 3B stock-native 输出归因：部分诊断完成

`native_cancel_reference` 直接调用 stock `vllm.v1.engine.async_llm.AsyncLLM`，
未调用 Prime 的 demand-load/reference/retirement。保留只读 worker 和 scheduler
观测。req_00003 的原生取消同伴后输出 SHA 为 `f6b412...0039`，与 Prime
取消路径完全一致；随后原生顺序同提示同 adapter 输出为 `2c2300...148c`，
与旧顺序对照一致。因此 Prime 引用/卸载实现不是这个输出变化的必要条件。
这是所测路径的独立复现，不精确定位浮点 kernel，也不宣布全部 adapter 正确。

req_00001 的原生取消输出与此前不同，同样保留，不只报告匹配的请求。
官方 [batch invariance 文档](https://docs.vllm.ai/en/v0.30.0/features/batch_invariance/)
提供消除 batch-size/order 依赖的可选模式，并说明性能代价；本次未打开该模式，
没有为追求 hash 一致而改变正式候选配置。

本次整体 `pass=false`：在最后的错误权重对照前发现 finance_lora 与
writing_lora_0011 的 safetensors SHA 均为
`dfd99d29ccc482c8634823bef3eb290a928e79e9de52bcf0fdabc5b581ab7bf3`。
两个逻辑 adapter ID 不能冒充两套不同训练权重。已有 100 请求结果包含另一
权重 SHA（如 code_lora_0015），后续只补同提示、正确/真正不同权重的原生对照，
选择第一套不同 SHA，不按生成结果挑选，也不重新生成工件。

状态表与完整 hashes：
`paper_results/ieee_tc/p2_backend/20260926_3b_nativecancelref4_partial.{json,csv}`。
64 次外置采样；峰值 4,978,348,032 bytes；high/max/OOM 全零。
GPU contexts 清空、scope 移除；因提前终止，未执行最终 adapter 显式 eviction，
不把进程退出释放写成完成了该检查。该诊断不进入性能排名。

此外，原生 0.30.0 对旧 3B profile 的 `enable_chunked_prefill=false` 发出
不支持关闭的警告（`arg_utils.py` 的 `_set_default_chunked_prefill_and_prefix_caching_args`）。
这不是上述输出变化原因的证明，但后续后端配置资格须选择受支持路径再冻结；
不能仅凭本次 native 复现，把全部旧配置直接放行到正式性能实验。

## 当前候选及本机条件

候选：[vLLM 0.30.0 官方发布](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)，
发布于 2026-09-22。官方默认 CUDA 13.0 路径；同版本用于 Prime 和独立 vLLM，
其他论文 baseline 保留其专用后端。

本机：4 × RTX 3090 / SM86 / 24 GiB，驱动 580.105.08，Linux x86_64。
[NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
要求 CUDA 13.x 驱动至少 580，但存在特性/PTX 限制。因此只能认为满足基本
驱动门槛，不能据此宣布所有 LoRA/JIT 算子已兼容，也不自动升级宿主驱动。

旧主环境保持：Python 3.12.12、vLLM 0.10.2、torch 2.8.0+cu128、Ray 2.54.0、
transformers 4.57.6、triton 3.4.0。

## 依赖解析证据

在 3/4 GiB、swap=0、CPU 2/26 的独立 build scope 中运行 pip dry-run：

- 显式 `--ignore-installed --only-binary=:all: --no-cache-dir`，官方 PyPI。
- 198 个二进制包解析成功，退出码 0，没有向旧环境安装任何包。
- 原始 report 位于 `results/ieee_tc/p2_backend_qualification/metadata_20260925/`。
- report SHA256：`feae96b521e953cd0670006e64354d800f16e4e3b4b6773caffb4e238405900a`。
- 完整版本/hash 锁：`paper_results/ieee_tc/p2_backend/vllm0300_py312_candidate_20260925.txt`。

| 包 | 解析版本 |
|---|---|
| vLLM | 0.30.0 |
| torch | 2.13.0 |
| torchaudio / torchvision | 2.11.0 / 0.28.0 |
| transformers / triton | 5.17.0 / 3.7.1 |
| cuda-toolkit / nvidia-cuda-nvcc | 13.0.3.0 / 13.4.92 |

后两 CUDA 包的版本差别是实际解析结果，不隐瞒、不先断言运行失败。后续
必须检查实际编译/加载路径；若失败，用具体错误与官方兼容矩阵决定版本适配，
不靠关闭错误检查或静默退回 CPU 来通过。

## 独立安装

目标：`/home/qhq/.venvs/primelora_vllm0300_tc_20260925`。
使用现有 Python 创建隔离 venv，拒绝已有目标目录；pip 使用 `--isolated`、
`--require-hashes`、`--only-binary=:all:`。不复制模型、LoRA、trace 或旧环境。
安装进程隐藏 GPU，实际运行前读回 cgroup 3/4 GiB、swap=0 和 CPU 2/26。

启动前保守登记新增峰值 80 GiB；需要至少 220 GiB 空间，实际约 339 GiB，
已有 `20260925_p2_install_preflight.json` 通过。这个 80 GiB 是 setup 空间上界
估计，不是 GPU 预算或论文性能数值；最终记录实际增量。

本次 setup 在专用 tmux server `tc-p2-0925-01` 的 `install` 会话运行，属于
`primelora-tc-build-b52bf775f5d34836860733f2764e13d5.scope`。不能重新启动同一目标
覆盖中间状态；先核查日志和最终 receipt。安装失败保留日志/环境，不盲重试。

日志、pip 原始报告不入 Git；curated 安装回执和版本/hash 锁入 Git。
安装成功、pip check 成功也仅证明依赖 setup，不证明 3B/7B 性能达标。

### 2026-09-26 实际安装与单卡运行检查

原安装于 05:18 完成：三个步骤均退出 0，198 个锁定依赖安装完成，
`pip check` 无破损依赖。安装 scope 峰值 3,221,925,888 bytes，high 事件
16,501，max/OOM/OOM-kill 均为零；high 造成的回收压力是安装条件，不是
推理性能结果。未覆盖旧环境，未重复下载或升级宿主驱动。

沿用 `ieee_tc_preflight.py` 的 guarded launcher，新增显式 `backend-check`。
在导入 torch 前验证服务归属、外置监控和完整安装回执；限定一张可见 GPU。
检查本机实际导入版本、原生 vLLM 接口、FP16 32×32 矩阵乘法与当前流 event；
退出后的 GPU 进程清理由外置观察确认。它不加载模型，不代表 LoRA 或性能资格。

| 检查 | 结果 | 边界 / 后续 |
|---|---|---|
| 独立安装及依赖一致性 | 通过 | 两模型仍待测 |
| 启动 attempt 1 | 模型启动前被 guard 拒绝 | 已结束安装的空 scope 仍为 active；确认 populated=0、无进程后清理，仅清本任务两个 build scope |
| import attempt 2 | 失败，保留原始回执 | 检查脚本误用旧 `vllm._C`，不是硬件或模型失败 |
| import/CUDA attempt 3 | 通过 | torch 2.13.0 / CUDA 13.0，vLLM 0.30.0；实际 SM86，FP16 结果全部为 32 |
| attempt 3 资源与退出 | 通过 | 23 个资源样本，3 次观测到本服务 GPU context；退出后清除，服务资源域删除 |
| 7B/3B、动态 LoRA、原生时序与引用 | 未测 | 下一主线任务，不将上述通过替代此项 |

attempt 1 未生成运行回执，也未执行 CUDA 检查：前置重型任务检查在创建
服务前拒绝；对应辅助 scope 为 `primelora-tc-aux-f80536a4b85a48b7a61800711c7a9632.scope`。
第一次尝试调用 service/test 专用清理 API 处理 build scope 被其类型检查拒绝；
随后只对已核验为空的两个具体 build UUID 执行停止，没有扩大 API 允许范围。

接口修正依据实际安装文件和
[官方 v0.30.0 CUDA platform 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/platforms/cuda.py)：
该版本导入的是 `_C_stable_libtorch`。未增加旧/新模块轮流尝试，未跳过原生
扩展检查，未改安装包。官方 GPU 安装文档要求实际软硬件兼容，基本驱动
门槛本身不证明全部 kernel 可运行：
[vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。

attempt 3 服务观测峰值 971,829,248 bytes，high/max/OOM/OOM-kill 均为零；
最低主机可用内存 110,798,262,272 bytes。它是轻量资格检查，不能外推
模型初始化峰值或完整 workload 内存占用。原始记录在
`results/ieee_tc/p2_backend_qualification/runtime_20260926/`；摘要与 SHA 在
`paper_results/ieee_tc/p2_backend/20260926_cuda_import_qualification.json`。
按计划 11.2 使用状态表，不从资格检查制作系统性能图。

检查入口的四个无 GPU 测试覆盖先保护后导入、未完成/不同环境拒绝、导入
失败无后端替换、原生扩展名称正确且无 CUDA 时明确失败。安全/原生 GPU
观察/外置回放合计 48 项通过。下一步使用现有 InferenceEngine 与冻结旧工件、
旧 trace，验证实际 worker 归属、时钟、slot、生成与释放；不另建推理框架。

### 真实 3B 首次检查与通信类型修正

复用原 3B seed42 trace 前四请求、两个既有 adapter、原始模型 profile；仅启用
共同固定输出和原生观测合同。单卡 TP=1，GPU/CPU/内存包络与外置监控不变。
这不是 open-loop 主比较，也不是远端工件实验，不产生 SLO 或系统排名结论。

| 项目 | attempt 1 观测 | 判定 / 下一步 |
|---|---|---|
| 原始 3B 权重与 LoRA runtime | 权重加载、编译和 CUDA Graph 初始化完成 | 硬件/后端基本模型路径已执行，完整资格未通过 |
| 首次 worker 观测返回 | `TorchVersion` 无法由原生 msgpack 编码 | 自定义观测传输错误，不是推理失败或 OOM |
| 实际请求生成 | 尚未开始 | 不填造 token/TTFT 结果；内层模型回执缺失明确保留 |
| 内存与退出 | 177 样本；service peak 5,147,086,848 bytes；high/max/OOM/OOM-kill=0 | 最低主机可用 106,486,501,376 bytes；仅终止本轮所属进程，GPU contexts 清除、scope 删除 |
| 修正后回归 | 预修正测试失败；修正后 565 项功能测试通过 | 同配置 attempt 2 验证真实通信和请求，不改模型策略 |

原始结果位于 `results/ieee_tc/p2_backend_qualification/model_20260926/`。
attempt 1 launch SHA 为 `e7dbf73848343357daacbc639ea15b45d968e24502b87b3009e1186ed25c8b0e`；
service log SHA 为 `44c057530f28ffa79d6ffba36a8929bcef43c5b0d35d3cd2e3920e4a888e62da`。
通信线程失败后，按既定所有权接口停止三条本轮进程，未 hard-kill，无其他作业受影响。

仅把两个版本标签显式转为普通字符串。依据
[vLLM 0.30.0 原生序列化源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/serial_utils.py)
与实际安装的 PyTorch `torch_version.py`：字符串子类不能直接作为该原生通信对象。
未开启不安全 pickle、未捕获后伪造字段、未跳过 worker 观测。后端、工件、生成
目标和模型参数不变；保留首次编译缓存以节省空间和重复准备，启动条件如实记录。

#### 3B attempt 2：四请求原生合同通过

| 项目 | 观测 | 限制 |
|---|---|---|
| 原始四请求固定输出 | 152/59/123/217 token，逐条相符，原生终态完整 | 共 551 token；不是 4,000 请求完整回放 |
| LoRA 原生状态 | 两 adapter 真实加载；其中两请求复用已加载 adapter；引用释放与最终驱逐通过 | 不代表 500 池、mixed-rank、并发取消全部合格 |
| worker / 时钟 | 实际 worker PID/cgroup/CPU affinity/单卡/clock 与外部检查一致 | TP=PP=1；不证明多副本控制正确 |
| native 时间重算 | 每条 E2E 分解与 TPOT 重算误差为 0 | 此处 dispatch 在后端 generate 边界，不是用户到达 TTFT |
| 资源与退出 | 65 样本；service peak 4,997,853,184 bytes；high/max/OOM/OOM-kill=0 | 最低主机可用 106,262,097,920 bytes；contexts 清除、scope 删除 |

摘要：`paper_results/ieee_tc/p2_backend/20260926_3b_prefix_qualification.json`。
首个请求触发原生 LoRA kernel JIT，日志保留；不把该四请求测量当 warm reference。
本轮启动复用了 attempt 1 编译缓存，不用两次启动差异归因算法改进。
实际 dense LoRA pool 为 1,825,046,528 bytes，两 adapter 的注册 CPU tensor storage
为 18,350,080 bytes；GPU slot 为固定预分配容量，不用磁盘文件大小替代显存。
下一步相同入口验证原始 7B profile，之后再扩展完整模型资格。

#### 7B attempt 1：启动工具搜索路径失败

| 项目 | 观测 | 判定 |
|---|---|---|
| 权重与图编译 | 原始 7B 权重完成加载；torch.compile 完成 | 单卡实际执行，不是模型模拟 |
| 原生 FlashInfer 初始化 | `FileNotFoundError: ninja` | 外层使用 venv 的绝对 Python，但未把该环境 bin 加入 PATH |
| 工具是否安装 | 原隔离环境已有 ninja 1.13.2，绝对路径执行成功 | 不需要重装、换后端或关闭 FlashInfer |
| 请求 | 0 条；模型资格失败 | 保留失败，不产生系统性能点 |
| 资源与退出 | 102 样本；peak 16,392,409,088 bytes；high/max/OOM/OOM-kill=0 | 最低主机可用 106,927,349,760 bytes；GPU contexts 清除，scope 删除 |

attempt 1 launch SHA：`8694787e8faa6be6bffe7ebd55356bf8b6acf5a20dcd95eafb3f865fba4126d2`。
FlashInfer 安装源码 `jit/cpp_ext.py:run_ninja` 使用 PATH 搜索 `ninja`。
下一次只在启动环境 PATH 前加入既有 candidate/bin；保持原模型 profile、
FlashInfer、数据与生成合同不变。这是明确的环境入口修正，不是失败后的算法兜底。
参考 [FlashInfer 安装与 JIT 依赖说明](https://docs.flashinfer.ai/installation.html)。

#### 7B attempt 2：四请求原生合同通过

| 项目 | 观测 | 限制 / 解释 |
|---|---|---|
| 真实固定输出 | 152/59/123/217 token，4/4 相符、终态完整 | 不代表 4,000 请求或正式排名 |
| worker 与 LoRA | 单卡实际 worker 归属/时钟通过；两 adapter 的加载、复用、引用释放、驱逐通过 | GPU pool 1,316,225,024 bytes；注册 CPU tensor 33,554,432 bytes |
| 原生时间合同 | 四请求 E2E 与 TPOT 重算误差均为 0 | 尚不是外置到达至完成的 Full 计量 |
| 编译依赖 | candidate/bin/ninja；实际 nvcc 为 `/usr/local/cuda-13.0/bin/nvcc`；SM86、两编译任务 | 未安装新工具或替换采样机制；第一次 FlashInfer JIT 开销保留 |
| 资源与收尾 | 174 样本；peak 5,031,833,600 bytes；high/max/OOM/OOM-kill=0 | 主机最低可用 106,054,262,784 bytes；GPU contexts 清除、scope 删除 |

摘要：`paper_results/ieee_tc/p2_backend/20260926_7b_prefix_qualification.json`。
驱逐 adapter 不等于释放整个 dense GPU pool；后者仍由 runtime 持有，直到实际
进程退出。两次初始化的文件缓存与编译缓存状态不同，不把峰值或启动差异称作
Prime 优化收益。两个模型的失败尝试、原始数据、hash 与状态表全部保留。

这一步通过的是实际两模型的基础 native path，不是以下尚未完成项目：
100 请求 smoke、完整 500 池、mixed rank/modules、并发/取消/抢占、完整 source
epoch 与原子 admission、多副本扩缩容、真实 Remote、warm reference，以及
正式 G1/G2。下一步扩展已有检查入口，优先让实际请求验证这些路径；不返回
无限堆叠与模型无关的独立测试。旧环境/结果/工件未覆盖，也未生成新 workload。

### 100 请求资格：保留原容量的 adapter 更替

使用同一 seed42 源 trace 前 100 条、原模型 profile、固定输出与原生合同，
顺序执行；不是 open-loop 主比较，也不是 warm reference 或真实 Remote。
未扩容缓存来容纳全部出现的 adapter，未修改 kernel、加载或淘汰策略。
在启动前根据官方 LRU 行为修正资格检查的收尾：快照已不在 CPU cache 的
adapter 必须返回 `evicted=false, reason=absent`；仍存在者必须成功移除。
referenced/externally_pinned、未知 source 或非空最终 cache 均失败，而非兜底放行。
[官方 LRU 路径](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
说明容量不足时会淘汰旧条目，因此不能强求所有曾加载 ID 到最后仍驻留。

| 项目 | 3B smoke100 attempt 1 | 7B smoke100 attempt 1 |
|---|---:|---|
| 正确完成 / 目标数量 | 100 / 100 | 100 / 100 |
| 原生 output tokens | 17,369，逐请求目标相符 | 17,369，逐请求目标相符 |
| 实际 logical ID / 权重 SHA | 29 / 2 | 29 / 4 |
| 加载前 GPU / registered CPU / local file | 45 / 26 / 29 | 28 / 42 / 30 |
| 最终清理前 GPU / CPU cache 条目 | 8 / 29 | 4 / 24 |
| 已被原生 CPU LRU 淘汰的 logical ID | 0 | 5 |
| E2E / TPOT 重算最大误差 | 0 / 0 ms | 0 / 0 ms |
| 与此前四请求 prompt/output hash | 四条全相同 | 四条全相同 |
| 资源样本 / peak bytes | 297 / 5,302,902,784 | 477 / 5,604,081,664 |
| high / max / OOM / OOM-kill | 全零 | 全零 |
| adapter cache 清空、GPU contexts 与 scope 释放 | 全部确认 | 全部确认 |

29/2 与 29/4 只描述各自这段前缀，不外推整个 500 池；逻辑租户身份与独立权重数分开报告。
local file 不意味着物理冷盘：资格输入预检已经读取内容 SHA；加载成本不用于
S1 层级性能结论。native GPU pool 仍为预分配容量，非随每次驱逐释放显存。
逐请求 CSV、摘要 JSON 与原始 SHA 已保存在
`paper_results/ieee_tc/p2_backend/20260926_{3b,7b}_smoke100.{csv,json}`。
7B 的 29 个逻辑 ID 超过原始 24 条 CPU cache 容量：30 次文件加载包含再加载，
最终仍有 5 个 ID 已正常淘汰。没有增大缓存来规避更替，也没有把 `absent`
与引用未释放混为一谈。两模型前四条输出与此前各自成功的四请求检查一致。
这些事实支持本机原生加载/更替路径的正确性，不支持跨系统延迟或 G1/G2 优越性。

### 实际并发批次资格（两模型通过，非完整资格）

现有模型资格入口增加显式 `--qualification-mode concurrent_pairs`，只使用
原 trace 前四条：0/1 共享 finance adapter，2/3 使用 writing/finance。
先取得各自引用，再同时提交；保留原生异步调度，不改变 batch/slot 配置。
原生观测额外返回最近在途批次的确切 request IDs；前端映射在原生 owner 中
读取并复制，既不猜随机后缀，也不关闭后端随机化。

| 检查 | 3B batch4 attempt 1 | 7B batch4 attempt 1 |
|---|---|---|
| 两组原生同批次、各自持有 KV | 两组均实际观测 | 两组均实际观测 |
| 共享 / 不同 adapter 引用 | 2 个共享引用 / 各 1 个引用 | 2 个共享引用 / 各 1 个引用 |
| 持有引用时显式驱逐 | 四次全部拒绝，原因为 referenced | 四次全部拒绝，原因为 referenced |
| 完成与原生 token | 4/4；152/59/123/217 | 4/4；152/59/123/217 |
| 与先前串行 prompt/output SHA | 四条均一致 | 四条均一致 |
| 最终 scheduler / reference / cache | 无请求、无在途批次、无延迟释放块；引用归零、adapter 清空 | 同左，均实际验证 |
| E2E / TPOT 重算 | 最大误差均 0 ms | 最大误差均 0 ms |
| 外置资源观测 | 62 样本；peak 4,989,104,128 bytes；high/max/OOM/OOM-kill 全零 | 56 样本；peak 5,029,855,232 bytes；high/max/OOM/OOM-kill 全零 |
| GPU contexts / 服务资源域 | 均确认释放 | 均确认释放 |

3B 两组分别有 76 / 111 次、7B 有 105 / 181 次资格观察。20 ms 观察周期只用于这个诊断，不参与在线
策略，也不将这次受密集观察/驱逐探测影响的时延用于主表或 warm SLO 标定。
原始回执与哈希、逐请求 CSV、摘要 JSON 位于 `20260926_{3b,7b}_batch4` 产物中。
3B result SHA：`446114635f076305dfbc6ec79c56af96fd3ac3465d8c169e33c3e317870e1288`；
7B result SHA：`0303d874d2b117e792371af24e81b5c34f3f5747584c32385599113b353da3ef`。
这一步验证真正并发，不是仅凭两个 asyncio task 断言已形成 native batch。
上述驱逐探测发生在取得引用后、提交生成前，不能称为在途取消时的卸载验证。

两次运行使用同一份实际源文件（之后只更新文档和 curated 数据）：

| 文件 | SHA256 |
|---|---|
| ieee_tc_preflight.py | aeeef76763b5b6c8a7dc497482b6bcb924a709203a6c01534fb00635592eaa3c |
| run_all_experiments.py | cffa9292b7e44a890a64528dc5af5b8c8f0afb5e39ecf3f6bc6c59981947bfe4 |
| scheduling/resource_coordinator.py | cc25fcbdc29303b71284216bfcad58703dfac7d85decabd39892e5919b743132 |
| scheduling/vllm_ieee_scheduler.py | 7ce9265bf885cafda2bddc0ed56bc150b49dcd6f8353649d1e5c843cf429255d |
| memory/gpu_monitor.py | d3032ddbc1438e74b119d7cea371d0f37eea6843e9c89f2f7b3320205f3b4cf5 |
| memory/residency_manager.py | 2c230bff24fb4c4bc87ef055493ead97d2a4bfa9520f26069dd3407de68a4643 |

### 原生取消边界（018e5a6 检查点的源码审计与反例）

继续复用此入口和原始请求，不另建框架。下一步先验证两个真实请求并发时的
原生 scheduler/KV 观测、共享/不同 adapter 引用与禁止活跃驱逐；再验证取消。
依据实际安装的 v0.30.0 与
[官方 AsyncLLM 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)，
取消生成器会请求 abort，但不能据此直接宣称设备工作完成。
[core_client 的 abort_requests_async](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)
只发送 ABORT；后续必须取得调度所有者的请求终态和在途执行证据。
此外，`OutputProcessor.abort_requests` 会在前端构造 `finish_reason=abort`
的结束输出。因此 `out.finished` 本身不等于 EngineCore 的原生终态；正式
取消集成不能把这类前端结束输出直接送入 `end_use` 或成功 TTFT/TPOT 统计。
依据 [官方 output_processor 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/output_processor.py)。
已用既有三 token 测试夹具做反例：给 actual `generate_prepared` 的完成输出
标注 `finish_reason=abort`，当时入口仍返回成功和 `native_terminal_observed=true`。
这证明当前接口缺少终止原因区分，不是一次实际 GPU 取消测量；原生取消资格
仍未通过。反例输入与结果保存在 `20260926_frontend_abort_counterexample.json`。
下一步需先拒绝这类通知作为正常终态，然后连接确切请求身份与后端在途完成，
而不是删除引用保护、睡眠固定时长后强行释放，或重新加载整个模型冒充请求级回收。

默认原生内部 request ID 带随机后缀，与前端 ID 不同。应保留真实映射，不能靠
字符串前缀猜测或关闭随机化；也不能用另一个成功请求的终态释放取消请求。
原生 scheduler 可以移除请求而仍保留在途批次/延迟释放块，因此单次“请求
不在队列中”的观察不足以证明所有权已结束。设计应关联确切原生 ID、调度
序列与完成事件，再执行引用释放；未知状态继续保留引用而不伪造成功。
完整池、Full 控制器、真正 open-loop 与远端路径仍是后续独立资格门槛。

### 请求取消与在途完成集成（2026-09-26，局部实测完成，边界仍开放）

这一修改解决计量正确性，不宣称吞吐或时延优化，也不改变 IEEE 公式：

1. 正常完成要求原生 `length/stop`；固定输出只接受 `length`。前端合成的
   `abort`、错误或缺失原因不能算成功，即使 token 数正好等于目标。
2. 小型、版本限定的 AsyncLLM 子类只记录原生 ADD 的 external/internal ID。
   不关闭随机 ID，不修改采样、batch、cache 或调度策略。取消时等待已有 ADD
   真正结束，防止一个延迟 ADD 在“已取消”后重新进入服务。
3. 同一 scheduler 所有者记录原生 add/remove、全部在途 SchedulerOutput
   和原生 deferred-KV fence。缺少已观察 add/remove、仍有相同 ID 的在途批次，
   或 KV fence 未完成，都不能签发 retirement。不是只看最近一个 batch。
4. EngineCore utility 返回 Future，随原生 executor 完成事件推进，不阻塞
   自己的调度线程，不固定睡眠推断完成。收到确切请求回执后才 `end_use`；
   正常完成同样经过此边界，随后 worker 引用释放保留既有 CUDA event fence。
5. 既有控制器、子进程 RPC 和 proxy 增加同一请求归还入口。取消仍是失败/
   取消结果，不因资源已归还而转为成功。未知绑定、丢失 acquisition 回复、
   EngineCore 失败继续保留所有权，不能猜测释放；进程级退出仍由外部所有者计量。

依据当次安装源码和 [官方 core 的异步批次与 utility Future 路径](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)、
[官方 scheduler 的 deferred block free](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)。
无 GPU 回归：574 项功能、54 项安全通过。第一次针对性检查有一个测试夹具
未设置普通完成原因，补上显式 `length` 后通过；没有放松生产校验。

真实资格复用原四请求，在共享/不同 adapter 两组中等待两个请求都实际解码，
再取消第一条。要求另一条仍在运行、保留独立引用并正确完成；共享 adapter
不能卸载，不同 adapter 的取消一侧可以在 retirement 后卸载。采用 20 ms
密集诊断采样，不纳入主性能统计。全量 Full、跨进程真实取消、排队/提交前
取消、完整池及失联恢复的模型级资格不由这四请求代替。

3B 首次真实取消检查的阶段结果（原始 attempt 不覆盖）：

| 检查 | 共享 adapter | 不同 adapter |
|---|---|---|
| 实际两请求解码后取消一条 | 通过 | 通过 |
| 取消后只剩另一条的引用 | 通过 | 通过 |
| 被取消 adapter 的卸载探测 | 正确拒绝 | 正确允许 |
| 另一条输出长度 | 59/59 | 217/217 |
| 另一条输出 SHA 与旧正常完成对照 | 相同 | **不同，需审计** |
| 最终引用、cache、GPU context 清理 | 通过 | 通过 |

因此 curated 状态将 ownership 通过与完整输出资格分开，不把原始 qualifier
的四个局部检查通过当成四条请求正确完成。两条是有意取消；其最终输出数量未知，
不填零。当前不同-adapter 对照的输出差异需要进一步解释。下一次同设置只关闭
“取消后主动卸载”探测，不修改模型、采样、token 目标、缓存容量或取消触发条件。
[vLLM 0.30.0 batch-invariance 文档](https://docs.vllm.ai/en/v0.30.0/features/batch_invariance/)
指出默认模式的输出不保证独立于批次构成，但这只能提出可检验解释，不能替代
当前案例的验证；不为消除差异就悄悄打开可能改变性能的 batch-invariant 配置。

3B retain-adapter 对照已结束：两条未取消请求的输出 SHA 与 cancel4 attempt1
完全相同，仍分别为 59 和 217 tokens，prompt SHA 全部一致。两次均只剩另一
请求的引用，最后清理完成。由此只能排除“差异必须由取消后的主动卸载产生”，
不能据此声称已证明浮点批次效应、正确 adapter 或全部取消场景。
两次检查的服务内存峰值分别为 4,952,301,568 / 4,933,423,104 bytes，
各 60 次外部采样，high/max/OOM/OOM-kill 均为零；旧数据没有覆盖。

| 来源 | 资源检查 | 输出审计 | 原始结果 SHA256 |
|---|---|---|---|
| 3B cancel4 attempt1 | 通过 | 不同-adapter 存活请求与无取消参考不同 | 6893ea227b435284de6a0e6affeff62f753fee9f8307296fb6cdceed9d2f4044 |
| 3B cancelretain4 attempt1 | 通过 | 与 cancel4 两条存活请求均相同 | 29383c230f3fb6c534dc30cdf113fa7428fd636fa66af6048b5519f02e8f996c |

输出审计未闭合的 curated JSON 明确 `pass=false`，并单独保留
`ownership_checks_pass=true`、两条有意取消和两条固定长度完成。

7B cancel4 attempt1 的两组均通过同样检查，存活请求为 59/59、217/217，
两条输出 SHA 和全部 prompt SHA 与旧正常完成参考相同。取消后每组仅剩
存活请求的引用；共享 adapter 卸载拒绝，不同 adapter 卸载成功，另一请求
继续完成。54 次外部采样，峰值 5,033,377,792 bytes，high/max/OOM/OOM-kill
均零，GPU context 清空且 scope 消失。原始结果 SHA256：
`2322a5be2bbf448c18b55afcbb23a82ee5b710dfbe7ad162020536b44ccdeb97`。

两模型本配置的原生 `defer_block_free` 路径未推进序列（回执 fence/processed
均为 0），不能宣称实测触发了非零 deferred-KV fence。全部在途批次的确认
仍由独立的 NativeIterationObservation 检查；非零 fence 的等待只在受控
测试对象上验证。最终没有 native admitted request/在途 iteration 遗留。

执行源身份（修改均限本仓库，没有改安装的 vLLM 文件）：

| 文件 | SHA256 |
|---|---|
| scripts/run_all_experiments.py | 9534ac8d6fc4d8aca24170f56d139e2ba24ffb325ffeb7ac7a6e676f09680704 |
| faaslora/scheduling/resource_coordinator.py | 0fc7da8bbb37ba7b377915a69ecd2ff1ca905c1c7fbf8f5f436b3a4a725bb678 |
| faaslora/scheduling/vllm_ieee_scheduler.py | bf1bc75686acd51e5d65d6d2fd0cd827f4d1107d791aa9049d978601cc36b814 |
| faaslora/scheduling/vllm_ieee_frontend.py | 2020beeedd381fb2d5aa0957ef9d63bb617e839dada3c77c9c672ab2516c1dca |
| scripts/dedicated_engine_worker.py | 05dc860609be78aaea29efb361aed4c7d0cdaa94ea5e8e5bd7f75c708f6cab5d |
| scripts/ieee_tc_preflight.py，3B cancel4 | a2c4bda7c96eb47d70cb933f08145e5bc8143eb84be0af0c439f63a246c807c9 |
| scripts/ieee_tc_preflight.py，3B retain/7B cancel | 0cee556c523ddf80466bd6a104699e30da5ccd3a75848f95c5ede28a2255f41d |

跨进程边界仍有具体待解决项：旧 proxy 在取消 socket roundtrip 时会设置
`_engine_dead=true`，随后拒绝包含 retirement 的全部 RPC。因此新入口和
fake-engine 控制器测试不等于真实断连恢复。必须区分“该 socket 不可信”和
“原生 engine 已死亡”，为确切 dispatch 的对账提供独立、可核验的控制通道；
不得复用被取消的 socket、重发生成请求、无证据清除失效标志或提前释放资源。
这个问题不影响上述本地 InferenceEngine 的实测结论，但阻止宣称 Full
跨进程取消已经合格。下一步与 3B 原生输出对照一起闭合，再进入完整池主线。

## 全池内容核查后的资格边界（2026-09-26）

后续 stock-native 两次同 prompt 对照已经完成，并进一步检查当前两池全部权重。
**3B 500/500、7B 498/500 的 A、B 全零**，不同权重 SHA 仅 2/4 个；7B 的
finance/medical 两份含非零值。此前顺序/批次/取消检查证明了各自记录的原生
调用、长度、所有权和清理，不证明 500 个独立训练模型或全面数值正确性。
3B 同 prompt 的不同 SHA 控制均输出相同 217 tokens；原始 mechanical pass
保留，但语义区分门槛没有通过。详见 [完整内容核查](ARTIFACT_CONTENT_AUDIT.md)。
不再重复数学上不可区分的零权重负对照，不修复/重新生成工件以掩盖发现。
计划禁止新增权重，少量非零 3B 独立正确性工件需用户另行确认；原池、trace、
旧测量保持原样。Full 集成及正式性能矩阵仍未完成。

## 原生计量接入注意事项

现有 runner 的 legacy `_derive_vllm_latency_metrics` 优先 finished timestamp。
新的 opt-in `timing_contract=ieee_tc_native_v1` 已接入 engine、子进程返回和
controller，严格按首末 token 分离完成通知开销；13 项无 GPU 测试通过，
详见 P1-D5。真实新版 stats/时钟/开销资格仍未完成，不能据此放行推理实验。

当前官方 [per-request metrics 文档](https://docs.vllm.ai/en/latest/features/per_request_metrics/)
有显式开启开关，要求统计记录可用；API TTFT 边界从 scheduled 开始，并非
本项目计划到达。streaming metrics 在最后 usage chunk 返回。资格时必须核对
实际 0.30.0 源码、原生字段及边界，不能仅凭字段名称拼接 admission D/T/O。
每请求统计本身也可能增加 CPU 开销，标准监控和详细 profiling 分开验证。

仍需按计划测试：FP16 两模型、mixed rank/modules、动态 LoRA、adapter
identity、CUDA Graph、batch/cancel、原生 token、真实 slot/eviction、KV block、
正确释放以及 warm 指标。实际 worker 限制和监控握手通过前不启动模型比较。

## 原生 worker 观察接口（已接线，真实模型调用待资格）

沿用 `gpu_monitor.py`，新增官方 `worker_extension_cls` 允许的只读扩展，
通过既有 engine collective RPC、dedicated worker 和 proxy 返回数据。
配置 `model.ieee_worker_observation=true` 才安装扩展；默认历史路径不变。
参考 [vLLM 0.30.0 WorkerWrapperBase 的 extension 接口](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/worker_base.py)。

读取：原生 worker PID/UID/cgroup/affinity、boot/time namespace、后端版本、
可见 GPU 与 local device、实际 CUDA mem-info、PyTorch allocated/reserved、
LoRA CPU registry、active GPU set 与 slot mapping。

LoRA A/B tensor 的 shape、dtype、view size 与底层 storage 分开记录，共享
storage 去重。不是用文件大小或 rank 估算显存，也不把整池分配再按多个 view
重复相加。当前仅接受已审计的 dense stacked A/B 表示；未知模块报错，不漏计。

默认不做 GPU synchronize。可显式 barrier 进行独立资格/剖析，但不把这种
同步放进每请求监控。无论是否 barrier，该快照没有 dispatch 引用或原子
reservation，始终标 `production_admission_snapshot=false`；不得直接冒充
IEEE confirmed tier 的租约证据。

五项 fake-tensor/实际 RPC 方法测试通过，覆盖 storage alias、CPU/GPU 区分、
slot 不一致、未知表示、同步标记、嵌套字段运输和错误不吞掉。它们没有加载
模型，也没有证明原生 worker 已满足资源包络。下一次真实模型资格必须读取
这些事实，验证后才能将其接入 owner；不能先把代理提示映射成 GPU-ready。

## 原生 GPU 引用接入（模型资格前的独立合同）

`model.ieee_gpu_references=true` 安装同一个 worker extension，并在真正的
worker CPU/GPU LRU 上维护请求引用；不是只修改 router 的 resident set。
同一份扩展的 observation 仍只读，不因启用引用功能而自动成为受保护快照。

- 当前限定每个 runtime TP=PP=1；多 runtime 横向扩容不等于 TP>1。
- snapshot 不持有引用；acquire 必须提交 worker incarnation 和 epoch。
- CPU-only、不再驻留或过期快照返回 conflict，不通过隐式 load 变成“原 GPU hit”。
- 第一个引用保护 native CPU/GPU cache；多个请求共享；最后一个引用只解除
  本 owner 的 pin，保留已有外部 pin。pin 不是本系统独创算法。
- 在 worker 当前 CUDA stream 上记录并等待 event，计入 acquisition 开销；
  不进行全设备同步。必须在真实资格中确认本机 dense LoRA copy/execute 的
  stream 和线程顺序，不能仅凭 fake event 测试宣称 readiness 已完成验证。
- 现有 engine/prepared/RPC 通道要求 generation 提交对应 adapter 的引用，
  并将原生 backend request ID 绑定到 lease。未见 native terminal 时不能释放。
  异常/取消不会假称 backend 已结束：保留引用直到明确终态或整 worker 回收。
- 显式卸载通过同一 owner；不能绕过 owner 调 remove、load_inplace 或 pin。
  正常 cache replacement 使用后端 LRU；检测到已有引用被旁路失效时，owner
  invalidated，停止使用该 worker，不把它重新解释为 ordinary cache miss。

这一步不实现 cold-load admission、active-request reservation、victim transaction
或慢层共享引用。没有将上述条件偷换为 True；回执明确
`request_admission_reserved=false`、`production_launch_authorized=false`。
后续真实模型资格用这些现有入口验证 hold/use/release/eviction，再完成控制器
快照与资源预约接入。没有使用本步骤数据声称 TTFT 或 GPU-s 已改善。

## 2026-09-26 D28：7B 原生容量等待资格已完成

这是上述原生引用工作的后续实测，不是正式 Full 结果。复用0.30环境、旧seed42
前32条输入和原有4个LoRA槽位，按出现顺序选5个不同adapter请求；前4个持有
实际引用，第5个在第1个真实decode期间尝试加载。没有缩小cache、注入sleep、
新增权重或负载。使用既有受限 dedicated worker、实际UUID分配和外置监控。

| 检查 | 实际结果 |
|---|---|
| attempt1 | 启动器漏传NVML组件SHA，模型启动前被拒绝；回执保留，非性能点 |
| attempt2目标输出 | 5/5；152、123、256、174、50，共755 native tokens |
| 容量冲突 | 4个原生live leases，无可驱逐槽位；req00008等待3764.347ms |
| 恢复边界 | req00000真正结束并确认释放后重新观察epoch、加载并完成；无周期轮询 |
| 引用与缓存 | 结束0个GPU/HOST live leases；随后受控adapter缓存清空 |
| 输入/输出核对 | 与D26 source32对应5条prompt/native-input/output SHA全部一致 |
| 原生TTFT/TPOT/E2E重算 | 最大误差均0ms |
| 物理占用 | 64.852 GPU-s，末token后7.742s仍计入；退出码0、实际上下文释放 |
| 资源保护 | 77次采样，峰值5699035136 bytes，high/max/OOM均0 |

逐请求CSV、两attempt及来源SHA见
`paper_results/ieee_tc/p2_backend/20260926_7b_capacity_wait.{csv,json}`。
源码检查点33ef68d已在真实运行前推送。676功能/56安全检查通过；所有147个历史
保护项未变。CPU容量等待、取消、多个等待者仍只有fixture证据；本次不称全路径
原生资格，也不解除独立adapter数值正确性和真实远程、全池、Full接纳等门槛。

下一步回到Full的物理KV/tier admission、实测profile和主动机制整合。此5请求
诊断已经回答当前问题，不重复以增加“实验数量”。

## 2026-09-27 D56：既有工件的原生HOST分配器实测

D55完成了整次部署的物理GPU计量接线，但HOST字节占满时的替换仍未通过。
本轮不是再次回放短前缀：复用现有preflight、受限服务和已审计工件，以
官方LoRAModel CPU checkpoint loader直接测量分配器行为，不加载backbone。
每池按权重字节、adapter ID排序，事先选每个权重SHA/rank/modules类的首项；
共3B两项、7B四项。全部文件前后SHA核对，不修改工件或生成负载。

每项执行：空对象集合→加载第一份→同时加载第二份→移除第一份→重新加载
第一份→移除全部。仅删除本测量拥有的Python对象，不调用allocator flush，
不预扣未来victim释放字节。对象集合供现有inventory读取，不模拟分配器。
同一进程保留跨类缓存，这是分配器机制诊断，不是各模型独立性能工作点。

### 默认分配器：第一次实测已完成

| 既有工件类 | 第一份移除后active减少 | 第一份移除后allocated减少 | 同形状重载新增CUDA host allocations |
|---|---:|---:|---:|
| 3B rank8 | 11,927,552 B | 0 B | 0 |
| 3B rank16 | 23,855,104 B | 0 B | 0 |
| 7B rank8，零/finance/medical三类 | 各16,777,216 B | 各0 B | 各0 |
| 7B rank16 | 33,554,432 B | 0 B | 0 |

最终所有checkpoint对象移除，active为0，allocator仍持有106,168,320 B。
3B rank8→rank16时，原有23,855,104 B缓存并未阻止新分配：大张量第一次
加载又增加23,855,104 B。因此总cached字节不能直接当任意shape的可复用额度。
同shape重载确实复用，但不能从该受控序列推断并发Full中的保证。

原始记录：`results/ieee_tc/p2_backend_qualification/host_20260927/native_host_allocator_default_attempt1.json`
及同名`_launch.json`。六项36步完成；17次外置资源采样，服务和watchdog退出0，
NVML确认本次context释放、服务资源域移除；本次辅助域与58项安全检查域为空后
停止。测量不是数值LoRA正确性、native registry/packing/GPU激活资格或SLO profile。

### 第二次最小验证的依据与范围

官方[PyTorch2.13配置解析](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/c10/core/AllocatorConfig.cpp)
允许`pinned_max_cached_size_mb:0`；[分配器实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/core/CachingHostAllocator.h)
将超过该阈值的块在依赖结束后归还，而不是放入缓存。本轮第二项在新的
独立受限进程中，对相同六项和相同步骤测量该官方配置，核对实际settings。
这不是在线反复flush补丁，也不修改Prime生产配置。若归还行为成立，仍须
评估频繁分配的开销、完整Full路径和预算内staging；不会直接宣布它是最优配置。
原生checkpoint加载语义依据[vLLM0.30源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)。

### 两种配置的最终结果与下一步

| 问题 | 默认配置 | 官方不缓存配置 |
|---|---:|---:|
| 核验的`max_cached_size` | -1（未设有限上限） | 0 |
| 六类工件的同shape重载 | 均无新CUDA host allocation | 3B每次224次、7B每次256次新分配 |
| 每类清空后的allocator allocated | 保留并随类别累积 | 每类均为0 |
| 全部步骤结束allocated | 106,168,320 B | 0 B |
| 外置采样的服务内存峰值 | 782,364,672 B | 707,137,536 B |
| high/max/oom/oom_kill | 均0 | 均0 |
| 服务/context清理 | 完成 | 完成 |

不缓存配置中移除一份3B rank8/rank16分别归还9,175,040/18,350,080 B；
7B对应16,777,216/33,554,432 B。**此选项还改变大小取整**：超过缓存阈值
的块不再向上取二次幂，不能把3B占用差异全部归因于释放策略。

两次均为单个新进程、固定相同输入次序，各六类36个状态点；不是独立性能
重复，也没有用两次服务峰值估计显著性或模型性能。没有H2D/在途DMA，故
没有验证GPU使用中的对象何时可以归还。源码、两个执行回执、watchdog SHA、
完整72行计数和限制见`paper_results/ieee_tc/p2_backend/20260927_native_host_allocator.{json,csv}`。

结论是收窄实现选择，而非宣布解决所有容量问题：

- 默认缓存的总cached计数不能作为任意incoming对象的可用字节信用；维持
  现有保守拒绝，不做减去cached总量的补丁。
- 官方不缓存配置确有可观测归还行为，可作为后续受限HOST配置候选，但
  失去重载分配复用；未在生产路径开启，未声称它改善TTFT或GPU-s。
- 完整方案仍需在相同HOST额度内显式容纳加载workspace，先通过收益及
  E(t)再执行替换，并在真实native worker中验证引用/在途copy/归还。
  不能因这次CPU loader测量通过而解除Full guard或补写缺失profile。
- 本问题的两次最小测量已完成，不再重复它们或旧请求前缀增加检查数量；
  返回Full预算内staging/替换与代表性成本profile主线。

## 2026-09-27：加载 workspace 分解与显式分配器候选（D57）

本次复用D56原始数据，不重跑两组allocator微测，也不重跑旧请求前缀。
瓶颈假设是：只给常驻adapter分配空间，会使官方load-before-evict路径在
替换时没有合法的临时空间；删除对象也不自动等于回收默认allocator缓存。
[vLLM0.30 worker源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
确实先加载再执行CPU缓存替换。IEEE原文也将staging从可用tier容量中扣除，
因此必须在现有总预算内部作空间分配，不能靠扩大预算解决。

对当前dense A/B、FP16、PyTorch2.13.0+cu130且实际确认不缓存的候选，令
S为已有checkpoint文件字节，R为由header形状和实际dtype得到的权重字节：

- 最终pinned tensor上界为R。
- 加载期间临时tensor上界W=S+R，保守同时计入源文件和转换对象。
- 一次加载的额外tensor峰值上界为R+W=S+2R。
- 执行时仍要求当前实际allocator占用＋该峰值≤同一个native HOST额度。
  所有已暂存对象、非LoRA pinned块和未结束操作仍计入当前占用，不预支驱逐
  或异步归还的字节。不将这些tensor上界称为总HOST/RSS上界。

这里的精确非取整R来自
[PyTorch2.13 allocator实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/core/CachingHostAllocator.h)：
缓存阈值为0时，正大小块不执行二次幂取整。其他配置继续使用原保守取整上界。

| 既有类 | 最终pinned上界R（B） | 临时上界W（B） | 单次加载峰值上界（B） | D56实际最终分配增量（B） |
|---|---:|---:|---:|---:|
| 3B rank8 | 9,175,040 | 18,379,520 | 27,554,560 | 9,175,040 |
| 3B rank16 | 18,350,080 | 36,730,056 | 55,080,136 | 18,350,080 |
| 7B rank8（三类同形状） | 16,777,216 | 33,588,328 | 50,365,544 | 16,777,216 |
| 7B rank16 | 33,554,432 | 67,143,184 | 100,697,616 | 33,554,432 |

最后一列是旧实测，其余为离线计算；**没有测量临时峰值**。全六类逐项表、
计算式和原始SHA保存在`paper_results/ieee_tc/p2_backend/20260927_native_host_workspace.{csv,json}`。
同SHA/同形状不被视为独立重复。

实现新增显式候选`ieee_native_host_allocator_policy: uncached_v1`：

1. 只在新runtime导入torch前设置官方选项；不修改父进程、不在线切换、
   不刷新allocator。默认模型配置不变。
2. 旧CUDA/HIP别名不能静默覆盖统一设置；若已有不同allocator调优则拒绝，
   不丢掉用户原设置。进程环境与payload必须一致。
3. native worker读回真实设置后，才允许使用非取整的加载上界；读回不匹配
   则失败，不按环境字符串直接认定已经生效。读回只做一次，避免热路径上
   重复整个allocator快照。后续实际占用检查仍逐次执行。
4. 候选属于模型配置身份，旧service/preparation profile不能直接套用。
   如后续选择此候选，共同vLLM基线需获得同一后端配置和验证机会。

范围：本次接通了候选配置和预算计算，不等于已经为完整Full分配好缓存/
workspace，也未证明多plan并发的空间上界或在途H2D归还。尚需在固定总预算内
冻结常驻容量、暂存并发和非LoRA开销，再做实际Full资格；不得用过大的native
额度隐藏问题，或用本表充当TTFT/profile/SLO数据。可能的分配开销回退必须测量。
Full启动保护、九个论文公式、旧结果和正式配置均保持不变。

## 2026-09-27 D58：固定HOST额度内的加载空间保护

本轮接通预算内的空间保护，不再重复D56的allocator测量。直接复用完整
500池审计和六个既有类的测量，生成
`paper_results/ieee_tc/p2_backend/20260927_native_host_workspace_contracts.json`；
原始SHA及离线重算一致。没有新工件、新负载、模型运行或新性能结果。

### 设计依据与不变项

原生worker串行执行CPU checkpoint加载，且先加载再替换；不能用尚未驱逐的
对象作为可用空间。如果主动staging占满余量，业务请求即使将来能替换旧项，
当下也无法加载。IEEE要求从tier容量中扣除staging，因而在既有额度内部
保护加载空间，而不是增大HOST预算、提前驱逐或预测allocator归还。
这属于执行容量约束，不替换IEEE的九个公式、收益目标或E(t)。

令B为已有native tensor额度、C为实际原生CPU cache条目容量，Rmax/Wmax为
当前完整既有工件集合在FP16下的常驻/临时上界，Pmax=Rmax+Wmax。
仅对实际读回成功的`uncached_v1`候选，在空owner上检查：

`B >= A0 + C*Rmax + 2*Pmax`。

A0是配置时实际已占tensor字节；两份Pmax分别允许一个主动加载和一次按需
加载。额度不足即判配置不可行，不修改B或C。它是保守可行性条件，不是
最优分区算法，也不是总服务RSS的保证。

| 冻结输入 | 3B | 7B |
|---|---:|---:|
| 已审计逻辑adapter | 500 | 500 |
| 不同内容/rank/modules类 | 2 | 4 |
| Rmax（B） | 18,350,080 | 33,554,432 |
| Wmax（B） | 36,730,056 | 67,143,184 |
| Pmax（B） | 55,080,136 | 100,697,616 |

实际每次主动加载前，用当前占用A、仅由已注册对象独占的真实storage X检查：

`A + P_incoming + (C*Rmax - X) + Pmax <= B`。

其中A始终包含所有plan的staging、共享storage、allocator保留块及其他
pinned占用。共享给staging的storage不能作为X抵扣。剩余cache增长量与一次
按需加载受到保护；没有按plan各分一份完整预算。按需加载保留原生替换策略，
仅检查`A+P_incoming<=B`；复用不新增加载峰值。所有路径仍做原有实际总字节
检查。incoming形状/dtype、原生cache容量变化或未知计数不能静默通过。

分区配置由现有控制器传入实际worker；同一物理owner的别名共享一个额度和
合同，不能重复预留或改变合同。取消plan后的staging释放须由实际owner确认，
再唤醒其他等待者，不靠固定间隔轮询。dense packing保持官方引用列表和
原地缩放，未扩展到另有tensor分配的MoE路径。

### 正确性证据及下一步

| 问题 | 本轮结果 | 证据层级 |
|---|---|---|
| 分区最低额度少1B | 拒绝，未改变owner/cache | CPU fixture |
| 当前占用多1B | 主动加载前拒绝，旧CPU/GPU对象不变 | 实际worker入口＋计数fixture |
| 主动对象已暂存 | 再次主动加载延后；同一实际占用下按需加载可接纳 | 实际worker入口＋计数fixture |
| registered/staged共享storage | 只计一次占用，但不给独占registered抵扣 | inventory/allocator计数fixture |
| 多别名及合同变化 | 同owner一次安装；更改合同拒绝 | 控制器/进程身份测试 |
| 回归 | 935功能、59安全/计量、30安装版native环境CPU检查通过 | 非性能检查，数量有交集 |

测试日志在`results/ieee_tc/p2_backend_qualification/d58_20260927/`。
CUDA均未初始化；没有把fixture字节、离线上界或测试数量当成实测峰值、性能
重复或Full通过证据。尚未选择生产B/C，也未宣称分区最佳或所有多plan执行
均有进展保证。非LoRA占用增长仍按实际值检查，可能令后续配置不可行。

下一步转到实际native copy/HOST归还与代表性Full/profile资格，不再增加
同类离线allocator重复。尤其PyTorch在有stream依赖时通过event队列延后归还：
完成fence不自动证明统计中的allocated已下降，不能预扣或用flush绕过。
实际多plan、物理字节压力、数值正确性、完整池及真实remote仍需资格。
Full启动保护保留，共同vLLM须获得相同候选配置与合理优化机会。

## 2026-09-27 D59：真实checkpoint传输后的HOST生命周期诊断

### 执行前固定的问题和方法

D56只测CPU加载/移除，D58据此保护工作空间。本次新问题是：已参与异步H2D的
真实checkpoint对象，在copy完成且所有Python引用删除后，allocator是否已经
归还内存；如果没有，单独同步及下一次正常checkpoint加载分别会发生什么。
依据[PyTorch2.13源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/core/CachingHostAllocator.h)，
free可能只登记stream事件，统计读取不处理该队列；不能从D56的CPU结果推断
H2D路径立即释放。这是需要实测的机制假设，不是已得到的结果。

- 在现有`backend-host-check`增加显式`--host-copy-lifecycle`，不建新框架。
- 同一个新受限进程，实际读回D57的`uncached_v1`候选，未启用后台事件处理。
- 复用同六个既有内容/rank/modules类及其SHA；不下载、生成或改变权重。
- 每类依次使用官方`BaseLinearLayerWithLoRA.set_lora`和已实现的pitched copy
  strategy，加载同一真实checkpoint；四个独立slot、max-rank64、slot2与既有
  copy诊断一致。逐模块原始A/B尺寸不变。只构造隔离的GPU槽位，不加载backbone，
  不声称native registry、merged packing、推理数值或Full已经资格。
- 保存加载前/后、setter返回且持有HOST、fence后持有HOST、删除后、再次fence
  后、下一次正常checkpoint加载后/删除后八个状态，记录实际allocator、inventory、
  分配/释放次数和资源占用。setter返回时查询stream，未观测到在途便如实记录。
- 两条路径均保留source引用直到fence。核对全部slot内容、零padding和未用slot；
  这不是不同系统吞吐比较，不用其耗时定SLO或profile。
- 不插入sleep、不flush、不用dummy分配促回收；下一次加载是显式测量动作，
  不绕过生产准入后宣称Full可以持续运行。
- 一次诊断后先清理、制表和解释，再决定下一步。生产allocator/预算/公式不变。

### 第一次真实传输诊断结果

一个新受限进程完成六类、两条copy路径、96个状态点。43次外置资源采样，
服务峰值723,648,512 B，high/max/OOM均0；服务和watchdog退出0，实际CUDA
context释放且服务资源域移除。辅助域和本轮测试域均核对为空后停止。

| 原有工件 | rank | 普通copy删除后仍占pinned字节 | 再fence后的字节 | 主动copy删除后字节 | 普通copy下一次正常加载中归还块数 |
|---|---:|---:|---:|---:|---:|
| 3B code | 8 | 9,175,040 | 9,175,040 | 0 | 224 |
| 3B code_0015 | 16 | 18,350,080 | 18,350,080 | 0 | 224 |
| 7B code | 8 | 16,777,216 | 16,777,216 | 0 | 256 |
| 7B finance | 8 | 16,777,216 | 16,777,216 | 0 | 256 |
| 7B medical | 8 | 16,777,216 | 16,777,216 | 0 | 256 |
| 7B code_0015 | 16 | 33,554,432 | 33,554,432 | 0 | 256 |

所有删除后可达tensor字节均为0，故普通路径保留量不是遗漏的LoRAModel引用。
六类两路径的全部GPU slot内容SHA一致；主动作业copy额外GPU tensor峰值为0，
普通路径为49,152–131,072 B。这仍是隔离setter诊断，不是Full性能收益。
全部setter后查询都已完成，**没有测试DMA仍在途时删除source**，也没有用
sleep制造在途状态。最后一份未copy的checkpoint移除后各arm均回到0。

完整96行和来源/launch/watchdog SHA在
`paper_results/ieee_tc/p2_backend/20260927_host_copy_lifetime.{csv,json}`。
原始10.4MB文件留在gitignored结果目录；没有重复生成权重或负载。

这支持“传输已完成不等于allocator已经处理回收事件”的解释。D58继续保守
计入这些字节是正确的，但`uncached_v1`本身不足以保证主动准备不等待后续
真实分配；因此不直接冻结为Full工作点、不增加预算、不把残留字节扣掉。
下一项若继续此假设，只比较官方后台事件处理选项能否自主归还这部分占用，
使用相同工件/程序/资源边界；生产策略保持未选定。该问题不再重跑CPU-only
allocator测试或旧请求前缀。

## 2026-09-27 D60：官方后台回收的第二次、最终局部对照

执行前协议：复用D59的六类、两条copy路径、八个状态和完整slot核对，
仅在新的受限进程启用官方
`pinned_max_cached_size_mb:0,pinned_use_background_threads:True`。
使用已有命令的显式`--host-copy-background`，不修改生产allocator候选。
实际snapshot必须读回完整解析配置串及max_cached_size=0；此版本没有单独
background布尔字段，不杜撰该字段或把环境变量单独当成生效证明。

假设来自D59及官方PyTorch2.13的默认pool事件后台处理实现：删除后的真实
pinned占用可以在下一次checkpoint分配前归还。判据为原有删除后/第二次fence后
两个观测点的实际字节，结果不强制为0，也不插入等待直到得到期望值。
若仍有残留就完整报告；不flush、不dummy分配、不增加HOST预算。不测新的
性能矩阵，不把一次观察作为及时性上界、CUDA Graph/private pool或Full保证。
保持原始非后台结果不变；两次独立进程不是统计性能重复。

这是同一假设的第二次最小比较，此后不继续allocator局部循环。先交付对照表，
再把可支持的候选带回Full/profile及等待队列实际进展验证。后台字节下降不等于
Prime等待队列已被唤醒；所有容量检查仍必须使用当时实际占用。

依据：[PyTorch2.13官方allocator源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/core/CachingHostAllocator.h)、
[配置解析](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/c10/core/AllocatorConfig.cpp)、
[vLLM0.30官方copy路径](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/layers/base_linear.py)。

### D60观测结果与裁决

一次新进程完成96个状态点；全部slot和padding内容核对通过。42次资源采样，
服务峰值727,416,832 B，high/max/OOM均0；服务/watchdog退出0，实际CUDA
context及服务域已释放，辅助域实际进程清单为空后停止。没有新增权重或负载。

| 工件/rank | 无后台：删除后占用B（D59） | 有后台：删除后占用B（D60） | 有后台：再次fence后B | 两条copy路径内容一致 |
|---|---:|---:|---:|---|
| 3B code/r8 | 9,175,040 | 0 | 0 | 是 |
| 3B code_0015/r16 | 18,350,080 | 0 | 0 | 是 |
| 7B code/r8 | 16,777,216 | 0 | 0 | 是 |
| 7B finance/r8 | 16,777,216 | 0 | 0 | 是 |
| 7B medical/r8 | 16,777,216 | 0 | 0 | 是 |
| 7B code_0015/r16 | 33,554,432 | 0 | 0 | 是 |

表中字节为普通native copy路径；主动pitched路径在两次运行均为0。D60所有
setter后stream查询仍已完成，未测试在途删除，也不能由此给出回收时间上界。
完整CSV/JSON在`paper_results/ieee_tc/p2_backend/20260927_host_copy_background.*`；
原始SHA为`e54e7932bc30fc0e82afafe89878ae48774d9fc907434b88296b32efb464bda4`。

裁决：接受官方后台事件处理作为后续Full集成候选；不是SLO/性能胜出结论，
也不自动冻结正式模型配置。保留D59数据和原候选身份，不再增加同类microtest。
下一主线是候选的实际启动身份、固定预算与等待状态更新，然后代表性Full/profile。
字节回收必须被实际观测才能释放额度；尚未证明等待队列持续进展或完整模型资格。

### Full候选接入（不是正式配置冻结）

现有runner、dedicated worker和native readback支持显式
`ieee_native_host_allocator_policy: uncached_background_v1`。旧`uncached_v1`
保持原语义，默认仍未选择；两个身份不能继承混用环境或旧profile。
后台策略没有改变exact-size常驻/工作空间上界，仍用D58分区，在实际占用未下降
前拒绝新的不合预算加载；不能以“最终会回收”为理由预扣。Full启动保护不变。

| 集成检查 | 判据 |
|---|---|
| 新鲜子进程/配置身份 | runner、worker及实际native解析配置一致 |
| 老环境/旧profile | 不默默升级，冲突拒绝 |
| 后台回收但占用未下降 | 仍按真实字节延后，预算不增加 |
| profile/Full资格 | 本次没有获得，下一步实际验证 |

源码审查发现现有movement唤醒来自请求引用释放、准备目标结束、plan关闭等；
这些事件不能替代对异步allocator归还的实际观察。后续需在原有状态更新路径
中处理该状态变化并验证等待进展，不能靠循环重试或人为分配触发回收。
本轮只接入有真实证据支持的allocator候选，没有假装该后续问题已解决。

最终检查：938功能、22安装版native环境CPU、62安全/计量检查通过，0失败/
错误/跳过；CPU检查没有初始化CUDA。测试数有交集，不是独立性能重复。
完整96行与原始计数独立核对，D59/D60的12组GPU内容SHA也一致。旧147项
保护清单和权威计划零变化，所有本轮进程及资源已清理。

## 2026-09-27 D61：真实 HOST 容量变化到准备队列的闭环

D60证明后台事件处理可以归还实际pinned占用，但native allocator没有向
Python准备队列提供归还回调。仅等待新请求、其他transfer结束或plan关闭，
不能保证这些事件之后发生的容量归还能让既有任务继续。此次补齐状态传播，
不是增加另一种allocator策略，也没有继续做隔离allocator微测。

| 论文规范语义 | 当前实现及证据 |
|---|---|
| 执行时按当前容量重新检查 | 延后记录保留实际拒绝时的占用；重新执行仍经过原budget/reference/admission检查 |
| 实际释放而非预测释放 | 只接受同物理owner、同单调时钟域的原生占用；删除Python对象不产生可用容量信用 |
| 共用冻结控制周期 | 只在既有control采样时检查，且仅检查存在HOST字节压力等待者的owner；不新增定时器或无条件重试 |
| 已撤回/失效的计划不继续准备 | 异步读取后核对仍存活的拒绝attempt；取消中的旧状态不能唤醒新任务 |

令A为实际总tensor占用、X为独占已注册tensor占用。D58保护的剩余workspace
与A−X相关；只有A−X下降才通知workspace等待者，A和X同步下降不算余量改善。
普通总字节预算等待者则使用A下降。这里是既有分区检查的等价状态比较，
不改IEEE九式、不增加B/C、不预留通知中观察到的容量。唤醒后若容量又变紧，
原执行器仍可正常延后。每个等待owner每次control采样最多一次snapshot调用；
没有对应等待者则不调用。真实调用开销尚需后续完整回放/S10测量。

### 即时正确性状态表（CPU检查，不是GPU性能实验）

| 检查条件 | 观察结果 |
|---|---|
| 实际占用不变，连续三个控制检查 | 无额外checkpoint加载 |
| 占用下降，但尚未传播状态 | 等待任务没有凭空完成 |
| 实际余量改善，传播后重新准入 | 原HOST准备完成，无需新请求触发；GPU原slot不变，文件/tensor引用释放 |
| A和X等量下降 | 不误判为受保护workspace增加 |
| 同owner两个等待任务 | 一次snapshot覆盖，只有实际余量改善才重新执行 |
| owner/clock不匹配或实际字节未知 | 拒绝该观察，不产生唤醒 |
| 读取期间撤回全部等待者 | 不唤醒，不生成容量通知记录 |
| 同一控制时间重复进入 | 只在原冻结周期允许的一次采样中刷新 |

结果metadata记录owner、实际A/X、原拒绝attempt ID、观察时间及
`capacity_reserved=false`。这些记录证明状态传播，不充当成功加载或物理预留。
新增五项检查包含现有runner、队列、文件引用和native owner执行路径；
容量变动由CPU fixture控制，**没有假称来自新的GPU测量**。

最终943项功能检查、16项已安装vLLM0.30/torch2.13环境CPU检查、62项系统
Python安全/计量检查通过，无失败/错误/跳过；CPU检查CUDA未初始化。
日志在`results/ieee_tc/p2_backend_qualification/d61_20260927/`；计数有交集，
不是独立实验重复。Full guard保留，正式模型配置/profile/B/C未选择。

结合D59/D60及再次核查的
[PyTorch2.13 allocator源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/core/CachingHostAllocator.h)、
[vLLM0.30 worker源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)，
本项接受为完整加载路径的候选正确性修复。下一步回到代表性profile和Full
多计划/生命周期资格，不增加同类microtest或重复旧请求前缀；本次不能推出
实际Full持续进展、正确adapter数值效果、共同SLO达标或优于baseline。
## 2026-09-27 D76：代表性 profile 的输入覆盖核对（离线，不是新模型实验）

在 D75 远程完整性检查运行期间，只读取既有冻结内容索引和源 trace；没有
启动 GPU、基线或另一组远程下载，也没有修改正在执行的检查代码。
重新核对已安装版本对应的
[vLLM 0.30 worker manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)：
文件读取、CPU LoRA 对象和 GPU activation 是不同路径。既有 D26 source32
只有 28 次 GPU、4 次 native HOST 观察，且 16 次 first-touch 在测量前预热；
不能据此填入 Remote、文件 HOST、NVMe 或并发类别的准备/服务初值。

| 既有输入覆盖 | 3B | 7B |
|---|---:|---:|
| 完整池逻辑 ID |500|500|
| 精确文件树内容类 |24|6|
| 开发期原 trace 前 1,000 请求实际 ID |60|60|
| 此前缀出现的内容类 |21|6|
| 还需由静态工件清单覆盖的内容类 |3|0|

3B 缺失三类的字典序首个既有 ID 为 `support_lora_0148`、
`research_lora_0104`、`finance_lora_0073`。它们来自已冻结的 500-ID 静态清单，
不是读取后续请求预测热点；后续受控准备测量可使用这些既有工件和既有开发
prompt。不能重新跑相同短前缀后宣称覆盖已补齐，也不能以 rank 相同替代精确
内容类。这里只确认输入覆盖，不代表已获得任何延迟样本或冻结生产 profile。

核对依据：`inputs/README.md` 指定的 3B remote、7B materialized 索引；
两源 trace SHA 分别为
`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`、
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
本次读取时重新校验一致。只在内存中建立 ID→类索引，没有复制完整 trace。
独立训练权重数量和全零权重限制不变；内容类数量不是模型多样性结论。

## 2026-09-27 D82：分层原生测量入口与来源生命周期修正

沿用 `ieee_tc_preflight.py backend-model-check`，增加显式
`native_source_matrix --source-profile-spec`。小型 JSON 只索引原 trace 的
request ID、静态 adapter ID、来源层级和资源配置，不生成 prompt、权重或
新到达负载。测量重用 Full 的来源分类、保护、加载、原生 token 事件和释放
路径；没有 router、虚构初始成本或 Full 资格豁免。五种来源为 Remote、
NVMe 文件、tmpfs HOST 文件、native HOST 张量和 GPU。

每个受控 wave 前的建态和驱逐单独记录；真实测量不预先消除 Remote miss。
并发波次记录实际接纳人数，不靠强制等待把所有请求伪造成同一并发类。
兄弟请求全部 join，包括有请求失败的情况。已完成来源变成更快/不同表示时
不能重贴标签。HOST 文件必须确实在 tmpfs，原始池始终只读，退出只处理本轮
拥有的临时目录。代表性覆盖和 profile 冻结仍需后续实测，不由此入口自动保证。

连续 Remote→NVMe→HOST 文件的 CPU 集成检查发现：原 native owner 将
`(adapter name, path)` 终身绑定；即使旧 CPU/GPU 对象已完全驱逐，同一 adapter
换到另一合法层级仍被拒绝。该失败发生在加载前，不是 GPU 性能问题。
修正将逻辑身份与物理副本生命周期分开：不同 name 仍永远不能复用 integer ID；
已有 cache、引用、staging 或准备目标存在时仍不允许换路径；只有完全退休后，
同名工件可由控制器持有的 SHA-confirmed 新文件来源重新加载，新对象获得新
incarnation，旧 lease/epoch 仍不可重用。IEEE 公式、路由排序和驱逐策略不变。

历史依据是 D26 单一来源短测量未覆盖此转换；代码沿革
`774bd7f`→`e2ba9f6`→`3297289` 保留。再次核对
[vLLM 0.30 原生 worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
其 LRU 对象的装入、复用和替换本来具有不同生命周期。本次不删除身份保护，
也不把路径字符串等同于工件内容身份。

| D82 检查 | 结果与边界 |
|---|---|
| 五类来源的 actual admission/event collector | CPU fixture 全部通过；无真实 GPU 延迟结论 |
| 并发中一个输出错误 | 保留失败，另一请求完成，两者引用释放 |
| 同 adapter 驱逐后 NVMe→HOST 文件 | 修正前确定失败；修正后集成路径通过 |
| 存活 GPU/HOST、staging、计划持有、错误 name、旧 epoch/lease | 仍拒绝非法重绑定 |
| owner/lifecycle/service/preparation regression | 278 项通过，3.680 秒 |
| 系统 Python preflight | 55 项通过，0.982 秒；输出前段非完整捕获 |
| 原有 offline basic smoke | 288 项通过，22.170 秒；6/8GiB受限CPU域 |

原始日志位于 `results/ieee_tc/p2_backend_qualification/d82_20260927/`。
此前测试 launcher 遗漏工作目录、编辑中重复函数声明和用 conda Python 跑
pidfd 检查均失败，不能算通过；后续修正测试条件，不放松安全门槛。
spec-only 导入的512MiB试验资源域发生回收停留，43秒时主动停止；改用已验证的
CPU测试6/8GiB域后7.717秒完成。没有启动模型，没有主机 OOM。
一次 basic smoke 启动遗漏 offline 环境，dummy-model HEAD进入外网等待；
74.767秒时停止本实验单元，保留终止记录，不算通过。上述288项来自随后明确
`HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1` 的完整运行。

下一项是现有 3B 原输入中的 `req_00000/finance_lora`（rank8）与
`req_00015/code_lora_0015`（rank16），五来源、两路 wave 的十次原生链路 pilot。
`20260927_3b_source_profile_pilot_spec.json` 固定共享 HOST16GiB、每 native
owner2GiB、NVMe16GiB，仍在推理服务72/80GiB内；保留现有32个 CPU entry、
8个GPU slot、服务并发8、共享移动并发3。这是公开候选配置，不是正式最优配置。
根据已核验 workspace 下界，该 native 额度可容纳32×18,350,080字节常驻与两份
55,080,136字节加载峰值；实际 worker 的基线占用仍须在安装预算时重新检查。
pilot只验证集成，不宣称覆盖全部24个内容类、完整服务分桶或性能优越性。

### D82 attempt1：实际进程导入暴露的配置覆盖（尚无GPU请求）

| 检查 | 观测 |
|---|---|
| 失败阶段 | engine_initialization，0条推理请求 |
| 原因 | 历史 runner 导入无条件写入两个 allocator 环境别名，覆盖启动时的显式候选 |
| 退出 | service=2、watchdog=0；原生GPU上下文释放已确认，服务域已移除 |
| 文件 | 本次两个受管层级临时目录已删除；原始池、只读远端发布缓存未改 |
| 性能结论 | 无；不能作为任何来源层级的时延样本 |

追溯到 `b68eaeb` 的历史默认赋值在 torch 导入后执行；D60/D61 的独立
loader检查没有经过该完整入口，既有单元测试也在已导入模块上更换环境。
现修正为：在 torch 导入前，仅在启动器没有指定任何 allocator 设置/候选时
应用历史默认；显式配置及别名冲突原样保留，交由原有配置/实际 native readback
检查拒绝。不能在已运行的进程中重新配置分配器，不移除保护。
源码依据为 [PyTorch 2.13 CUDA allocator](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/c10/cuda/CUDAAllocatorConfig.cpp)
对加载时和运行时后端身份一致性的检查。新增真正子进程导入回归，避免只测
辅助函数而漏掉模块顶层副作用。此修正不改变IEEE九式、缓存容量或路由策略。

修正后72项启动/worker检查通过（11.230s），原有288项offline smoke通过
（21.876s）；147项历史保护检查无变化。attempt1最高服务内存1,424,781,312B，
67次采样的high/max/OOM均为0，最小主机可用115,999,780,864B。
下一次使用全新attempt2路径，原attempt1及全部失败证据保持不动。

### D82 attempt2：配置字符串不是当前配置状态

| 检查 | 观测 |
|---|---|
| 启动 | 3B真实模型、FLASH_ATTN/CUDA Graph完成初始化 |
| 拒绝点 | 首次worker状态读取；尚无LoRA请求或remote fetch |
| 原因 | 本项目将最近一次allocator设置字符串误作完整生效配置 |
| 内存 | 91采样，峰值10,657,054,720B；high/max/OOM均0 |
| 退出 | service2/watchdog0；原生GPU释放已确认，服务域/层级临时目录移除 |

核对 [vLLM0.30 Worker._scoped_allocator_max_split](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu_worker.py)：
模型加载前后设置GPU分配器max_split。
[PyTorch2.13 parseArgs](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/c10/core/AllocatorConfig.cpp)
只重置列出的GPU参数；未指定的pinned上限和background选项继续保留。
[snapshot输出](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/csrc/cuda/memory_snapshot.cpp)
中的`PYTORCH_CUDA_ALLOC_CONF`只是`last_allocator_settings`，不能与启动环境
字符串逐字比较来证明完整有效状态。这是观察器缺陷，不能据失败推断HOST缓存
实际已开启，也不应为迎合观察器在模型运行后重设整个allocator。

修正仍要求完全一致的启动环境/模型策略、Torch版本和实际typed
`max_cached_size==0`，其验证范围明确为无持久pinned缓存；原始最近配置字符串
继续记录。Torch2.13没有导出background布尔值，记为requested、readback=null，
不能声称单独读取到了该值。背景归还的进展沿用真实容量事件/引用完成检查，
不会假定立刻释放，也不会按未来腾出字节提前准入。GPU/HOST安全预算不变。
新增回归验证GPU-only参数更新不造成假拒绝、非零/缺失上限仍然拒绝。

420项worker/owner/原有smoke检查通过（22.388s）。随后在已安装原生
Torch2.13/vLLM0.30调用官方max_split上下文：前/中/后的实际cache上限均0，
最近配置字符串依次为启动策略、max_split20、恢复值；修正后的观察器均通过。
三个观察点`cuda_initialized=false`，无模型、请求或张量分配，不是额外性能
测量；原生程序退出0，服务cgroup实际为空，high/max/OOM均0。
