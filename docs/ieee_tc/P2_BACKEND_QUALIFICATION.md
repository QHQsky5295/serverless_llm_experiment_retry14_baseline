# P2：共同 vLLM 后端资格（进行中）

已通过两模型各 100 请求的顺序原生合同检查；并发、完整模型/池资格与性能资格仍未完成。
旧环境与旧结果不覆盖。

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
