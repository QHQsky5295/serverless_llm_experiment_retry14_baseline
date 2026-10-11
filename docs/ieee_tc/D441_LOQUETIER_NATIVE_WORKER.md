# D441：Loquetier 原生服务接入与共同计量

日期：2026-10-12。接续 D437–D440；本次是必要兼容、正确性、计量和安全
适配，不是 baseline 性能优化，不是新的 GPU 性能结果。

## 问题、依据与边界

既有容量实验只证明一份 125-ID 静态分片及 batch2/context1024 KV 可以装入
GPU；不能证明请求执行、四进程 HOST 峰值、数值正确性或共同生命周期计量。
D440 的 HTTP 合同也不能代替实际原生 worker。此次假设是：保持原生加载、
静态映射和生成流程，接入共同接口后，可以逐请求核验原生绑定与完整计时。
如果真实 forward/完整池失败，保留失败，不以改 batch、缩池或换后端补救排名。

核对了完整计划、当前状态、冻结 V1/单轮 V2、最新基线保真要求，以及 D438
容量入口、D439 在线队列、D440 HTTP 修改历史。联网核查
[官方仓库](https://github.com/NJUDeepEngine/Loquetier)，保留其虚拟模块、
MixedLoraModel、ModelGeneration 和 SMLM。具体源文件依据本地固定的
`aae33baeeb19777129c1ccbff99a898d4a0e2c63` 与三个历史兼容补丁，
不把网页访问说成对本地每个字节的验证。

用户再次批准一次性只读压缩交付缓存：继续复用 D78/D80，不重复发布、
不新增解压工件池。真实读取、传输、竞争和本地物化计入服务；离线打包不计。
本次未操作远端，也未下载模型或工件。

## 实现及其不变量

沿用现有 `replay_vllm_engine_trace.py`，没有新建回放框架。

- 增加明确的 Loquetier native CLI、配置、来源检查。完整 500-ID 仍按 D437
  的静态字节分片，四份各 125 个；不读取请求热点，不做自适应集群调度。
- 原生解释器、内核、源码视图、配置、当前服务限制先检查，再导入/加载 GPU
  模型。每个 worker 只看到一张卡；Python3.9、torch2.1.2+cu121、FP16、
  batch2/context1024、apply_scaling=A 不变。
- 复用 D438 加载顺序，原始 PEFT 表示与准备后表示保留共存；不偷偷改为
  offload/dedup。InputRequest 使用共同 prompt token IDs 与固定目标长度，
  保留原生 add_requests/generate 及整数 token 边界。
- 共同入口在完整工件准备前开启，准备发生于共同部署通知之后。准备结束后
  顺序启动四个 worker；请求等待其预定分片，不被改送到其他 ready 副本。
- 原生 registry/path/index、实际 PID/start ticks、内核及来源形成 ready 收据；
  父服务还检查 loopback health 和实际 GPU census，之后才释放排队请求。
- 原生初始化/forward 致命错误保留日志并退出；不隐式重试、不换后端。
  worker 退出通知明确不等于 GPU 释放。父服务仍使用既有实际进程退出、
  census 和物理 lease 检查，不改 GPU-s 并集算法。
- S-LoRA/vLLM 原路径保留；整池准备函数对 S-LoRA 的默认调用接口不变。
  未知数值正确性字段仍为未知，接口成功不替代数值验证。

## 环境缺口与最小处理

原 Loquetier 环境没有 aiohttp；CPU-only import 检查在任何模型加载之前
得到 `ModuleNotFoundError`。这不是 GPU 性能失败。采用与现有公共外层环境
相同的 aiohttp3.13.3；其[官方项目声明](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/pyproject.toml)
支持 Python≥3.9。仅新增八个缺失的传输依赖，具体版本及 wheel SHA 见
`configs/ieee_tc/loquetier_http_py39_additions.txt`。先 dry-run，再
`--no-deps --only-binary=:all: --require-hashes --no-compile`，不升级既有
torch/Transformers/PEFT/内核，不编译，不改变基线机制。
原 attrs26.1.0、idna3.15、typing_extensions4.15.0 继续复用。
dry-run/install 回执保存在 D441 原始目录，安装后检查另见结构化 summary。

## 已完成验证与证据限制

- 接入 14 项 worker/配置/共同父服务检查。真实 loopback 验证准备前接收请求、
  整池准备先于持卡、顺序启动、静态目标、错误进程拒绝、退出后释放的调用顺序。
  其中 worker、GPU census、lease 明确是 synthetic，不能算实际 GPU 证据。
- 公共回归 208 项通过，10.103 秒；包括此前 ServerlessLLM/vLLM/S-LoRA
  路径及 Loquetier HTTP/在线/worker。预期的致命 forward 与取消连接错误
  来自负面测试，不是隐去的失败。新增 14 项单独检查也通过（1.746 秒）。
- 既有原生环境 26 项实际原生 CPU/在线边界检查通过（2.263 秒），无 skip。
  这发生在补 HTTP 依赖之前。安装后原生 worker/HTTP/在线/原生 CPU 共
  49 项通过（4.426 秒），无 skip；固定配置、源码视图和实际 HTTP API
  检查通过，六个既有包的版本与内核 SHA 均保持不变。
- 保留两项检查入口错误：把仅供 Python3.12 公共父服务的两个测试也放入
  Python3.9 后，51 项中一个父服务测试得到 502。源码定位到公共入口的
  `asyncio.timeout`（Python3.11+）；冻结部署本来使用外层 Python3.12、
  原生 worker Python3.9，未改公共入口以绕过它。两个父服务检查已在实际
  外层版本的 208 项中通过，原生 49 项不含这两个外层测试。另一次静态
  API 检查误把 `handler_cancellation` 查在 RequestHandler；实际参数属于
  Server，改正检查对象后通过。均非性能 attempt，不掩盖、不改算术代码。
- 更早 196 项开发回归通过（10.241 秒）保留身份；以上都是代码检查，
  不是同一性能键的重复运行或择优。
- 测试 GPU 隐藏，CPU2,3,26,27、memory.high/max=3/4 GiB、swap=0；
  各辅助域按实际空进程树及 InvocationID 逐一收尾，详见 summary。
- 旧 147 项保护清单不变，三份用户修改文件、Loquetier 三补丁、S-LoRA
  十补丁 SHA 不变；未动旧结果、主表、图、论文或冻结指标。

## 固定资产与下一步

小型 source view：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d441_20261012/source_view`。
manifest SHA `2d8e801033a4f3b4062b94dbaf34e6c59c1613f4e2427705091435c6f9e82bb7`。
使用现有 `prepare-loquetier-measurement-view` 入口生成；未覆盖旧 D438 视图。
worker 配置见 `configs/ieee_tc/loquetier_7b_static_native_v1.json`，分片沿用
D437 SHA `049441391b67e500328546c3c8e1001fd1ce197f0c1a93e150fc26e8bdfe4a77`。

接下来仍需：冻结实际 HTTP 回放/远端/环境证据、核验 Loquetier 原生分析规则，
fresh safety gates 与代码备份，然后 100 请求 → 500-ID 功能覆盖 → 原 1000
兼容 → 单轮 4000。真实 forward/数值资格、四 worker 资源及完整 W0 尚未通过。
不重复已经通过的 D438 容量检查或封存的四系统 W0。

剩余 7B/HydraServe/ElasticLocality、W1/W2/G2、必要 PrimeLoRA 优化、3B、
A1–A5、S1–S13、审稿设计文档与后续图表仍 OPEN；完整计划未缩减。
