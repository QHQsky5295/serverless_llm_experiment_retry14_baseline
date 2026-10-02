# D172：原生 RPC 状态解码候选资格

2026-10-02。开发候选，非正式性能结果、非 7B 验收。
遵循 PrimeLoRA-PLAN 当前版本与冻结指标 V1；只推进 Prime 7B。

## 问题、证据与唯一假设

D171 使用 D164 完整回放与 D170 暂定 warm 阈值，得到联合时延达成上界
52.75%；1547/1799 个 TTFT 超限请求在 engine dispatch 前已耗尽期限。
正确性、共同 reference 冻结和 G1/G2 仍未完成，不能将该上界称正式 SLO。
D162 当前配置的 CPU 帧诊断中，538 个 controller 业务样本有 71 个位于
RPC JSON 解码；这是出现次数，不是 CPU 时间占比。D161 的小 generation
终态回复不能代表所有控制回复大小。D169 保留的实际状态快照编码后约
1.5–4.3 MB，提供了独立于新 GPU 回放的组件输入。

可证伪假设：降低事件循环中同步状态解码的 CPU 工作，可以减少请求提交前
的控制等待。若组件变快而同合同 Full 不改善，不能据此宣称服务收益。
不把所有排队/准备时间归因于 JSON，也不重复已完成的 D151/D153/D159/D163 优化。

## 官方依据与改动范围

- [Python asyncio 官方说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)：同步 CPU 工作会阻塞其他异步任务。
- [vLLM 0.30.0 serial_utils 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/serial_utils.py)：使用 msgspec/MessagePack；借鉴高效序列化原则，本文候选**仍保留 JSON wire**，不声称移植了其 IPC。
- [msgspec 官方类型说明](https://msgspec.dev/supported-types)：核查整数、浮点和 Unicode 语义；实际部署版本另由本地环境记录。

只在现有 runner 增加 `_decode_native_rpc_frame`，替换 native response 与
native progress-frame 两处解析。直接从 bytes 解码，不先构造整份 Unicode。
native 环境已有 msgspec 0.21.1，没有安装包或修改后端版本。

保留原 stdlib JSON 编码、换行 framing、8 MiB 限制、timeout、channel、worker、
状态所有权/epoch/物理 allocation 图、取消及原生首末 token 校验；legacy
backend 保留原 decoder。没有删字段、类型投影、跨请求缓存 confirmed 状态、
异步校验旁路、兜底 parser 或 null 替换；九个公式与配置保持不变。

边界差异明确披露：新 decoder 拒绝 JSON 非标准 NaN/Infinity 和超范围浮点；
不会像旧 stdlib 那样接受非有限值。当前有效状态合同必须有限；不是对所有
stdlib 可接受字节串的无条件等价宣称。非法 UTF-8 仍抛 UnicodeDecodeError。

## 组件测量表

复用 D169 四份 `sources_before`，以现有 sender 方式包装编码；每份三轮交替
顺序、每轮每 decoder 20 次。完整类型/字典顺序/嵌套字段及浮点 bit 表示一致。
三轮均值如下；单位 ms，越低越好。全部 24 个原始行另存 CSV。

| 快照 | wire bytes | 原解码 | 候选解码 | 均值相对下降 |
|---|---:|---:|---:|---:|
| 0 | 1518630 | 48.480109 | 14.245557 | 70.62% |
| 1 | 2215094 | 38.661592 | 14.967994 | 61.28% |
| 2 | 3609914 | 58.167186 | 27.866608 | 52.09% |
| 3 | 4304329 | 59.590223 | 45.739169 | 23.24% |

这是 CPU 组件样本而非三个独立 workload/seed；不计算服务 CI，不按该比例
外推 TTFT、SLO 或 GPU-s 收益。按计划 §11 用精确表交付，不制造系统性能图。

## 正确性与资源资格

| 检查 | 结果 | 说明 |
|---|---|---|
| tests1 | 54 项，1 error | 非法 UTF-8 的测试错误地预期 JSON DecodeError；失败源码与日志保留 |
| tests2 | 55 项通过，0.862 s | 修正测试异常类型，并显式核对原 stdlib 的同类异常；服务代码未为此增加兜底 |
| regression1 | 849 项通过，152.670 s | 原 file/planning/request/launch/native 回归加 decoder/service-event 覆盖，计数与针对性测试重叠 |
| 四份快照 | 全字段/类型/浮点位一致 | 不重复 GPU 测量，不生成新权重或 trace |
| CPU 包络 | 3/4 GiB、swap 0 | CPU 2,3,26,27；六个实际任务均已退出，final memory events/swap 全 0 |

针对性检查在 native 环境 msgspec 0.21.1；扩展回归复用 CPU 环境 msgspec
0.20.0。两环境身份不能混称相同。扩展回归命令 wall 164.89 s，RSS 峰值
1202000 KiB；组件测量 wall 33.15 s，RSS 1091776 KiB。

磁盘资格：复用 D152 的公开下载缓存审计，仅删除经官方 PyPI SHA/大小/链接、
进程和项目引用核查通过的 vLLM 0.15.0 wheel 下载缓存及 HTTP header；
释放 509206528 allocated bytes。已安装环境、编译缓存、权重、trace 和唯一
原始数据未动；审计及删除回执保留。推理机 150/100 GiB 门槛未改变。
远端 D78/D80 一次性交付缓存直接复用，本次无远端操作或重新生成。

## 决定与下一步

候选通过最小正确性与组件资格，**暂不认定服务优化被接受**。备份后仅进行
一次普通 7B W0 Full 4000 验证，沿用 D164 cap4/D157 profiles、同输入/生成、
60 秒准备、真实已发布工件、资源包络和 SLO 定义；不加 profiler、第二项优化
或新调参。比较 D164 完整请求阶段、尾延迟、TPOT、物理 GPU-s 与暂定时延
达成，并保留退化和失败。数值正确性、共同参考、Resident、旧 Prime 新指标
比较和 G1/G2 仍为 OPEN；7B 未验收前不推进 3B 或外部 baseline。
