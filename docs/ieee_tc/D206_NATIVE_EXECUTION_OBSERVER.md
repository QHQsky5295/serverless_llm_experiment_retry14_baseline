# D206：实际执行边界采集器的 CPU 资格

日期：2026-10-03。状态：CPU 资格完成；尚无本次 GPU 观测。
这是 D205 映射检查器的采集端，不是新的性能优化，也不是数值正确性结论。

## 问题与范围

D203 的 4,000 个请求均完成原生固定输出合同，但与 D195 有 126 个输出 hash
不同。D204 已核实逻辑提交 ID，D166/D167 已验证短前缀的槽位内容；这些证据
没有覆盖实际每轮前向使用的 token→slot 映射。不能由上述检查推出差异无害。

联网核对了 [vLLM v0.30.0 原生 runner](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu_model_runner.py)：
模型执行入口位于 forward context 内，随后单独调用 logits 计算。
据此把采集边界放在这两个入口外侧，不假设 Python LoRA 算子包装能观察
CUDA Graph 的重放。安装环境源码另做 SHA 绑定；网页解析行号不当作本地行号。
Punica 原始文件的本次联网请求失败，使用已安装源码及 D205 的已封存源码身份，
不声称此次已经联网取得该文件。

## 本次改动

复用 `gpu_monitor.py` 的 worker extension 和既有 observation RPC，增加显式
`execution_observer` 命令。必须声明 `qualification_only=true`、使用 barrier，
不能同时混入槽位内容审计。默认参数为 None，普通 Full 不安装任何执行包装。
没有修改 site-packages、调度、准备、admission、公式、配置、SLO 或生成合同。

- `start`：warmup 后、scheduler 已排空的隔离诊断安装；只支持 vLLM 0.30.0、
  dense、TP/PP/DP=1、无 speculative/ubatching、原生 token 输入。
- 包装成功的 `begin_use/end_use`，保存独立的 backend request ID、adapter ID、
  lease 与 owner；不能从正在观察的 batch 反推“正确答案”。
- 包装 native `execute_model`、`_model_forward`、`compute_logits`，保留原返回值。
  每轮记录 scheduled requests、真实 token 数、padding、graph mode、stream、
  forward/logits 前的实际 CUDA metadata，以及同步后的返回或错误。
- 记录实际执行模型树内的 LoRA module 身份、Punica wrapper 与 A/B buffer 指针；
  每次重新比对，拒绝脱离模型树、替换模型/管理器/缓冲区或换 owner thread。
- 采集 token/sampler 的映射、排序、group counts、starts、active slots、no-LoRA
  和实际 specialize/default grid 选择；D205 检查器负责离线验证这些原始值。
- 至多 64 个请求绑定、显式上限不超过 10,000 次执行、32 MiB 序列化事件缓冲；
  到达上限报错并保持 failed，不能静默截断。`read` 排空事件但保留累计计数、
  sequence 和失败状态；`stop` 只在无活动引用时还原原方法。
- 安装失败回滚；停止前先检查所有包装的所有权，避免部分解除。

采集器有同步与读回开销，所以 `timing_qualified=false`。它尚不验证 kernel
算术、输出 logits、采样正确性、全部 500 个逻辑 adapter，且 `n_correct=null`。
即使实际 GPU 映射检查通过，也不能据此把 126 个输出差异自动解释为舍入误差。

## CPU 检查结果

| 检查 | 结果 | 能支持什么 |
|---|---:|---|
| 第一版：worker/reference/smoke | 494/494 通过 | 第一版采集和原有引用回归 |
| 补齐模型树绑定与安装/移除检查后 | 496/496 通过 | 最终源码的 CPU 资格 |
| 最终新增 observer 测试 | 14/14 通过 | 两个执行边界、外部绑定、失败保留、资源边界、RPC |
| 最终测试用时 | 20.416 s | unittest 本身；不是推理性能 |
| GPU/数值/Graph replay 资格 | 未完成 | fake device 的 FULL 标签不是实际 Graph 证据 |
| 新 Full 性能回放 | 未运行 | 本次不产生速度或资源改善结论 |

两次检查都在 CPU 2,3,26,27、memory high/max=3/4 GiB、swap=0 的独立资源域；
按序结束，无并发 GPU 工作。终态 memory events 与 swap 均为零，资源域已释放。
第一版不是失败实验；第二版加入实际模型树绑定、原子移除检查与两项测试。
测试组重叠，不能相加当作独立重复。原脚本、日志和两版源码快照均保留。

负例涵盖未绑定/错误 ID/错误计数、错误 wrapper/权重/metadata 指针、脱离模型树、
漏 logits、原生异常、容量耗尽、重复 begin、活动时移除、不同线程、compiled bypass
和 microbatch。故意损坏的 group 数据原样保留，由 D205 离线检查器拒绝，不在
采集器内“修正”。没有放宽 D205、数值容差或物理安全检查。

精确资源、SHA、原始来源与保护检查见同名前缀 curated JSON、CSV 和证据包。
按执行计划的绘图规范，本节采用资格状态表，不生成没有实际性能数据的提升图。

## 下一步与不能关闭的事项

下一次继续现有 preflight 的短前缀资格入口，不建新回放框架：在排空后安装，
逐请求读取，绑定外部 lease/adapter 与事件 sequence，验证每个真实 forward/
logits 轮次，保存缺失/错误，不仅检查一个快照。集成后仅做对应轻量检查，再
执行一次既有 22 请求、四种内容类别的实际 GPU 诊断；不重复 D205 CPU probe。
复用 D166/D167 槽位内容证据，不新增工件、prompt 或宽松数值参照。

GPU 之前先审计可重建缓存并满足原 150 GiB 新任务门槛，本次没有删除任何数据。
此隔离诊断用既有本地工件，不属于真实远程性能实验，不需要启动 174 服务。
一次性远端交付缓存授权已由 D78/D80 完成，本次不重建。

7B 的数值正确性、126 个输出差异、共同 warm/Resident、旧新 Prime 的实际 G1/G2
验收仍开放。之后才是 3B，最后恢复基线、M1/M2、A1–A5、S1–S13。
不得把 CPU 资格、完成请求或胜过上个慢候选写成系统达标。
