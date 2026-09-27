# D85 — Full 初始化绑定与校准入口一致性

日期：2026-09-27。基于代码 `26ee1220b6ca49158f111afc8418308a7a15b456`。
这是进入 Full 集成前的实现检查，不是新性能实验、正式 profile 或独立审查结论。
项目禁止子代理，因此 `experiment-audit` 的独立审查流程未执行；以下为可复核的
本地代码、配置和确定性测试证据，不标注独立审计 PASS。

## 已完成的测量能说明什么

D84 的 3B 368/368、7B 92/92 已完成、清理、分析和绘图，证据已推送 `26ee122`。
其中分别有 8/2 条显式 warmup，不能当成独立重复。实际加载阶段的层级差异已可
测量，但 HOST tensor 的 admission-to-acquisition 等待仍可能抵消加载本身的节约；
未据此宣称 Full 的端到端优势。

原始 D/T/O 与 d、五种来源、原生 token、实际子进程及物理 GPU 生命周期保留。
这些运行的配置不含 `ieee_admission_profile`，因而不是启用完整主动准入的运行。
输出长度的已完成观测可以复用为开发初始化；性能时延不能仅改配置标签后复用。

## 必须一起确定的配置边界

| 项目 | 可复用依据 / 当前事实 | 进入新测量前的要求 |
|---|---|---|
| 模型和后端 | D84 实际 child `model_config`；3B cap8/slots8/C32、7B cap2/slots4/C24；vLLM0.30、TP1 | 保留实测的 eager、dtype、scheduler、allocator 和资源身份，不用父描述替代子配置 |
| 原生完成长度 | `20260927_d84_completion_length_audit.json`；3B 两桶均值 98.375/171.69230769230768，7B 121.33333333333333/256 | 按真实完成事件初始化，不用 future target、桶外平均或猜测零 |
| 输入分桶 | scheduler 用含特殊 token 的 `num_prompt_tokens`；service collector/router 当前均用 canonical content tokens | 分别声明两个域；759 内容 token 可对应 760 原生 token，不能把两个计数混用或隐藏 tail |
| 完成/需求窗口 W | 论文两处采用 `(t-W,t]`；当前 HotnessTracker 可从旧控制配置推导 | Full 显式绑定同一 W，禁止无意继承旧 `auto` 或把测试 fixture 的 10 秒当已选参数 |
| transfer limit | D84 owned movement 为 3，历史两模型正式配置也为 3；D81 四并发交付已完成 | 与 native transfer observation、shared file domain 和实际 movement owner 一致；不再重复 D81 |
| 原生准入初始化 | 必需 `window_s/model_backend_id/profile_id/profile_means/transfer_limit`，当前 D84 无此对象 | 从已验证长度证据组装，并记录其 SHA、实际最终配置；不能把 ID 文本当测量真实性证明 |
| 服务与准备 profile | D84 分别观察 40/20 service classes、96/24 exact-content preparation classes | 完整绑定 model/config、环境、资源、输入 SHA；只使用实际已观察类；不允许近邻或零成本补类 |
| 更新与路由 | service/preparation `ewma_beta`、`service_bin_ms` 是显式控制设置 | 必须声明并按模型冻结；不得误用旧 arrival EWMA 的同名系数而声称已验证最优 |
| 扩缩容 | `IEEEReplicaControl` 要求 queue/active/TTFT 上下界、TTFT window、scale-in cooldown 和 interval | 与旧 RPS/forecast 策略区分；开发配置与最终共同 SLO 工作点身份分别记录，不按 W1 间隔反调 cooldown |
| 实际物理预算 | 总 HOST16GiB/native2GiB per owner/NVMe16GiB 是已测候选；workspace 有来源审计 | 多副本 HOST 预留与 shared file budget、GPU lease、activation/pending owner 一起核验；不是每副本各有16GiB |
| Full 启动 | 现有 `_require_ieee_full_qualification` 仍拒绝生产 Full | 有完整集成证据后才能收紧为可验证放行条件，不能用运行开关直接越过 |

本表不是候选“最优配置”。未确定的控制参数不得由测试样例自动填入；也不需要
为设置每个元数据字段重新采集整池、下载或运行整套矩阵。

## 本轮修复的具体缺口

可证伪假设：**仅在 collector 配置中加入准入 profile，仍不足以测量 Full 的
需求路径，因为 collector 没有像真实请求入口那样登记 pending KV demand，
也没有把该 intent 交给原生 generation。**

直接证据：

- `scripts/run_all_experiments.py::_exec_request_in_reservation` 在选定副本并预约后，先调用
  `_register_ieee_pending_admission`，再保护来源和准备 adapter；generation 传递
  `pending_admission_id`。`_finish_runtime_request_reservation` 等待 matching close
  acknowledgement 后才归还 capacity。
- D84 的 `scripts/ieee_tc_preflight.py::collect_native_source_wave` 原先没有上述登记
  和 intent 参数。因此旧 source-only 数据仍有效，但不能描述成同一 Full 路径。
- `source_profile_inputs` 当前仍只允许已审计 HOST 的四个 override 字段；正式
  admission initializer 的验证/组装入口尚未接入，不能宣称已能启动新配置测量。

本轮仅复用实际 Full 的方法：先登记 pending intent，登记非 GPU pending load，
再获取来源；generation 带相同 intent；结束、冲突重选和取消沿用原有 owner
释放流程。没有新增推理 fallback、取消物理保护、使用未来长度或变更九式。
旧无 profile 配置中的 pending 登记仍显式无操作。

## 无 GPU 验证

| 检查 | 修改前 | 修改后 |
|---|---|---|
| pending 登记早于 HOST source protection、同一 intent 进入 generation，close 早于归还 capacity | 失败：未登记 | 通过 |
| 已知 stale source 重选，旧 pending 关闭后新建不同 intent | 失败：未登记 | 通过 |
| 登记回复丢失且关闭未知，不生成、不假释放、不盲目重试 | 失败：跳过登记直接生成 | 通过 |
| request lifecycle/pending/scheduler/service routing 相关回归 | 新用例先保存 red 证据 | 219 项通过，2.697 秒 |
| 原有基础回归、历史保护与主机状态 | 无 GPU 性能任务 | 288 项通过，21.478 秒；147 项受保护内容零变化、GPU 已释放 |

测试采用显式小型模拟后端，不是 GPU 性能/数值正确性证据。
完整日志：`results/ieee_tc/p2_backend_qualification/d85_20260927/`。
按计划11.2用状态表，不为代码检查绘制性能图。

## 原始参考与适用范围

本轮在线核查 [vLLM0.30 配置源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/vllm.py)：
`additional_config` 参与 runtime hash（530–538行）。因此增添 admission profile
确实改变配置身份；这并不单独证明某一性能增量。原生成功结束/取消入口来自
[scheduler 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)，
本地完成长度窗口只消费成功终态，不把取消长度作为成功样本。

## 返回主线

下一步仍是完成上表的 Full 初始化绑定：在现有 preflight 中验证并组装可追溯的
admission initializer，统一窗口和实际 transfer owner，再核对其余控制与预算
配置，才启动受影响的原生路径测量。不要先跑一个半配置版本，再因身份不匹配
重跑。随后进入真实 Full activation/lifecycle。生产 profile 尚未导出或冻结。

不重做 D78 发布、D80 全池下载、D81 并发功能检查、D83/D84 已结束运行；不恢复
baseline。M1/M2/A1–A5/S1–S13 均未开始。既有零权重集合不能证明 adapter 数值
区分性的问题仍保留，不用本轮 owner 测试替代它。

## D85 后续：已接入原生完成长度初始化（无新增 GPU 运行）

以上“组装入口尚未接入”描述的是 `c854857` 时点。现在原有
`backend-model-check/native_source_matrix` 支持显式 `admission_initialization`：

```json
{
  "kind": "native_completed_length_binding_v1",
  "audit": {"path": "已完成长度审计的路径", "sha256": "对应SHA256"},
  "model": "3b",
  "window_s": 5.0
}
```

不是直接填写均值的自由配置：入口核验 audit 与其原始 run 的 SHA，重新检查
所有原生终态、完整 request 集合、native token 数及原生 prompt hash；排除已
标记 warmup，同一原始请求的不同 source/round 必须一致后去重，再计算各桶
实际完成长度均值。与 curated 不符、桶缺失、模型/后端/trace/实际子配置不符
均拒绝。没有该字段的历史 source-only 入口保持原行为；不修改旧 spec 或结果。

| 离线复用检查 | 3B | 7B |
|---|---:|---:|
| 独立原始 request ID 数（不是独立实验重复） | 21 | 6 |
| 原生 prompt ≤759 / >759 的样本数 | 8 / 13 | 3 / 3 |
| 两桶实际完成 output 均值 | 98.375 / 171.6923 | 121.3333 / 256 |
| 从真实 movement owner 派生的 transfer limit | 3 | 3 |
| 新增 GPU 测量 | 0 | 0 |

候选 W=5s 来自两模型旧 workload 配置的
`max(arrival_window=2s, scale_interval=2s, historical_TTFT_SLO=5s)`，
并非新共同 SLO 或最优 W；Full demand window 尚待显式绑定同一值。这里只接受
显式正窗口，不以测试 fixture 或缺失值自动兜底。原生 completed-length window
仍按论文 `(t-W,t]` 的已完成统计更新，不使用未来请求的 target。

采用原有 child factory 解析实际配置后，两模型离线组装通过。完成长度的复用
不代表可复用旧 D/T/O 性能 profile：新增 initializer 仍进入实际 runtime 身份。
新 collector 收尾复用 `SharedFileTransferDomain.retire`，保存 shared pressure
事件及退休前后状态，再停止后端；并未改变原生压力或物理容量公式。

验证：59项 OS/初始化测试通过（1.311s）；343项生命周期、pending、scheduler、
routing、transfer 相关测试通过（7.188s）；288项基础回归通过（22.195s）；147项
历史保护清单零变化。最初两次检查的环境/导入错误保留在
raw 日志：conda 没有 OS gate 必需的 pidfd API；system Python 没有被不必要的
experiment package import 引入的可选依赖。证据校验已去除该非必要导入，OS
检查使用既有 system Python；不安装依赖、不削弱进程安全门槛。

依据 [vLLM0.30 Request 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/request.py)，
原生 prompt 计数包含实际输入，output 计数取实际 output token IDs 长度。这里只
借此确认计数语义，不借此推断 Prime 性能。完整验证/配置身份见
`paper_results/ieee_tc/p2_backend/20260927_d85_measured_admission_initializer.json`。

下一步集中确定余下 Full/controller 配置和 profile 类覆盖，再进入受影响的
原生路径与 Full 多次激活验证。没有新增主比较结果或性能优越性结论。
