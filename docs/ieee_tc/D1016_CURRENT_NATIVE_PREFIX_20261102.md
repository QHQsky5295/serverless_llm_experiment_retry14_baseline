# D1016：当前 3B 短输出停顿的单次原生诊断

2026-11-02，接续 D1015；本文件先登记问题与执行身份，结果另行追加。
当前是 baseline 必要适配和完整观测之后的 PrimeLoRA 优化阶段。HydraServe
继续使用已验证的隔离部署，不接管用户已有共享 k3s，不重开基础适配。

## 问题与终止条件

D1000 完整 W2 为 3690/4000 联合达标（92.25%），仍未达到冻结 95%。
D1014 减少查询的候选已拒绝；D1015 逐请求复用表明其净减少 38 条达标请求，
资源差仅 −0.381829 GPU-s，不采用该策略。D1000 的 154 条 TPOT 违约中
147 条输出不超过 64 token，现有完整请求日志只有阶段包络，不能分出控制
事务与 engine step。D863 的原生诊断早于文件 owner、物理清点和到达预取
修改，不能用旧占比冒充当前瓶颈。

本次唯一问题：当前短输出的首末 token 窗口内，同步控制调用、input 非控制、
engine step、其余时间分别是多少？先测这个缺口，再决定一个有证据的新候选；
不是再测试已拒绝的只读 deferral、files-first、提前等待、实例数或 token 预算。
若窗口覆盖不足，报告缺失，不扩观察上限或重跑凑样本。只有新依赖证据才可
重开已关闭方向；没有证据就不堆叠补丁。

## 执行合同

- 配置是保留 D1000 的精确克隆，仅 output/NVMe/HOST 三个独占路径不同。
- 复用现有 D1000 runner 和 D328 原生观察 wrapper；正式默认路径不变。
- 完整 source42、W2 map 先全量验证，再执行原序前 1000 请求。保留完整源
  4000 身份，不重建任何工件、到达、prompt、target 或 adapter 映射。
- `formal_run=0`，`diagnostic_prefix_v1`；不以短诊断 SLO 替代完整回放。
- 只启用既有 `host_spans_v1`；不启用 Python 栈、step 子组件或 HOST 清点
  子计时，不修改观察器、线上算法、原生后端或 IEEE 九公式。
- 每 native 进程 64 MiB / 200000 spans 上限不变。记录覆盖与截断；主机
  span 不是 CUDA kernel 时间。观察开销保留，不做事后扣除或无观察器胜负比较。
- 仍为 stdlib 解码、真实远端已发布 gzip、共同固定输出、D648 SLO 与 V1/V2。
  原始 `n_correct=null` 不改写成独立全量数值正确性通过。

执行键：`d1016_20261102/d1016_3b_native_prefix1000`。启动命令：

```bash
FAASLORA_D1000_RUN_ID=d1016_20261102 \
FAASLORA_D1000_RUN_TAG=d1016_3b_native_prefix1000 \
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261102_d1016_3b_native_prefix1000.yaml \
FAASLORA_D1000_PREFIX_COUNT=1000 \
FAASLORA_D1000_NATIVE_PROFILE=host_spans_v1 \
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

原始目录位于 `/home/qhq/serverless_llm_experiment/results/ieee_tc/`
`p2_backend_qualification/d1016_20261102/`。独占创建，不覆盖旧 run。

## 远端与安全边界

本次只读检查发现 3B port 18080 已由 lab14 的 PID 1984520 提供服务，
2026-10-31 启动，位于 session-8529.scope。认证 health 确认为既有 3B 根目录、
`prepublished_gzip_v1`、`artifact_timing_v2`，clock ID
`remote-process-monotonic:b7a04b62f38c4ae6adabb4da5c0fd8dc`。
对应用户 unit 当前 failed，原因为 Address already in use，不是 HTTP 服务
失效。不能把该 standalone PID 冒称在 unit 的资源限制内或本次启动成功。
沿用健康且配置一致的服务，不杀监听者、不重复启动，不停止另一个 7B 服务。
本轮未拥有其启动身份，结束后也不停止该既存服务。远端 NIC 当前协商 1 Gbit/s，
不据此声称应用实际吞吐等于 1 Gbit/s。测量期间不改远端或额外读取工件。

既定服务 MemoryHigh/Max 72/80 GiB、swap 2 GiB、CPU 4–23/28–47；外置
回放与监控 3/4 GiB、swap 0、CPU 2/3/26/27。空启动 MemAvailable≥102 GiB，
磁盘≥max(150 GiB,100 GiB+1.5×新增峰值)，运行低线 16/100 GiB 不变。
启动前可用内存约 104 GiB、磁盘约 150.55 GiB，无 GPU 计算进程。沿用已通过
的进程树限制、watchdog、cleanup，不重复广泛资格；不并发压缩、测试或构建。
不删除其他项目、未知数据或唯一证据。

## 官方来源、验证与变更清单

本轮复核 [vLLM 0.30.0 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
与 [profiling 文档](https://docs.vllm.ai/en/v0.30.0/contributing/profiling/)。
输入处理和 engine step 是不同边界，主机 step 还包含输出处理等工作，不能
等同纯 GPU 计算；工具观察可能影响性能。官方结构支持分解方法，不证明本机
当前瓶颈。已有 atomic_control_yield 不重新当作待实现的优化。

37 项定向测试通过，含配置仅三路径变化、opt-in/default、W2 原序前缀完整验证、
open-loop 与原生计时语义；两 shell 的 bash -n 通过。不是性能成功证据。

按 system-profile 的变更清单：

| 文件 | 变化 |
|---|---|
| configs/ieee_tc/20261102_d1016_3b_native_prefix1000.yaml | D1000 三路径克隆 |
| scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh | opt-in 复用原生观察 wrapper，默认不变 |
| scripts/ieee_tc/profile_d328_controller_python.sh | 登记本诊断 tag/output |
| tests/test_ieee_tc_d456_diagnostic_contract.py | 2 项定向合同测试 |
| 本文档 | 问题、身份、边界、结果记录 |
| 生产策略、core_timing、后端 | 无修改 |

PLAN SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
V1 SHA：`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
V2 SHA：`193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。
D648 SLO SHA：`d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`。

保存、清理与分析复用 D863/D723 的原生窗口分析，不另建框架。147 项旧结果与
用户修改保持零变化；不 stage 用户 manifest 或 EXECUTION_STATUS。显式提交
本轮文件并推送 origin V2 与 FaaSLoRA 专用 backup，不强推分歧 V2。此独立
文档承接状态，不覆盖用户修改的历史 tracker。

尚未完成：最终同版本 W0/W1/W2、G1/G2 主比较，A1–A5、S1–S13、正确性范围
汇总、审稿证据文档/图表、最终 V2 集成。当前诊断不替代这些任务，不宣布最优。

## 本次实际结果：归属检查误判导致提前中止

入口提交 `08eec2d40fd2c70e92a9d33f276c4ecd32e46c55` 已在启动前推送并
核验 origin V2 和 FaaSLoRA 的 `backup/d1016-current-native-prefix-20261102`。
tmux 启动于 17:30:58，确实进入模型执行；不是只有启动命令成功。

这次没有完成 1000 请求。外置回放记录 N_plan=1000、N_arrived=56、
N_submitted=55；服务收到 55，记录 45 个终态，其中成功且原生合同匹配 44、
中断 1。未到达/未完成部分不补成成功或超时。完整源仍为 4000。
观察到的 GPU 时间为 348.282748 GPU-s，完整 GPU 时间为 null，不参与资源
排名。没有运行要求完整前缀的分析器，不用中途 live 5000 ms SLO 当冻结指标。

最早错误是旧 `_assert_clean_gpu_environment` 在 live_progress 报告
`gpu2 pid=1445406 [No data] 7326MiB` 为 foreign。核对发现：

- 同 PID、start_ticks=311918514 在独立监控的 74 个样本中属于本轮服务资源域。
- 原生 allocation journal 也绑定相同 PID/出生身份和 GPU；没有外部计算进程。
- 本轮在线控制实际发出一次自然缩容，原因 `ieee_all_low_cooldown`，
  started=3119266.967794040、released=3119275.314833835；未接纳请求、无
  unresolved request。worker 日志显示正常 shutdown 请求及 SIGTERM 收尾。
- 旧检查先取 nvidia-smi 表、再读瞬时 parent/children 集合。退出阶段可能
  已丢失父子链，但先前 GPU 表仍保留该 PID；这不构成外部作业的证据。
- 所以早期 commentary 中的“后端退出”只描述现象；不是已经证明后端崩溃。
  不能将这次误判归因为 LoRA 数值失败、CUDA OOM、观察上限或外部抢卡。

183 个资源样本无保护中止；服务峰值 17284587520 B，high/max/OOM/swap
均为 0。launcher pass=false，service/replay 返回 -15/1；watchdog 返回 0。
最终三个原生进程已释放 GPU、服务资源域移除，147 项旧文件和两个用户 dirty
文件 SHA 不变。远端既存服务未停止、未重启。三个原生观察文件正常写入 close，
但只有部分负载，不能完成原先的瓶颈问题。

保留产物：`paper_results/ieee_tc/p2_backend/d1016_interrupted_native_prefix/summary.json`，
绑定原始回放、终态、完整失败 outcome、worker 日志、原生计时和资源来源。
原始 `analyze.py` 是预备的完整分析入口，未对失败输入执行；`close_failed.py`
只整理这次失败。其首次辅助启动因 scope 名不符合既有 UUID 身份格式，在读取
实验输入/生成产物前失败；使用原规范名字后完成，未放松资源检查。

## 修正决定与验证，不是性能优化结论

使用 [psutil 官方说明](https://psutil.readthedocs.io/stable/#psutil.Process.children)
核对父进程消失会丢失后代链；并检查现有 `NativeGPUCensus`：它已经按服务
cgroup、PID/start_ticks 判断归属，保留未知身份、阻止外部计算进程、检查逃逸，
最终 GPU release 也由它见证。它是本轮真实启用的独立每秒监控，不是新写的
“忽略错误”分支。旧祖先链检查来自历史提交 `60737ddf`，早于该资源协议。

因此只在已核验的 TC 服务中由既有独立监控承担 GPU 归属保护：

- 每次先调用现有 verify_current_service，核对 live watchdog 出生身份、
  服务资源域和实际资源限制；环境标志本身不构成授权。
- 必须存在非测试的原生 GPU census 与已锁定 NVML binding 身份，否则拒绝。
- 非 TC 路径保留原有外部 GPU 检查；不扩大允许 PID 列表、不忽略 `[No data]`
  或未知进程、不加重试/sleep，不改变缩容、路由、admission 或后端公式。
- supervisor 持续检查 watcher 退出；外部 GPU/逃逸/监控失败和实际释放规则
  均未改。不同采样视图不能再以父子链缺失推定“外部”。

新增修改：`scripts/run_all_experiments.py` 的 GPU 环境检查与
`tests/test_basic_smoke.py` 的 4 项合同测试。6 项相关 runner 检查通过；7 项
实际 supervisor 解释器的身份/退出检查和 9 项既有 native census 检查通过。
首次将 supervisor 测试误用 serving venv，7 项中 1 项因其 signal 模块缺少
pidfd_send_signal 失败；改用既定 `/usr/bin/python3` 后原测试通过，未改测试
或监控语义。保留该失败，不当作生产 watcher 功能缺失。

下一步是修正测量错误后的新 attempt 身份，不覆盖本次失败，不做同键择优。
只完成原登记的 1000 请求问题，不追加另一套性能 smoke 或扩大诊断样本。
重新启动前仍检查磁盘/内存、源与备份；当前尚未运行修正后的 GPU attempt。
最终主线与全部未完成项不变。
