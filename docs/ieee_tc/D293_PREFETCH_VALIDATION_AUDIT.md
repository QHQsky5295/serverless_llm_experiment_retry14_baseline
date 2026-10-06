# D293：7B 启动预取诊断；延迟改善但尚未达标

日期：2026-10-06。执行主线仍为 Prime 7B → Prime 3B → 基线/主比较。
本记录不是正式 G1/G2 结果，也没有宣告优于旧 PrimeLoRA。

## 实验问题与运行状态

D290 的初始就绪超过共同 60 秒准备窗口，排队成为用户 TTFT 的主要部分。
因此只测试 vLLM 已有的 safetensors 预取路径，不改变 IEEE 九个公式、路由、
admission、工作负载、输出长度或远端交付边界。

| 运行 | 状态 | 可以说明什么 |
|---|---|---|
| D290 | 100 请求完成 | 未启用显式预取的短诊断参照 |
| D291 | 正式请求前失败 | 新增启动配置被旧 service-profile 身份规则拒绝；不是性能点 |
| D292 | 正式请求前失败 | 已执行预取，但子进程派生的加载回执被错误当作父配置差异；不是性能点 |
| D293 | 100/100 返回成功、原生输出数量匹配；生命周期完整 | 候选短诊断，数值正确性与正式评价仍未闭合 |

D291/D292 原始结果不覆盖、不混入性能分母。启动加载策略从 post-ready
service-profile 身份中分离（`175b37a`），只读派生回执从父子配置相等检查中
分离（`ec7fdb0`）；运行请求的语义配置仍需严格匹配。D293 启动版本为 `632bd7d`。

官方 vLLM 提供将 checkpoint 提前读入 page cache 的 prefetch 加载策略；它是
已有后端功能，不是 PrimeLoRA 的新算法贡献。
[官方加载配置](https://docs.vllm.ai/en/stable/api/vllm/config/load/)。
本机四个 worker 日志均确认实际选择 `prefetch`，checkpoint 大小为
13,476,876,272 字节。初始 worker 的权重加载为 4.53 秒，model-loading 阶段
6.669 秒；这些阶段不是整个 activation，不能混称冷启动时间。
父进程 result metadata 的 resolved-loader 字段仍为 null，不能用其填造成功；
实际选择的证据来自四个保留的 worker 日志。

## 按冻结 V1 重新计算

两个运行均为原 W0 seed42 trace 的同一 100 请求前缀，并非完整 4,000 请求。
逐请求 ID、adapter、到达偏移、prompt SHA、目标输出、generation contract、
native token source 全部相同。trace/subset SHA 相同，request-map SHA 为
`a83e56ea59d72e5f2c01bb0566a48b55450cbcaf77135933c2fa6cfb96151974`。

| 指标 | D290 | D293 | 解释 |
|---|---:|---:|---|
| 初始就绪时间 / s | 103.504 | 50.496 | 本次观测下降 51.21% |
| 平均用户 TTFT / ms | 13,558.675 | 2,170.015 | 本次观测下降 84.00% |
| P95 用户 TTFT / ms | 39,448.237 | 10,538.035 | Type-1；下降 73.29% |
| 平均 dispatch/admission / ms | 12,863.727 | 1,526.385 | 已包含 arrival lateness |
| P95 dispatch/admission / ms | 38,645.078 | 10,087.224 | 分位数不能跨阶段相加 |
| 平均 service TTFT / ms | 694.948 | 643.630 | 下降 7.38% |
| P95 service TTFT / ms | 1,760.353 | 1,933.152 | **退化 9.82%** |
| 平均 TPOT / ms | 35.922 | 34.011 | 原生事件计量 |
| P95 TPOT / ms | 44.015 | 42.639 | Type-1 |
| 物理生命周期 GPU-s | 714.257 | 769.308 | **增加 7.71%** |
| 原生数量合同完成数 | 100 | 100 | 不等于数值正确性分母 |
| `N_correct` | 未知 | 未知 | 不生成 GPU-s/correct-request 排名 |

原生 first-token/arrival 的 TTFT 重算误差、last/first-token 的 TPOT 重算误差
两次均为 0 ms。D293 实测互斥窗口 GPU-s：业务前 38.272、到达窗口 611.239、
drain 34.251、cleanup 85.546，合计 769.308。清理和准备仍计入资源。
扩容事件均为三次，但请求触发位置不同：D290 为 20/22/24，D293 为 8/9/12。
不能假定更快启动必然降低整段生命周期占用。

### 两项离线统计修正

1. 旧 D290 绘图器把 arrival lateness 加到了已经为 `d-a` 的 dispatch wait 上。
   新绘图器只使用 `dispatch_admission_wait_ms`，并逐请求检查
   `overall_ttft = dispatch_wait + service_ttft`，不修改原始请求记录。
   原 D290 图和文档保留为历史；本目录的 corrected 版本才是当前分解。
2. runner 终端和旧汇总的 P95 是线性插值；冻结 V1 要求 Type-1。
   新分析由原始请求重算；图、表、manifest 均用 Type-1，另外保留 legacy 值。
   因此前述即时报告约 10.55 秒更精确地修正为 10.538 秒，不是重新运行。

## 不能越过的证据边界

- 只有每配置一次短回放，不计算运行间 CI，不外推 4,000 请求。
- D291/D292 已经读取 checkpoint，文件缓存没有随机化；禁止整机 drop cache。
  不能把 D290/D293 的全部启动差异归因于预取。
- 6 条输出 token-ID SHA 不同：`req_00005/00059/00074/00077/00084/00091`。
  对应 legal、finance、medical adapter，清单及 SHA 保存在 comparison manifest。
  输入和输出数量一致不证明这六条数值执行正确，也不证明它们错误。
- 官方 vLLM 0.30.0 说明默认不保证完全可复现，在线调度的批组合可能影响输出。
  这是可检验的候选解释，**不是本次六条差异的证明**；不以它自动关闭资格问题。
  [官方可复现性说明](https://docs.vllm.ai/en/v0.30.0/usage/reproducibility/)。
- D236 的独立首位置参考、D247/D248 的 22 请求序列诊断与 D249 的旧错误回执
  隔离均继续保留；不能用它们自动代替本次 100 请求或以后 4,000 请求的资格。
- 共同 warm SLO 和 Resident 预算尚未正式冻结，开发 5000 ms 不是正式阈值。
  物理占用增加和 service-tail 退化均保留；预取候选暂不获 G1/G2 接受。

## 资源、网络与数据保护

D293 launcher/service/watchdog return code 均为 0，四份 allocation journal
均真实释放，无 open lease；后续 NVML 四卡均仅 15 MiB 显示占用、无推理进程。
服务域峰值内存 18.048 GiB，主机最低 MemAvailable 87.301 GiB；服务
high/max/OOM/oom-kill 事件与 swap 均为 0。推理机 150/100 GiB 规则没有下调。
运行尾部可用磁盘一度低于新的重型任务 150 GiB 门槛；下次启动必须重新检查，
不因本次清理后显示约 151 GB 就假定已经满足 GiB 门槛。

运行结束后停止了所属远端 7B 服务。只读复查：推理机 `eno1np0` 的 sysfs 为
`speed=1000, duplex=full, carrier=1`；远端 `eno1` 报告 `1000Mb/s, Full`。
这证明当前链路协商为 1 Gbps，不证明应用有效吞吐达到该值，也不说明此前
100 Mbps 的根因已经确认。未执行 NIC 配置、重启或压测。

用户批准的一次只读交付缓存已经在 D78/D80 完成，继续复用共
1,581,215,261 字节的已发布集合；本次没有新增权重/负载/解压池。
部署前归档制作不进请求计时；请求触发的必要读取、传输、本地准备仍计入。
远端配置没有在运行期间调整。

147 项旧保护清单复查为 146 项一致；唯一差异是另一个仓库已有 dirty 的
`serverless_llm_baselines/scripts/replay_openai_trace.py`，当前 SHA
`04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22`。
该文件本轮未修改、未 stage、未回滚；**不能写为全清单通过**。
被保护的历史结果与旧图未改变。

## 可复查产物与下一步

- 表/来源：`paper_results/ieee_tc/p2_backend/d293_prefetch_audit/`。
- 图：`figs/ieee_tc/p2_backend/d293_prefetch_audit/`，含 D290 corrected 子目录。
- 脚本复用 `scripts/ieee_tc/plot_d290_diagnostic.py`，新增显式 reference/data
  目录、字段/恒等式检查、Type-1 和禁止覆盖；12 项 CPU 测试通过。
- 初次测试发现绘图依赖不在推理环境；改为轻量校验与绘图延迟导入，复用现有
  Anaconda plotting 环境，不安装包、不更改推理环境。字体缓存旧记录导致的
  预览失败保留，新稿显式注册已安装 TNR，不作字体替代。
- 图按 academic-plotting 和 Plan §11：3.45 inch 单栏、嵌入 TNR regular/bold、
  下方加粗子标题、300 DPI PNG 与矢量 PDF；已检查渲染、字体和页面尺寸。

下一步首先沿用 D225–D249 的已有证据，对六条同后端差异审计实际权重/slot/
request binding 与批调度；只有现有日志确实缺失时才使用已有 opt-in token
采集补最小诊断，不新增工件、不靠重复运行寻找相同输出。之后才推进受保护
完整 7B 回放及共同测量标定。3B、外部基线、M1/M2、A1–A5、S1–S13 均未关闭。
