# D436：7B W0 四系统同负载比较

2026-10-12。离线复用已封存的 PrimeLoRA D371/D397、ServerlessLLM D414、
vLLM D423 和 S-LoRA D435；没有新 GPU 实验、基线调参、缓存重建或远端操作。
本分析使用冻结指标 V1、单轮 V2 和 2026-10-11 基线保真规则。不修改论文。

## 原始观测

四项均核对完整 4,000 条请求、132 个实际出现的逻辑 adapter、458,224 个原生
输出 token。500-ID 可访问池、source-item、到达、adapter、target、canonical
prompt、原生 prompt-token、工件内容身份完全匹配。请求内容摘要仍为 D424 的
`384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399`。

| 系统 / 运行 | 联合 SLO | ≥95% | GPU-s/request | 平均 TTFT ms | P95 TTFT ms | 平均 TPOT ms | P95 TPOT ms | 平均 E2E ms |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| PrimeLoRA D371/D397 |97.700%|是|3.620631|1150.798|2837.896|31.032|51.741|4825.670|
| ServerlessLLM D414 |98.125%|是|4.022441|973.687|1052.246|26.737|32.929|4000.734|
| vLLM D423，四卡 Resident |99.475%|是|4.028844|358.019|1200.507|24.798|27.177|3182.814|
| S-LoRA D435 |94.425%|否|3.789269|7396.417|17699.067|31.252|45.953|10962.479|

request 分母是完整 source/content/native/worker/physical 审计下的合同有效请求；
不能把它改写成全部请求独立数值正确性证明。原始 `n_correct=null`、数值资格
false/未知、`formal_run=false` 保留。500 逻辑 ID 中 498 零权重、2 非零权重，
4 个独立权重 SHA、6 类工件树；这是已有工件池的限制，没有重建或隐藏。

## 关键发现

1. **达标资源优势存在，但不是全面领先。** 当前三项达标工作点中 PrimeLoRA
   的物理 GPU 时间最低，相对 ServerlessLLM / vLLM 少 9.9892% / 10.1323%。
   联合 SLO 分别低 0.425 / 1.775 个百分点，P95 TTFT 高 169.699% / 136.391%。
   这是固定已知 W0 的单轮观测，不是跨场景最优、统计显著性或 G2 完成。
2. **S-LoRA 的主要违约集中在初始准备期。** 完整联合达标 3777/4000，距离
   3800 条门槛差 23 条。D435 已封存诊断中，全部 217 条 TTFT 违约位于源序列
   前 500 条；后 3500 条联合达标 3494 条。该分段说明观察到的时间集中性，
   不能删掉前 500 条、只报 warm 结果，也不能将分段当作独立重复。
3. **PrimeLoRA 相对 S-LoRA 的原始资源少 4.4504%，P95 TTFT 低 83.9658%，
   联合 SLO 高 3.275 个百分点。** 但 S-LoRA 不在本轮共同 SLO 可行集中，
   不能称其为“两项均达标的 G1 资源差”。其完整行和失败判定都保留。
4. **远端流量差异不能直接证明层级机制的独立贡献。** PrimeLoRA、
   ServerlessLLM、vLLM 各 132 次 HTTP / 131,480,060 wire bytes；S-LoRA 的
   原生全静态池准备为 500 次 / 420,740,410 bytes。各自策略均在部署通知后
   使用同一已发布 gzip 协议，无请求中打包。4000 次 S-LoRA 准备引用不是
   4000 次传输。三项按需路径之间没有本 W0 的远端字节优势；对 S-LoRA 的
   差异包含准备策略不同，不是 hierarchy/confirmed-state 的单因素证据。

## 时间边界和输出变化

TTFT 从共同计划到达到首原生 token，TPOT 用首末原生 token。E2E 保留实际
完成边界：PrimeLoRA 为 controller completion，其他项为 external HTTP
completion。共同补充端点“到达至末原生 token”均值依次为
4487.361 / 3977.746 / 3176.705 / 10955.707 ms，不将其改名为 E2E。

分析不向 S-LoRA 伪造 `native_lora_name` 或 `native_dispatch_monotonic_s`。
其真实 adapter 通过 RPC 路径/index 和已封存的完整 worker registry 审计绑定。
其 native entry 使用 model RPC registration，另外三项使用 vLLM engine entry。
因此 `submit_to_native_entry_ms` / `native_entry_to_first_ms` 只作标明边界的
描述性分解，不能视为完全同义的跨后端阶段。S-LoRA 本次两阶段均值为
7218.628 / 175.076 ms；D435 原 worker-entry 分解仍保留，不改写历史数字。

输出 digest 不同的请求数：PrimeLoRA–ServerlessLLM 203，PrimeLoRA–vLLM 231，
PrimeLoRA–S-LoRA 224，ServerlessLLM–vLLM 137，ServerlessLLM–S-LoRA 110，
vLLM–S-LoRA 157。完整 ID 记录在 JSON 中。不假定差异全部源于舍入，也不将
跨后端逐 token 相等加为新的准入条件。有界独立数值证据仍按原范围解释。

## 复用、核验与来源

仅小幅扩展现有 `check_d299_full_context.py`：显式 `--include-sealed-slora`，
默认三项输入不变；输出目录必须不存在。先核验封存 audit/manifest SHA，再
关联原始请求和物理账本，重算所有指标并与封存值核对。不重新下载已审计且
回收的对象，不要求当前分析源码字节冒充历史执行源码。

本次完整离线关联成功，旧三行全部既有字段逐项相等。D424 三个结果文件 SHA
不变；Resident 数值只复用、拒绝重新冻结：U_ref=16115.377162149176 GPU-s，
Q_ref=1200.5068596918136 ms，B=12086.532871611882 GPU-s，仅适用 7B W0。
新目录的 reference 文件与 D424 字节一致，SHA 为
`737c84804023c9f98677fb5f19c8240b4b05cabdb52f6ecdbbfd4527964c3269`。

产物位于 `paper_results/ieee_tc/main_comparison/d436_7b_w0/`：

- comparison.json SHA `59a6dfa6b3a512ac561c65fb334de4843895c5c7f120e709302d50afd6704427`。
- metrics.csv SHA `5e204c1d8514494821b71b730e27bb5e2a8ad2d25de7fc002ee1882d0f08047a`。
- JSON 保存每项阶段分解、8 个源序列分段、全部输出差异和分析来源 SHA。
- 46 项 CPU 测试通过，0.390 秒，包括后端边界、非可行行保留和拒绝参考漂移。
  测试终端记录没有冒称独立持久日志或独立审阅者 PASS。
- GPU 隐藏，CPU 2/3/26/27，MemoryHigh=3 GiB、Max=4 GiB、swap=0。
  完整关联峰值 1,699,659,776 bytes，high/max/OOM/swap 均 0。
- 两次分析启动错误分别为直接脚本入口缺 package path、既有 baseline 环境缺
  ijson；没有生成结果，也不是性能重跑。改用模块入口和已有 LLM 环境成功，
  无安装、无改输入、无 fallback。正确命令如下。

```bash
/home/qhq/anaconda3/envs/LLM/bin/python -B -m scripts.ieee_tc.check_d299_full_context \
  --matched-w0-output paper_results/ieee_tc/main_comparison/d436_7b_w0 \
  --include-sealed-slora
```

此命令需从主 worktree、在上述受限 CPU-only scope 中执行；目录已存在后应
拒绝重跑覆盖。CSV 使用标准 CRLF，diff 使用命令级 cr-at-eol 识别，不改结果。

## 下一项与尚缺证据

下一主线是剩余 7B Loquetier / HydraServe 原生资格：复用已完成的只读审计，
只做机器兼容、正确性、计量和安全所必需的适配，不搜索基线性能配置。
随后依据完整 7B 比较定位必要的 PrimeLoRA 优化；不提前跳到 3B。

W1/W2 的 Resident 参考、G2、ElasticLocality、3B、dLoRA 3B、A1–A5、S1–S13、
CE/价格敏感性、两轮审稿设计文档和后续正式图表均未因此取消或宣告完成。
尤其 confirmed-state、层级收益、扩容与 churn 因果问题仍需计划中的机制对照，
不能用本表四个 whole-system 观测替代。当前不画图、不改变评价尺子。
