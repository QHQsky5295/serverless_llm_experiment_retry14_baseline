# 既有 7B/3B 工件内容核查：2026-09-26

## 结论与范围

按计划 §4.3、§6.7、§12 检查当前本地 frozen 工件的全部 500 个逻辑 ID。
这是原始工件的只读 CPU 检查，不生成、训练、修复或替换任何权重，不是性能实验。
清单、文件内容、大小和修改身份在扫描期间均保持不变。

| 当前工件池 | 逻辑 ID | 不同权重文件 SHA | A/B 全零 ID | 含非零 A/B 的 ID |
|---|---:|---:|---:|---:|
| Llama-3.2-3B | 500 | 2 | 500 | 0 |
| Llama-2-7B | 500 | 4 | 498 | 2 |

3B：rank 8 的 350 份与 rank 16 的 150 份分别具有相同权重 SHA；全部为 FP16，
没有 NaN/Inf。7B：348 份 rank 8 和 150 份 rank 16 全零；`finance_lora` 与
`medical_lora` 的 A、B 含非零值且 SHA 不同。非零不等于已经验证领域训练质量，
也不凭 A、B 分别非零就宣称其乘积在每层都非零。

检查的是当前本地文件。历史运行或远端副本是否具有完全相同内容，要用各自
保存的 SHA 进一步核对，不能把当前扫描直接回填成历史测量。

## 数学意义与论文证据边界

标准 LoRA 的附加项为缩放后的 `BAx`。任一操作数为零即可证明该层更新为零。
这也解释了 PEFT 默认初始化为何可保持原模型行为；官方说明是 A 随机初始化、
B 为零，而不是本次观察到的 A、B 都零。
[PEFT 官方 LoRA 初始化说明](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#initialization)。

- 不能将这些 ID 表述为 500 个独立训练或领域专用模型。
- 零权重仍可能走实际 LoRA 注册、内存、传输和计算路径；因此发现零权重并不
  自动使过去所有计时失效。但必须证明路径确实执行，并限定为结构性系统负载。
- 全零适配器的输出不能区分“加载了正确 ID”“加载了另一个零适配器”及
  “根本没有应用 LoRA”。原生 ID/slot/加载回执证明控制路径，不替代这一
  数值正确性证据。不能用同一份被测输出自己充当正确性的独立答案。
- fixed-output 协议减少了生成长度差异，但不能消除工件内容/压缩代表性问题。
- GPU-s、SLO 和缓存收益的主张仍需完整公平实测；不因本次发现先验断言 Prime
  或某个 baseline 胜负，更不为获得非零输出修改工件。

## 刚完成的 3B 原生输出对照

原 run tag：`llama32_3b_nativeadapterref2_attempt1`。
同一个既有 `req_00003` prompt，先后使用 `finance_lora` 和 `code_lora_0015`。
后者按原始前 100 请求中首次出现的不同权重 SHA 选择，没有按输出筛选。

| 检查 | 结果 |
|---|---|
| 原生后端 | stock vLLM 0.30.0 AsyncLLM |
| prompt / 原生 input IDs | 两次相同 |
| 实际/目标 token | 各 217/217 |
| 两次 output IDs SHA | 相同：`2c2300cd16afc6c4e1295a8695da2899ffc2e9dace654f7389d97f4cfb25148c` |
| 权重内容 | rank/file SHA 不同，但两者 A、B 全零 |
| 结论 | API、长度、清理通过；语义可区分的负对照未建立 |
| 外部资源证据 | 64 次采样，服务峰值 4,926,136,320 bytes，high/max/OOM 均零 |
| 收尾 | 原生 scheduler/cache 空，GPU context 已释放、服务资源域已移除 |

原始文件中的 `pass=true` 仅覆盖其实际断言：生成完成、token 数和清理。
没有不同输出断言，不可升级为适配器应用正确性通过。原始 `input_mode`
沿用了 concurrent-pairs 标签，实际 requests 和代码路径明确是两个顺序对照；
原始结果不改写，curated summary 已显式指出两项限制。

该发现解释本次“切换不同 SHA 仍同输出”的不可区分性，**不解释**此前 3B
取消/批次改变输出的具体数值根因。后者已由 stock AsyncLLM 独立复现，
不能归因于 Prime 的引用/驱逐，也不能未经证明断言某个 kernel 有错。

## 字节与来源：不能把填充当 LoRA 参数

| 内容（500 个逻辑目录求和） | 3B bytes | 7B bytes |
|---|---:|---:|
| safetensors 权重文件 | 5,978,564,400 | 10,922,202,000 |
| `adapter_data.bin` | 8,932,962,246 | 0 |
| 全部目录文件 | 19,482,573,296 | 12,985,984,450 |

3B 填充文件逐字节检查均为零。这些数是**逻辑文件字节**，不是独占物理磁盘、
HOST tensor、GPU pool 或网络线上字节。共享 inode、page cache、压缩均需分开。
现有 artifact 服务使用 tar gzip (`remote_artifact_node/server.py:123`)；全零权重
及填充高度可压缩。因此不能把本地按目录大小注入的等待视为真实远端等价，
也不能未经新证据假定远程路径对所有方法影响相同。按原计划分别记线上字节、
打包/传输/解包，保留实际部署的路径语义。

来源审计事实：

1. 3B `.publicmix_generation_manifest.json` 记录 500 个 ID、rank/名义大小，
   没有足以证明已训练的来源信息；当前目录无 `faaslora_artifact_spec.json`。
2. 7B manifest 的 `pool_type=publicmix_v2`，但 `public_count=0`、
   `generated_fill_count=500`。目录名 publicmix 不能充当真实公开工件证明。
3. 当前 `generate_adapter_synthetic` 明确使用 `torch.zeros` 写 A/B，并按名义大小
   填充零字节；函数注释中的 randomly initialized 与实际代码不一致。
   当前内容与这类路径相容，但未恢复确切生成命令，故不声称已经证明历史来源。
4. 既有 sanitized 脚本可将 NaN/Inf 替换为零，但其路径不是当前被测路径；
   没有证据就不能把本次全零归因于 sanitization。
5. 对 public_candidates、archive、remote 和两个 remote 备份目录的 2,521 份
   adapter config 做限定范围检索，未发现 base 指向 Llama-3.2-3B 的候选。
   这不是整台服务器所有文件都没有合适工件的证明，也没有下载替代品。

## 主线处置与待决策

已完成：本次原生对照制表、两池全量内容审计及可复核 CSV/JSON。

可以继续：7B 现有非零工件的独立数值正确性验证、Full 公式与生命周期接入、
Serverless 历史审计和其他不依赖新增权重的工作。两个原池保持原样。

暂不准入：把 3B 当前全零池视为“适配器语义已正确验证”，或把两池写成
500 个独立训练模型。本次资格不能放行完整 TC 主性能结论。

需要用户确认的范围变更：计划禁止新增权重，而已有查找范围内没有非零 3B。
建议仅允许补充少量明确来源、兼容的已训练非零 3B 工件，用于独立正确性对照，
不替换/复制原 500 池、不重建 trace、不删除旧测量；是否升级代表性实验范围
另行决定。若不允许，则保留结构负载限定及该语义证据缺口，不能强行通过。

不要重复运行相同的零权重负对照；它在数学上不可区分，不是多跑 seed 能解决。
M1/M2、核心消融和敏感性仍未开始；本检查不是这些实验的替代品。

## 复核与产物

沿用 `scripts/ieee_tc_preflight.py artifact-audit --path <existing-pool>
--expected-adapters 500 --output <new-json>`；只读输入、拒绝覆盖输出。
按 SHA 复用张量统计，所有逻辑文件均独立哈希，扫描前后检查文件身份。
七项小型测试覆盖零/非零操作数、NaN、缺失、重复 ID、扫描期间变化及零修改。
测试中的小数组只是临时数学夹具，不是模型工件或实验负载。

- `paper_results/ieee_tc/p2_backend/20260926_artifact_tensor_audit.{csv,json}`
- `paper_results/ieee_tc/p2_backend/20260926_3b_nativeadapterref2.{csv,json}`
- 原始审计：`results/ieee_tc/p2_backend_qualification/model_20260926/artifact_tensor_audit_3b7b_attempt1.json`
- 原始 SHA：`224b9337ecaba2590ac888e9d0e3b9eb41a17074ce77b952e98e935259adf60f`

CPU 扫描使用独立 1/2 GiB high/max、swap=0、CPU 3/27，峰值 1,074,528,256 bytes。
产生 43,008 次 high 事件（文件缓存回收/节流）；max/OOM/OOM-kill 均零。
因此不从扫描耗时推断性能。结束后进程为空，仅停止该已空的专属 scope。
没有并行模型运行、整机 page-cache 清理、权重修复或历史结果覆盖。
