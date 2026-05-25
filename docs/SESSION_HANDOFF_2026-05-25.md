# Session Handoff: 2026-05-25

本文是 PrimeLoRA/FaaSLoRA 当前最新无上下文交接入口。新的 Codex/大模型会话
应先读本文，再读规则、索引、进度和日志末尾；不要从旧记忆或旧 handoff 重新推断。

## 1. 先读顺序

1. `docs/SESSION_HANDOFF_2026-05-25.md`
2. `docs/CODEX_INTERACTION_RULES.md`
3. `docs/DOCUMENTATION_INDEX.md`
4. `docs/PROJECT_PROGRESS.md`
5. `docs/对比实验日志.md` 的末尾

然后检查机器和仓库：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
git status --short --branch
git log --oneline -5
tmux ls || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

正常情况下，主仓只应剩下历史未提交的
`configs/generated/lora_manifest_1000.json`。除非用户明确要求，不要提交它。

## 2. 仓库与边界

主仓：

```text
path:   /home/qhq/serverless_llm_experiment_retry14_baseline
branch: retry14_continuous_queue_v2
remote: faaslora_origin -> https://github.com/QHQsky5295/FaaSLoRA.git
```

baseline / fair-comparison 仓：

```text
path:   /home/qhq/serverless_llm_baselines
branch: main
remote: origin -> https://github.com/QHQsky5295/serverless_llm_experiment_retry14_baseline.git
```

代码和结果边界：

- FaaSLoRA/PrimeLoRA 系统实现归主仓。
- SGLang、vLLM、ServerlessLLM、S-LoRA、新 baseline gating 和 cross-system
  replay harness 归 `/home/qhq/serverless_llm_baselines`。
- 不改 baseline 核心算法来制造好结果。可做环境、wrapper、materialization、
  replay、summary、audit 层适配；一旦需要改核心调度/内存布局/语义，就记录为
  gate evidence 或不可正式复现。
- 不提交远端密码、代理、登录信息或任何凭据。

## 3. 当前论文数据状态

默认论文数据仍然是：

```text
figures:      figs/
curated data: paper_results/final_v2/
paper draft:  paper/primelora_current_draft.tex
```

这些是当前 draft 默认引用的输入。`paper/primelora_current_draft.tex` 仍指向
`figs/paper/...`，尚未切换到 true-remote mirror。

真实 remote 镜像是：

```text
figures:      figs_remote_full_real_remote_v1/
curated data: paper_results/final_remote_full_real_remote_v1/
```

截至 2026-05-22，true-remote mirror 已补齐并与当前 paper draft 引用的
paper-facing 文件集合对齐：

- 主表、TTFT decomposition、Fig.7 lifecycle；
- Fig.1 intro teaser、Fig.5 normalized；
- motivation、ablation、service-readiness、control-path；
- backend-portability；
- operating-load 和 adapter-pool sensitivity。

校验状态：

```text
figs_remote_full_real_remote_v1/SHA256SUMS                    -> 62 entries OK
paper_results/final_remote_full_real_remote_v1/figs/SHA256SUMS -> 62 entries OK
paper_results/final_remote_full_real_remote_v1/SHA256SUMS      -> 68 entries OK
```

重要：true-remote mirror 是非覆盖镜像，不自动替换 `figs/` 或
`paper_results/final_v2/`。只有用户明确要求“替换主论文图/数据”时，才可以切换
paper 输入或复制 mirror 到默认目录。

## 4. 主实验闭环事实

当前默认主实验：

- Llama-2 7B
- Llama-3.2 3B
- 4000 requests
- 500 LoRA adapters
- 100% LoRA-bound requests
- Zipf exponent 1.0
- hot-set cap 48
- hot-set rotation 500
- replay scale `s8`
- metric schema `e2e_v3`

默认主表结论：

- PrimeLoRA 是 CE-first，不是 raw-latency-first。
- SGLang 仍是 raw latency 最强的 always-on runtime。
- 论文文字应强调 lifecycle cost efficiency、adapter readiness、elastic control，
  不能声称 PrimeLoRA 赢所有延迟指标。

true-remote 主表趋势与默认主表一致，但数值不同。若准备替换论文默认数据，
必须同步更新正文中的百分比和解释。true-remote 下，以每个 backbone 的最高 CE
非 PrimeLoRA baseline 作参照：

- Llama-2 7B：PrimeLoRA CE 比 SGLang 高约 `3.8%`，Cost/req 低约 `28.4%`，
  吞吐保留约 `96.8%`。
- Llama-3.2 3B：PrimeLoRA CE 比 SGLang 高约 `14.5%`，Cost/req 低约 `60.5%`，
  吞吐保留约 `100.0%`。
- 两个 backbone 平均：CE 高约 `9.2%`，Cost/req 低约 `44.5%`，吞吐保留约
  `98.4%`。

不要继续使用旧正文里 `6.7% / 20.7% / 13.7%` 等默认数据百分比来描述
true-remote 结果。

## 5. Sensitivity 与 a500 规则

`a500` 是默认主实验点，不是额外 adapter-pool sensitivity 点。

adapter-pool sensitivity 只额外跑：

```text
a100, a200, a300, a400
```

`a500` 必须复用 canonical main round。不要在 adapter-pool sensitivity 中无理由
重跑 `a500`。

## 6. ServerlessLLM-new 与新 baseline 结论

ServerlessLLM-new：

- 已作为非覆盖 true-remote candidate 闭环。
- 结果在 `paper_results/new_serverless_baselines_remote_v1/`。
- 不替换旧 `ServerlessLLM` row，也不自动并入默认 `figs/` 或 `final_v2/`。

ServerlessLLM-new warm-min4：

- `min_instances=4 + post-deploy wait 90s + target=32` 已跑完 3B/7B 4000 请求。
- 公平性裁决：这是不合理的 baseline 优化，不应作为任何性能对比表数据行。
- 理由：它把 ServerlessLLM-new 从按需扩缩容的 serverless 系统改成四个常驻
  instance 的 serverful 部署，绕过冷启动和 scale-up，而不是解决 serverless 问题。
- 诊断价值：full trace 下 TTFT 只改善约 `0.8%/0.6%`，说明主要瓶颈是稳态
  dispatch/server queue 积压，而不是启动期冷启动。

其他候选系统当前纳入边界：

- Medusa：local build/import gate 通过，但本机缺 SPDK hugepages、NVMe/Optane
  device、`/dev/gdrdrv`、passwordless sudo；不能正式跑 true-remote LoRA `e2e_v3`。
- FaaScale/LambdaScale：IPC/RDMA binding 可适配到 build/import，但本机无可用
  InfiniBand device，源码也没有 ready Llama-3.2 3B + LoRA/PEFT workload path。
- dLoRA：3B official period-migration full 4000 通过，CE `45.5240`；7B 在当前
  4x3090 上会走到 `# GPU blocks: 0, # CPU blocks: 1024`，需要改核心
  dLoRA/vLLM memory layout 才能继续。只能作为 appendix / limited 3B evidence，
  不进完整 3B+7B 主对比行。
- Loquetier：真实 adapter gates 到 3B 256 adapters / 1024 requests 和 7B
  128 adapters / 256 requests；3B/500 adapter preflight 在 24GB RTX 3090 上
  materialize adapter weights 时 OOM。只能作为 scale-gate evidence。
- AIBrix：构建/sidecar LoRA smoke 可通过，但完整系统需要 Kubernetes GPU control
  plane；本机 `kubectl`/`helm` root-only、Docker socket denied、无 kind、无
  passwordless sudo。
- HydraServe：Python control modules 可 import，vLLM 0.4.2 fork 保留静态 LoRA
  参数，但官方系统需要 Docker/Kubernetes GPU deployment，且 request path 不保留
  per-request adapter identity。
- Sarathi-Serve：artifact branch 无 LoRA/adapter/PEFT path；main branch 只有未接入
  的 `LoRAModulePath` dataclass。支持 500-adapter workload 需要新增 LoRA serving
  语义，超过公平适配边界。

## 7. 新系统调研规则

如果用户要求找 2020-2026 Serverless+LLM inference baseline，不要直接启动长实验。
先做候选系统复现可行性表，字段至少包括：

- 论文/系统名；
- 年份/会议；
- 代码地址；
- 是否开源且能构建；
- 是否真的做 LLM inference；
- 是否支持或可公平适配 LoRA/adapter workload；
- 是否能映射 `e2e_v3`；
- 适配代价；
- 建议进入主表、附录或不采用。

只要要改核心语义、核心调度、核心 memory layout 才能吃我们的 3B/7B、500 adapter、
4000 request true-remote workload，就不能作为公平正式行；可以记录为 appendix/gate
evidence 或 related work。

## 8. 若用户要求替换论文默认数据

当前状态是“可以准备替换”，但尚未替换。执行前必须再次确认用户明确要求替换。

安全路径：

1. 先确认 `paper/primelora_current_draft.tex` 当前引用的 `figs/paper/...` 在
   `figs_remote_full_real_remote_v1/paper/...` 中全部存在。
2. 选择非破坏方式优先：增加 remote build variant 或路径宏，而不是直接覆盖 `figs/`。
3. 若用户明确要求覆盖默认论文图，才把 remote mirror 同步到 `figs/` 和对应
   `paper_results` 默认位置。
4. 替换后必须更新正文百分比、表格解释、ServerlessLLM 描述和 lifecycle cost 段落。
5. 编译论文并检查 LaTeX / PDF。
6. 提交前确认没有 stage `configs/generated/lora_manifest_1000.json`。

## 9. 当前不要做的事

- 不要自动覆盖 `figs/`。
- 不要自动覆盖 `paper_results/final_v2/`。
- 不要重跑 `a500` adapter-pool sensitivity。
- 不要把 ServerlessLLM-new warm-min4 放入任何性能对比表。
- 不要把 dLoRA、Loquetier、Medusa、FaaScale、AIBrix、HydraServe、Sarathi-Serve
  包装成完整 3B+7B 正式主表行。
- 不要提交凭据。
- 不要提交 `configs/generated/lora_manifest_1000.json`，除非用户明确要求。

## 10. 快速恢复提示词

新的会话可以直接使用：

```text
你是 Codex，在 /home/qhq 上继续 PrimeLoRA/FaaSLoRA 项目。请先阅读：

1. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/SESSION_HANDOFF_2026-05-25.md
2. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/CODEX_INTERACTION_RULES.md
3. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/DOCUMENTATION_INDEX.md
4. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/PROJECT_PROGRESS.md
5. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/对比实验日志.md 的末尾

然后运行：

cd /home/qhq/serverless_llm_experiment_retry14_baseline
git status --short --branch
git log --oneline -5
tmux ls || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

默认事实：
- 默认论文数据仍是 figs/ 和 paper_results/final_v2/。
- true-remote 完整镜像是 figs_remote_full_real_remote_v1/ 和
  paper_results/final_remote_full_real_remote_v1/，已经补齐 paper-facing 图表，但尚未
  替换默认论文输入。
- a500 是默认主实验点；adapter-pool sensitivity 只额外用 a100-a400。
- 不覆盖 figs/ 或 paper_results/final_v2/，除非我明确要求替换主论文数据。
- 不提交 configs/generated/lora_manifest_1000.json。
- 不提交任何远端密码或凭据。
- ServerlessLLM-new warm-min4 是不公平的诊断实验，不进入性能对比表。
```
