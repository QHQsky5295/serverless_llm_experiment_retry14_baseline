# Final Remote-Fair True-Remote Snapshot

生成时间：2026-05-14。

本目录是 `remote_fair_real_remote_v1` 的可恢复快照，只包含已经闭环且有用的
true-remote 数据、表格、图和对比文件。它不会覆盖旧的 `paper_results/final_v2/`
本地闭环快照。

## 口径

- GPU 推理节点：`192.168.4.178`
- remote artifact HTTP 节点：`192.168.4.174`
- Llama-2 7B endpoint：`http://192.168.4.174:18081`
- Llama-2 13B endpoint：`http://192.168.4.174:18082`
- Llama-3.2 3B endpoint：`http://192.168.4.174:18080`
- workload：4000 requests、500 LoRA adapters、100% LoRA-bound、s8 replay
- fair rule：cold/startup/first-touch materialization 必须从同一 remote backing
  store 发生；系统自身的本地 cache、host-memory pool 和 runtime residency 保留。

## 目录

- `tables_figures/main_remote_fair_real_remote_v1_7b3b/`：
  Llama-2 7B 与 Llama-3.2 3B 的五系统 true-remote 主表、TTFT decomposition、
  lifecycle cost 图、CSV 和 manifest。
- `tables_figures/backend_portability_real_remote_v1_7b3b/`：
  vLLM/SGLang matched-backend portability true-remote 表、图、CSV 和 manifest。
- `source_summaries/baselines/`：
  baseline harness 的 true-remote summary JSON gzip 副本，包含 7B/13B/3B。
- `source_summaries/primelora/`：
  PrimeLoRA-vLLM 与 PrimeLoRA-SGLang true-remote summary JSON gzip 副本。
- `comparisons/`：
  每个 baseline true-remote round 与 local-sim remote-fair 闭环结果的差异表。
- `SHA256SUMS`：
  当前快照内所有文件的校验和。

## 使用边界

7B 与 3B true-remote 主表中，PrimeLoRA-vLLM 仍为 CE 第一：

- Llama-2 7B：PrimeLoRA CE `118.84`，SGLang CE `114.47`。
- Llama-3.2 3B：PrimeLoRA CE `212.55`，SGLang CE `185.66`。

13B true-remote 结果保留为诊断数据，不合并进主表：当前 13B 下
PrimeLoRA-vLLM CE `60.35`，SGLang CE `85.83`，不满足主表中 PrimeLoRA
合理 CE 第一的写作目标。

PrimeLoRA-SGLang true-remote 作为 backend portability 扩展使用：

- Llama-2 7B：PrimeLoRA-SGLang CE `173.73`，SGLang CE `114.47`。
- Llama-3.2 3B：PrimeLoRA-SGLang CE `466.57`，SGLang CE `185.66`。

失败、partial、smoke 和 debug round 没有纳入本快照。
