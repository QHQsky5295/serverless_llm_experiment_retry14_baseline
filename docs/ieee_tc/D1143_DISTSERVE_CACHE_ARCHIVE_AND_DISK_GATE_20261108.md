# D1143: DistServe 缓存归档与推理机磁盘门禁恢复（2026-11-08）

## 论文实验状态

本节点只处理实验环境的可恢复缓存，不改变任何已发布结果、模型权重、LoRA 工件、请求轨迹或用户工作区文件。清理前，本地推理机 `/home/qhq` 的可用空间为约 150.2 GB，低于计划要求的 150 GiB 启动门禁；清理后可用空间为 `163,672,645,632` B（约 152.5 GiB，`df -BG` 显示 153G），因此 7B 正式回放门禁重新满足。

## 对象与可恢复性证据

被处理对象为：

```text
/home/qhq/serverless_llm_baselines/cache/distserve_official_20260615/
```

该目录包含 1,348 个 `.pt` 转换权重分片及一个 `done` 标记，大小为 `13,480,771,999` B。`distserve/downloader/downloader.py` 明确将该目录定义为从本机 Hugging Face 模型转换得到的缓存，并在 `done` 存在时复用；本机仍保留原始模型：

```text
/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf/
```

该模型的 `config.json` 和 `model.safetensors.index.json` 均存在。因此该目录是可由现存原始模型和已冻结的官方转换器重建的派生缓存，不是唯一实验结果。清理前还确认没有正在运行的 DistServe 服务或该脚本的回放进程。

## 远端恢复归档

清理前先将整个目录以未压缩 tar 流保存到远端制品节点 `primelora-artifact-174`（主机名 `lab14`），未修改远端服务或 k3s：

```text
/home/lab14/primelora_archives/distserve_official_20260615_20261108T144707Z.tar
```

归档包含 1,350 个条目（目录、1,348 个文件和 `done` 等条目），大小为 `13,483,048,960` B。远端 SHA-256 为：

```text
68b9eaac6f0d8c6679b011ac12f9a8bffecc98d0b48824e0808baed2222a8a6c
```

同时保存了 1,348 个源文件的逐文件 SHA-256 清单：

```text
/home/lab14/primelora_archives/distserve_official_20260615_20261108T144707Z.manifest.sha256
```

因此后续若需要复现 DistServe，只需从该归档恢复缓存，或按照官方转换器从仍在本机的原始模型重建；两条路径都不需要重新生成 LoRA 工件或请求负载。

## 清理边界与下一步

- 已删除的仅是上述可重建的 DistServe 派生缓存；没有删除任何 `results/`、`paper_results/`、`figs/`、模型权重、LoRA 工件、trace 或用户 dirty 文件。
- 推理机的 150 GiB/100 GiB 规则保持不变；本节点不改变远端制品节点的独立容量公式。
- 现在可以重新执行计划中的 7B baseline 合同复核；正式回放前仍需使用现有 GPU idle/cleanup gate，并保持唯一 `campaign_id/model/seed/run_tag`。

## 可追溯锚点

- 计划文件 SHA-256：`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`
- 清理前文件数：1,348
- 清理前目录字节数：13,480,771,999
- 清理后本地可用空间：163,672,645,632 B
- 本节点只读归档操作未改变远端 artifact service，也未接触共享 k3s。
