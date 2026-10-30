# D947：可重建下载缓存清理回执

日期：2026-10-30  
计划 SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

## 清理对象

本次只清理两份 Hugging Face 下载中断缓存。两份文件均为 `.incomplete`，已有对应的完整模型文件，未被进程打开，且没有活动下载进程；没有删除模型权重、LoRA 工件、trace、原始结果或用户修改。

| 路径 | 清理前大小 | 文件时间 | 清理理由 |
|---|---:|---|---|
| `/home/qhq/serverless_llm_experiment/models/mistralai--Mistral-7B-Instruct-v0.3/.cache/huggingface/download/tFt5id3gCLBzpax0WVpbeEjTCck=.76d5729be995fb6510c7f103c5b8825d3d82f8126ac949db6be367a638694d1f.incomplete` | 10,728,839,687 B | 2026-03-15 | 同目录已有 `model-00001/2/3-of-00003.safetensors` 完整文件 |
| `/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf/.cache/huggingface/download/fPHULxv55kAe7RSfHmmL42LIc1I=.ee62ed2ad7ded505ae47df50bc6c52916860dfb1c009df4715148cc4bfb50d2f.incomplete` | 5,454,683,826 B | 2026-04-13 | 同目录已有完整 `model-00001/2-of-00002.safetensors` 和索引文件 |

合计释放 16,183,523,513 B。清理后根文件系统可用空间为 177,068,855,296 B（约 164.9 GiB），重新超过推理节点 150 GiB 重型任务门槛。清理只作用于可重建缓存，未改变实验配置或结果。

## 后续规则

其余大文件仍保留。下一次清理必须逐项确认：可靠副本或完整替代物、无进程/句柄使用、没有配置引用、不会破坏历史证据；不能按目录名称批量删除。正式实验开始前仍需重新检查磁盘、内存、swap、GPU 和服务状态。

