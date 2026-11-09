# D1174：清理 D1172 无效尝试留下的可重建 NVMe 缓存

日期：2026-11-10。

## 原因

D1172 因部署—回放 barrier 缺失被裁为协议失败；其请求级日志、service.log、
远端监测、launcher/launch 证据仍保留。失败运行的临时 NVMe 目录包含 459 个
从已发布远端工件重建的副本，占用约 11.44 GB，使推理机可用空间从约 167.5 GB
下降到约 156.1 GB，低于既有 161.06 GB 启动线。D1173 因此在性能启动前被磁盘
preflight 拒绝，未产生 GPU 结果。

## 处理

只删除以下失败运行的可重建临时目录：

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1172_20261109/7b_nvme_demand_weighted_preload_w2`

不删除 D1172 的 raw 日志、launch 证据、远端 journal、配置、代码、模型、LoRA
工件池、trace、adapter map、curated evidence 或任何用户 dirty 文件。D1172 的
部署选择和失败时序可由保留的 service.log 与 remote monitor 复核；缓存本身不是
不可替代的实验观测。

清理后重新核对可用空间，再以全新执行键启动 barrier 修正版；D1173 的 preflight
拒绝和 D1172 的协议失败均不进入性能比较。
