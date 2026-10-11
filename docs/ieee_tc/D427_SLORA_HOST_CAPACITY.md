# D427：S-LoRA HOST 下界，不是性能失败

2026-10-11。复用全部 500 个原 7B adapter、D425 日志清单和固定原生源码。
当前 FP16 TP1 非 GQA swap loader 同时保留分模块和 packed HOST 张量。
350 个 rank8、150 个 rank16，单份 10,905,190,400 B（10.15625 GiB）；
两份 × 4 副本为 **81.25 GiB**，已经超过共同 80 GiB 上限，不能直接启动。
1/2/3 副本下界为 20.3125/40.625/60.9375 GiB，尚不含其他服务内存。

`prepare_ieee_tc_serverless_stack.py` 增加显式纯函数
`slora_host_storage_bound` 及布局源码 SHA。返回必要下界，不将剩余空间>0
作为部署合格，更不修改原生算法或未知正确性。调用方必须核验布局 SHA。
它还未接入新的 S-LoRA 服务入口；该入口本身仍待实现。
`test_ieee_tc_slora_measurement.py` 新增 6 项 CPU 检查，包括真实原生 loader
的小张量 storage 关系；mock pin 为 clone，不初始化 CUDA、不声称实测 pinned RSS。
原生 Python3.9 21 项通过 1.553s，全套 148 项通过 8.081s。

107 份历史日志与 D425 SHA 相符，指定 HOST RSS/cgroup 模式无匹配。
API 的 ModelProphet 内存估算不能替代 HOST 实测和共同资源限制证明。
两份表示也见于[官方固定版本](https://raw.githubusercontent.com/S-LoRA/S-LoRA/c1ddf488781ea7f551cd0bb07bfd097124c93411/slora/models/peft/layer_weights/lora_layer_weight.py)。
现有十文件兼容修改、用户 replay/relayserve 和所有旧性能记录保持不变。

原始 evidence 在主实验 results 的 `ieee_tc/slora_qualification/d427_20261011/`；
审计 SHA `6aa7f573cb48a72e66417d87e3b4c5c9792ed00a10fd3944d24f3a7c8bc5f977`。
主仓库详细报告 `docs/ieee_tc/D427_SLORA_HOST_CAPACITY.md`。
本轮没有 GPU 实验、远程服务操作、数据重建、runtime 适配或清理。
主仓库保留 12 成员证据包（逐字核对 raw），SHA
`2eaa2248bbdb4bc2dfb13043f389b272c9ac4bbd396dbdc862cf24e602368946`。
两个官方 loader 文件经 HTTPS 与固定 Git HEAD 逐字核对；本机已有兼容差异
单独记录。147 个历史保护项无变化；四个本轮 CPU scope 已安全关闭，零 OOM。

下一步只核查 packed 路径是否允许保持原语义的单份表示；不成立则核验合法
较少副本配置。之后立即回到共同启动/计量集成和必要资格，不能将这里当成
S-LoRA 整体性能失败，不继续无依据地试超限配置，也不重复封存的其他三系统。
