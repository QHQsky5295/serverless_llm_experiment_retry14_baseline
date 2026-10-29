# D937：3B W2 内容驱动 NVMe 容量候选协议

## 假设

D935 的 500-adapter W2 内容索引占用 19,495,223,296 个 4-KiB 文件分配字节，
即 18.156 GiB；当前 Full 的 16-GiB NVMe 预算不能容纳整个已观测集合。D935
中 511/4,000 个请求在 dispatch 前仍处于 remote-cold，remote-cold 组的 TTFT
明显高于本地层级。候选假设是：只把由冻结内容索引和文件分配粒度计算出的容量
提高到覆盖集合并保留有限安全头部，可以减少重复远端获取，降低 P95 TTFT，而不
改变 GPU 预算、路由、准入、请求 trace 或九个公式。

## 唯一候选配置

内容索引实测分配字节为 18.156 GiB。按“集合大小 × 1.25”得到 22.695 GiB，
向上取二进制 MiB 档位为 24,576 MiB（24 GiB）。这是由实际内容字节和固定文件
分配粒度推导的单一候选，不是根据运行结果反向选择的容量。HOST 仍为原 16 GiB，
GPU/TP、`max_instances=3`、transfer limit、service/preparation profile、
remote delivery mode 和 W2 输入完全不变。

脚本 `scripts/ieee_tc/run_d937_3b_full_w2_nvme_extent.sh` 仅通过已有
`FAASLORA_NVME_CAPACITY_MB` 接口覆盖该容量，使用新的 raw/NVMe/HOST 路径；不
改旧 D886/D935 结果，不复用旧 cache，不重建工件或负载。

## 验收

运行必须满足：4,000/4,000 终态成功、固定长度输出契约和 TPOT 完整、无 fallback、
GPU 上下文释放和服务路径删除通过，且无 OOM/memory.high/max/swap/disk stop。
比较字段固定为 remote-cold 比例、远端传输次数/字节、P95/平均 TTFT、联合 SLO、
物理生命周期 GPU-s/request 和 cache 层级分布。

若 remote-cold 显著下降且 P95 TTFT 改善，同时 U 不超过 D935 的资源预算可行域，
候选进入下一轮跨模型验证；若 P95 不改善、准备阶段把成本转移到 GPU 生命周期、
或安全包络被触发，则保留阴性结果并撤回该假设。无论结果如何都不重跑 D937 同一
执行键、不筛选有利请求、不修改共同评价口径。
