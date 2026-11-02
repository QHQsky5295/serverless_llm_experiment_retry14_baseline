# D1011：3B 四路制品 movement 候选

日期：2026-11-02  
阶段：PrimeLoRA 3B 开发优化；尚未运行  
执行键：`d1011_20261102 / d1011_3b_four_lane_movement_w1`  
参照：D1000（当前开发参照）与 D81（远端四路交付资格）

## 1. 证据与可证伪问题

D1000 的 3B Full 配置为 `max_concurrent_loads=3`，但 D81 已在相同的
prepublished-gzip 交付方式下完成 3-way wave、每 wave 4 个并发传输，且本地/远端
资源峰值、服务 high/max/OOM/swap 事件均为 0。D1000 仍观察到 493 个
`remote_mismatch`，说明单纯依赖三路 movement 可能让 arrival-owned NVMe
准备滞后于请求到达。

## 2. 可证伪假设

在已具备 D81 四路交付能力的前提下，把 Full path 的全局 movement 上限从 3 提到
4，可能缩短 arrival prefetch 排队，使更多请求在普通非阻塞路径中直接观察到
NVMe/HOST/GPU，而不需要 D1008–D1010 的 blocking 等待。若远端/本地竞争成为
新瓶颈，TTFT、联合 SLO 或物理 GPU-s/request 会不改善或恶化。

本候选只改一个控制量：

```yaml
max_concurrent_loads: 4
```

模型、trace、adapter map、W1、路由、admission、扩缩容、生成合同、服务/准备
profile、远端 endpoint 与生命周期计量完全沿用 D1000；不设置 `max_instances=4`，
避免混入 D1005 的扩缩容变化。

## 3. 不变量

- movement semaphore 是全局共享上限，不是每 GPU 独立配额；
- request-time transfer、receive、materialize 和竞争仍计入 TTFT/E2E；
- 不复制工件池、不读取未来热点、不改变公式或共同 SLO；
- 4-lane 只在已有 D81 资源资格证据基础上测试，不把 D81 功能测试当作性能结果。

## 4. 预注册验收与裁决

先运行 100-request smoke，检查有效配置绑定、共享 movement 上限、资源释放和
无失败；随后使用 D1000 同一 seed-42 trace、adapter map、W1 执行唯一
4,000-request full replay。

必须通过：

1. 4,000/4,000 native terminal、无失败、GPU lease 全部释放；
2. TTFT/TPOT/E2E 重算误差为 0，movement 记录显示有效的全局四路上限；
3. 与 D1000 比较联合 SLO、TTFT P95、Remote mismatch 数、远端字节和物理
   GPU-s/request；
4. 只有联合 SLO 提升且物理 GPU-s/request 不高于 D1000，或两者均达到预注册
   的明确改善时，才保留为开发候选；否则拒绝“仅提高 movement 并发”的解释，
   不继续无新证据地提高到五路或更高。

D1011 的首次 smoke 执行键 `d1011_smoke_20261102 /
d1011_3b_four_lane_movement_w1_smoke` 在服务启动前被 preflight 拒绝：可用磁盘
约 160.9 GB，低于冻结的 161.1 GB 启动地板；因此它没有产生模型、请求或性能
观测，不能算作失败性能结果，也不会用同一执行键重跑。随后对已完成且已审计的
D1008–D1010 原始文件生成 SHA 清单并无损 gzip 压缩，释放空间后使用新的 smoke
执行键复核配置。D1011 不因预期有利进入论文主结果。原始结果、审计和配置与
D1000 分离保存。
