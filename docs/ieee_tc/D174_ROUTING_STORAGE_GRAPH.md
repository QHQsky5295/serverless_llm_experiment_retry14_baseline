# D174 — 路由观测保留实时存储图，省去无消费者的 tensor 明细

2026-10-02。P2/P3 单一开发候选，父版本 `bd7cb1c527216b009548bf78779ba4fadf2c6936`。
状态：定向、组件等价和广泛回归均通过；资格完成，不是性能接受。

## 问题、历史和可证伪假设

D173 完整 W0 的平均 TTFT 4.630496 s、P95 11.634821 s；新解析器没有
带来明确端到端净收益。约 90.64% 的平均首 token 等待在 native dispatch
之前；这不证明全部等待由 HOST 枚举导致。D162 的当前 cap4 独立诊断中，
四个 native core 的 HOST inventory 分别出现在 128/552、116/531、89/529、
98/527 个业务采样栈中，不能将采样出现比例当作 CPU 时间比例。

D144 已消除同次重复 pin 查询；D151 已省掉路由不消费的 staging/allocator
部分；D132 在独立 frontend 验证注册图，再给 controller 投影。上述工作
不重复。本次核查发现 HOST tensor 的 name/shape/stride/view bytes/offset
明细仍在 GPU core 每次完整构建，传给 frontend 后没有消费者。
`NativeSourceSnapshot._footprints` 使用 allocation graph、adapter edges、
dtype、共享/独占容量和 GPU pool，而不读取 `host_tensor_views`。

假设：在生成点不物化无消费者的描述表、在同一次遍历直接聚合 dtype，
能够降低原生核心同步 CPU 与消息表示开销，保持每个 adapter 的同一
实时容量/表示。普通 Full 是否改善仍须实测，不能从字节减少推断 TTFT 收益。

## 同类实现对照与第一性原则

- [vLLM 0.30.0 model manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/model_manager.py)
  的 `list_adapters` 返回注册映射副本；激活使用注册对象与 slot 表。我们
  新增的每次全 tensor 观测不是该原生查询接口的必然成本。该源码对照解释
  哪项工作是本项目自己引入的，不证明 vLLM 或其他系统总体不存在 CPU 瓶颈。
- 同一官方实现先合并/优化，再 pin CPU 权重；
  [LoRA weights](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/lora_weights.py)
  也含原地 scaling。因此不能仅按 adapter ID 缓存一个永久有效的 footprint。
- [vLLM 官方 CPU 优化分析](https://vllm.ai/blog/2024-09-05-perf-update)
  将对象构建和同步 CPU 工作作为推理瓶颈来源，并区分 frontend 与 engine。
  本次借鉴“消除没有消费者的工作”，不搬用其性能数字，也不把完整验证
  从独立 frontend 搬回 GPU loop。
- [dLoRA 原论文](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
  是请求/adapter 协同的相关参照；论文级设计不能证明其没有当前观测成本。
  本次具体优化依据是已核查的本地调用链与相同版本官方实现，不凭系统名称移植。

落实用户新增要求：后续每次优化继续说明本系统与同类实现的具体差异、
为何该额外工作出现、哪些证据支持改动；不把“其他系统没有这种问题”当作前提。

## 唯一改动与不变量

`_ieee_lora_host_inventory(..., include_tensor_views=False)` 仅由
`routing_source_snapshot` 选择；缺省仍返回原完整描述表。

- 每次重新读注册集合并遍历所有当前 tensor，检查 CPU、dense、2D、非空、
  storage 指针/容量、每个 view 的 pinning、alias 一致性、完整 A/B、额外 tensor。
- 同次 dtype 聚合不再从全描述表按 adapter 重复扫描；结果和完整旧表一致。
- 完整 storage union、共享 owner edges、exclusive bytes 和表示均保留。
- 缺少描述表用“字段不发送”表示，不能发空表冒称没有 tensor。
- frontend 原完整 `_footprints` 验证、owner/epoch/source/clock、selected-copy
  重检查、reservation、实际加载/回收、planner/admission 全不改。
- 完整 source_snapshot、staging、allocator 和预算消费者继续得到同值完整数据。
- 不引入跨调用缓存、TTL、fallback、并发/超时/profile/阈值变化，不改 IEEE 九式。

## 资格与比较状态

| 检查 | 状态 |
|---|---|
| 完整图等价、动态替换/移除、共享、packed、异常拒绝 | 26 项定向检查通过，其中新增 8 项 |
| frontend 图验证与完整旧端点保留 | 通过；坏 union、exclusive 和共享 owner 均拒绝 |
| 受影响回归 | 942 项通过，100.519 s；与定向检查有重叠，不相加当独立证据 |
| 实际保留观测的投影字节和 source 等价 | 四份 source 对象与最终路由消息完全相等 |
| 普通 4000 请求 Full | 未执行；先满足推理机 150 GiB 启动门槛 |

按 academic-plotting 和计划 §11，本轮资格用状态/精确数据表，不将组件
结果包装成 GPU 性能图。原始目录为 `results/ieee_tc/p2_backend_qualification/d174_20261002`。
复用 D144 组件比较方法、D172 有界 CPU 执行器和既有 D169 四份原生快照；
不再解析 D173/D164 大原始结果，不新建权重、trace 或远端缓存。

### 组件证据

复用 D144 的七种微型 CPU fixture，对照 Git 中精确父版本 helper；新缺省
完整输出/错误逐项相同，路由输出精确等于去掉描述字段的完整输出。
共享、partial、packed、staged-alias 四种非空正常样例的 pinning 查询均
保持 6 次，而 stride/element_size/offset 各从 6 次变为 0 次。空集以及
额外 tensor、别名 pinning 冲突也保持相同结果或拒绝。没有用缓存布尔值
代替真实查询，未初始化 CUDA。

| 保留快照 | 注册 adapter 数 | HOST view 数 | 原 JSON 表示字节 | 投影后字节 | 缩减 |
|---|---:|---:|---:|---:|---:|
| 0 | 8 | 2048 | 755574 | 336182 | 55.5064% |
| 1 | 12 | 3072 | 1074629 | 444997 | 58.5906% |
| 2 | 20 | 5120 | 1713722 | 662970 | 61.3140% |
| 3 | 24 | 6144 | 2031752 | 771272 | 62.0391% |

这四份是 D169 真实观测的离线投影，不是新 worker RPC。表中只计相同
JSON 编码规则的表示大小，不冒称 native msgpack 的线上字节、吞吐或
时延；四份也不是四次独立 workload，不出 CI。组件测试使用现有原生
环境 Torch 2.13.0+cu130，但仅 CPU tiny storages，不能代表 CUDA pinned
inventory 的真实成本。完整实际回放仍是必要的效果检查。

失败保留：第一轮 26 项中 1 项失败；负向样例原本给第一条 private
allocation 再写相同 owner，实际未制造冲突。改为定位两个 owner 的共享
allocation，再破坏它，第二轮 26 项通过（0.018 s）。原测试源码、失败
日志和 source SHA 保留；生产候选没有为此改动，异常检查没有放宽。

四个已结束的 CPU 任务均在 3/4 GiB、swap 0、CPU 2,3,26,27 中运行，
末次 memory.events/swap 为零、资源域自动移除已核对。无 GPU 或远端服务
运行。本轮接受的仅是信息表示优化的正确性资格；备份后以相同 D173
cap4/D157 profile 与 W0 4000 请求做一次普通 Full，不重复组件实验寻优。

Plan SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`；
冻结 MetricV1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
当前只优化 7B，数值正确性、共同 reference/Resident、新指标旧 Prime 对照和
G1/G2 仍开放；3B、外部基线及后续 M1/M2、A1–A5、S1–S13 顺序不变。
