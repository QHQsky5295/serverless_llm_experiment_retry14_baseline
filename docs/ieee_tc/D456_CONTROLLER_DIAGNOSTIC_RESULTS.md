# D456：当前配置控制路径诊断结果

2026-10-12。一次原始前缀 1,000 请求，带 Python 帧观测器；不是普通 Full
性能点，不与无观测器的完整 W0 排名，不修改已封存的 D371/D454。

## 数据与资源核验

| 项目 | 本次观测 |
|---|---:|
| 完整原生生成/请求/adapter 绑定 | 1,000/1,000 |
| 原生输出 token | 122,510 |
| 共同阈值联合达标 | 928/1,000（92.8%） |
| 平均 / Type-1 P95 TTFT | 1,646.725 / 7,273.100 ms |
| 平均 dispatch/admission 等待 | 949.184 ms |
| 平均 service TTFT | 697.541 ms |
| 平均 / P95 TPOT | 33.818 / 59.054 ms |
| 实际生命周期 GPU 时间 | 3,854.234795 GPU-s |
| 真实 HTTP 获取 / 线上字节 | 60 / 74,953,014 |
| 请求内打包 | 0 |
| 服务内存峰值 | 20,021,817,344 B |
| high / max / OOM / swap | 均为 0 |

D434 独立保留日志逐请求验证相同源条目、adapter、target、canonical prompt
hash 和原生 prompt-ID hash。未知数值正确性字段仍为 null；绑定完整不等于
独立证明 CUDA kernel 数值正确。两端 transfer ID、内容 SHA、字节及发布状态
一致。initial 1；natural_scaleout 9 个触发中 8 个 ready、1 个 cancelled，
不能称为 9 次完成扩容或 first-service。初始启动 52.289 s。

GPU 实际释放和服务资源域移除后才停止本次精确身份的远端 3B/7B/monitor。
旧只读 D78/D80 缓存复用，无重新发布、解压池、权重/trace 生成或基线优化。

## 函数证据及解释边界

完整记录包含 1 controller、1 planner、10 frontend、10 GPU core。
主控制线程业务窗口有 538 次采样，以下为包含该函数的样本数，可相互重叠：

| 路径 | 样本数 |
|---|---:|
| 驻留 epoch 执行 | 129 |
| 单副本准备规划入口 | 116 |
| 文件层 `preparation_snapshot` | 90 |
| `_planning_source_observations` | 43 |
| `_local_file_inventory` | 25 |
| `_file_replacement_capacity_from_inventory` | 17 |
| 规划来源发布 | 21 |
| RPC 响应反序列化（最近项目帧） | 18 |

最近项目帧为 `main` 的 343 次样本不能解释成 main 函数自身计算；示例实际
停在 asyncio selector。文件快照的样本进一步定位到 lstat、路径祖先/相对路径
构造和可替换容量索引。该结果支持优先检查**同步状态构建**，不是重新把纯
背包规划外移一次：纯规划已在独立进程。普通请求的锁等待、文件变化、物理
容量及执行重检查不能为了减小观测开销而省略。

185 次带边界记录的调用全部成功完整、无 partial：native 观测段均值
10.593 ms，native 线程 CPU 10.092 ms；frontend ready 到 parent received
均值 36.882 ms（P95 189.747 ms）；parent 全程均值 169.239 ms。
这些不是全部请求，也不是纯网络/排队时间；并发调用重叠，不累计为 TTFT。

控制进程最后样本累计 CPU 466.585 s；观测线程在最后一次采样前累计
0.417 s。后者不包括最后一次采样，亦不能证明所有观测扰动仅有 0.417 s。
最大采样调度迟到 0.738 s。计数不是 CPU 时间占比，GIL 和 C/CUDA 不可见
部分、1,000 前缀覆盖不足及没有无观测器 A/A 都限制因果推断。

## 与历史及原始资料的对应

D455 已保留完整 W0 后段到达/完成接收延迟增长；本次提供当前 cap14 配置
函数线索，不移植旧 C8 的 D328 性能数字。D93/D153、D188/D193、D329–D331
已有单次库存、静态描述与状态发布优化，下一候选必须检查这些历史，不能
恢复全量重复读取，也不能用 TTL/epoch 替代真实物理状态。

[Python asyncio 文档](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
说明事件循环上的同步工作会推迟其他任务；
[vLLM 官方 CPU 路径分析](https://vllm.ai/blog/2024-09-05-perf-update)
支持分别检查控制与输出路径。这只是方法依据，不借用其加速比例。
[文件枚举接口](https://docs.python.org/3.12/library/os.html#os.scandir)
可用于评估表示/遍历成本，但缓存的 DirEntry 属性不能代替必须的新鲜复查。

下一步：沿这组真实调用检查重复路径构造和文件视图成本，登记一个保持
owner/内容/容量/取消语义的候选；先等价性与最小验证，再完整普通回放评价
净收益。不能从本次采样预言加速幅度，不放松共同 SLO 或控制阈值。

原始目录 `results/ieee_tc/p2_backend_qualification/d456_20261012/`；汇总位于
`paper_results/ieee_tc/p2_backend/d456_controller/`。复用 D328/D200 栈分析、
D453 请求/远端核验及 D452 独立输入见证，只变更 campaign/count/当前 schema。
主线仍为必要 PrimeLoRA 优化、7B W1/W2/G2、3B、全部消融/敏感性与论文证据。
没有生成图，没有宣布系统最优或完整实验计划完成。
