# D816：请求必需物化与可选准备的文件容量协议

## 问题、历史与可证伪判断

D814 的 W2 在第3285号请求获取远端headers后、本地空间预留时失败；随后
同键主动准备遇到已失败movement是后果。D815复用真实请求入口证明：24KiB
预算内已有三个可回收副本时，20KiB峰值的请求路径失败，而相同布局的主动
objective路径能够回收并完成16KiB发布。这个反例不证明D814当时所有副本均
可驱逐，但证明Full的请求必需容量路径缺失，不能靠扩大预算处理。

核对D277（GPU lease）、D449及commit48265b8c（只为普通demand_lru接入文件
回收）、D609/D627/D631/D680和IEEE Eq.(4)–(7)/replacement段落；既往优化
没有实现此路径。本轮仅修正这一个假设，不再重复已关闭的元数据/GC/加载
批量化方向，不改变autoscaler、需求窗口、profile、后端或SLO。

## 协议及论文边界

| 论文规范语义 | 当前实现证据 |
|---|---|
| 主动准备按需求加权收益、层级预算和替换目标选择 | 原h/d、DP、benefit-loss替换不变；Full仍为planned_only |
| 已到达请求的必需加载优先于可选准备 | owner登记live required interest，分配事务中执行byte-bounded request LRU |
| 使用中、在途、共享/外部及未结算字节不能冒充空闲 | 原引用/物理保护保留，required目标额外受保护，全部victim选择成功后才删除 |
| 暂无空间不等于永久错误 | 有合法movement的临时冲突返回deferred；由真实owner事件唤醒，无sleep轮询 |
| 可选准备不能任意破坏收益选择 | 没有objective的中间staging只等待；真实请求加入后才能按请求路径回收 |
| 请求撤销不应留下无主权限 | interest随实际等待/IO清理退出；撤销后可选准备不再拥有必需加载权限 |

request_lru_v1只依据发布/读取的可观测先后顺序，不用未来需求、场景名、
新系数或扩大容量。它是普通请求缓存服务语义，不包装为研究贡献，也不把
Full整体切为ElasticLocality的demand_lru。原主动目标保护、shared bytes及
原生HOST余量可能使请求等待；单个必需传输本身超过tier绝对预算仍明确失败。
冷源发布到真实read lease之间保持请求所有权，避免刚完成副本被并发回收。
替换事件保存policy、required intent IDs、victim及物理字节；snapshot保存
live movement interests。没有修改IEEE九个行间公式或投稿源文件。

官方参考是[vLLM 0.30 cache.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/utils/cache.py)
中touch/pin/popitem对可观测recency与pinned保护的分离，以及
[worker_manager.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
的请求加载与CPU对象LRU。这里只借鉴普通请求缓存原则；不复制其临时对象
超额行为到物理文件预算，也不声称它证明本协议性能。

## 验证、资源与局限

- 四个受影响测试模块共457项通过，20.912秒；加入reader-release案例并加强
  真实HTTP取消覆盖后，8项聚焦测试通过，0.728秒。两组有重叠，不相加为465。
- 真实_ensure_local_async、同writer合并、请求取消、reader/plan释放、发布到
  pin间隙、oversize明确失败均覆盖；生产两文件在两次测试间SHA相同。
- broad/追加测试内存峰值722972672/702898176B，high/max/OOM/swap均0；
  147个旧投稿保护文件及三处用户修改SHA不变。CPU受限，无GPU回放。
- 测试源与回执在paper_results/ieee_tc/p2_backend/d816_required_file_capacity/；
  raw在results/ieee_tc/p2_backend_qualification/d816_20261025/。
- 这些是微小真实文件的正确性证据，不是性能数据，也未解决D711的独立数值
  正确性未知。没有把失败实验删掉或改成成功。

下一为新版本D817完整3B W2，复用D814全部参数/trace/map/生成/远端交付，
只改输出与cold-cache路径。无额外GPU smoke，不重跑原键。保留D812/W0、
D813/W1、D814/W2失败，合看三场景之后再决定性能方向。W1仍未达联合SLO，
全任务未完成；同最终版本主表/Full、A1–A5、S1–S13及审稿交付全部保留。
