# D304：复用完整回放定位解码尾部，未新增性能实验

2026-10-07。继续7B主线，复用D300/D303各4000条请求及真实activation记录。
已完整重读源计划和冻结指标V1。没有修改服务、模型配置、九式、阈值、
工件、负载或远端；没有画图、重新推理、重测warm或推进外部baseline。
本项使用analyze-results的观测—解释—下一验证流程，优先排除错误归因。

## 1. 问题与观测定义

D303相对同配置D300的P95 TPOT上升10.203%，虽然平均/P95 TTFT和物理GPU-s
下降。先检查“是否仅由更频繁扩容引起”，不依据一个最慢请求下结论。

复用原生同主机单调时钟：每条请求的decode区间为first-token至last-token；
其他副本startup区间为activation.trigger_at至engine_ready_at。重叠取区间
并集，不相加重复区间。实际dispatch必须晚于对应副本engine-ready。
同时记录同副本其他请求的非GPU准备包络（admission至native dispatch），
以及native dispatch至last-token的请求占用。后者不是GPU kernel忙碌时间。

输出分组2–32、33–128、129–256只作本次事后描述，不改变任何在线规则或
冻结SLO长度分组。使用原D170临时TPOT阈值；下文“超阈值”均非正式SLO失败。
全量逐请求区间、全部分组和来源在
`paper_results/ieee_tc/p2_backend/d304_decode/analysis1/`。

## 2. 完整条件组结果

| 条件 | D300请求数/超阈值数 | D303请求数/超阈值数 | D300平均TPOT ms | D303平均TPOT ms |
|---|---:|---:|---:|---:|
| 全部 | 4000/118 | 4000/139 | 33.285 | 34.464 |
| 未与其他副本startup重叠 | 3803/94 | 3654/103 | 32.595 | 33.105 |
| 与其他副本startup重叠 | 197/24 | 346/36 | 46.597 | 48.816 |
| 未与同副本非GPU准备包络重叠 | 2325/33 | 2151/27 | 29.731 | 29.903 |
| 与同副本非GPU准备包络重叠 | 1675/85 | 1849/112 | 38.217 | 39.769 |
| 输出2–32 tokens | 1114/115 | 1114/131 | 43.327 | 46.057 |
| 输出33–128 tokens | 1442/3 | 1442/8 | 29.603 | 30.275 |
| 输出129–256 tokens | 1444/0 | 1444/0 | 29.214 | 29.703 |

其他native请求的decode期间时间加权占用均值由1.636升至1.801。与startup
重叠组分别为3.203和3.264，明显不同于未重叠组1.555/1.663；负载同时变化。
业务开始后的startup区间并集为139.775/225.824秒，包含尚未结束的initial
区间，而不是将其重新分类为natural scaleout。完整activation类别均保留。

观测与解释：

1. D303的139条超阈值请求有103条不与startup重叠，因此“扩容启动是全部
   解码尾部问题的原因”不成立。startup仍可能参与部分退化，不能据此排除。
2. 131/139条属于短输出。对于相同绝对暂停，少量decode步数的平均TPOT更
   敏感；这是待进一步核验的机制解释，不是允许删除短请求或重设SLO。
3. 112/139条与同副本准备包络重叠，值得进一步收窄。该包络还含排队、
   文件准备和多个原生调用，不能把整个重叠秒数称为“GPU加载阻塞”。
4. 重叠分类使用实测decode长度，慢请求本身更容易发生重叠；各条件组的
   请求、并发和启动时段均不同。以上不是配对因果效应，不做反事实扣减，
   不把8000条请求当8000次独立实验，不给单次运行CI。

## 3. 历史与官方实现复核

- D176已消除已持有HOST对象的不必要二次全量查询；D184已重叠pending与
  准备流水；D202已去掉无消费者的HOST描述表。不得把这些旧优化重新提出。
- 当前`IEEEBackendGPUReferences.acquire`仍保留完成屏障和原生pin，最初
  来自`cd5b68d`。本次先读原始receipt：4000条均有completion_fence_ms，
  平均1.266588ms、Type-1 P95为0.164327ms、最大121.364890ms。这个字段
  不能代表release屏障或其他调用；也不支持直接取消所有完成屏障。
- [vLLM 0.30 UniProcExecutor官方源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
  在返回Future之前仍同步执行worker方法。外层await或non_block标识并不
  保证CPU准备工作已离开推进decode的线程。这与本机扩展的执行方式相符，
  但没有证明本次每个停顿的原因。
- [vLLM 0.30优化文档](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)
  说明chunked prefill优先decode，以及iteration预算存在TTFT/ITL权衡。
  当前chunked prefill已开启，不能把开启旧功能当新优化，也不照搬大GPU的
  token预算推荐。继续保持1024预算，不依据这组条件均值直接换参数。
- [同版本LoRA worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  是下一步检查原生load/activation边界的参照；不能将Prime新增的观测成本
  泛称为所有Multi-LoRA系统不可避免的代价。

## 4. 下一项可证伪问题，而非立即改服务

仅继续一个问题：**同副本的原生加载事务是否实际覆盖短请求的decode尾部，
还是当前宽准备包络主要反映更高并发？**

现有4000条receipt已经保留`native_load_invoked`、`native_load_started_monotonic_s`、
`native_load_completed_monotonic_s`、`acquired_monotonic_s`和completion_fence_ms，
可以先用原始窄区间复核，无需新增GPU实验或增加观测开销。GPU-hit的native
load时间明确为null，不能填零并冒称一次测量。原生load区间含load及acquire，
仍不是孤立的H2D或独占CPU时间。保留加载/非加载与并发条件、全部请求。

若窄区间不支持主要阻塞，就归档这一假设并转向可测的其他来源；若支持，
再审查哪些工作能离开推理推进线程、哪些必须由同owner原子检查，先验证
线程/生命周期/对象身份不变量，后做最小实验和完整回放。不得放宽pin、
复制完成、预算或confirmed-state语义，也不先改人工阈值、并发和NUMA配置。

## 5. 校验、资源和主线

13项CPU测试通过（新5项区间测试＋原8项timing-bound），覆盖同钟、顺序、
副本身份、重复ID、并集、端点、self排除和空组N/A。实际两次4000条重算与
原TPOT/超阈值计数一致。所有逐请求输入来自先前封存源SHA；D303与D300
共同输入核验复用D303/context1，不重新制作工作负载。

分析及检查均3/4GiB、swap0、CPU2,3,26,27；主分析峰值742457344B，
high/max/OOM/swap均0。无新增GPU实验。旧结果、用户manifest与服务代码不变。
summary SHA `4f7338731ae2eee6f88888a1e1c4d2c28cf90bef45c9badfb11a03c49b148e3f`。
17项manifest输入/输出SHA复核通过；147项保护清单仍仅既有baseline脚本
差异，原始pass=false不改。六个有界CPU资源域均已无进程并按身份停止。
磁盘约160.62GB，仍低于150GiB新重型任务门槛；下一GPU前必须安全清理
已结束的可重建缓存，不能降低门槛或删除原始证据。

7B仍未完成正式正确性/SLO/GPU-s验收；3B、外部baseline、主比较、消融和
敏感性矩阵全部保留待执行，不因本次定位进展跨阶段。
