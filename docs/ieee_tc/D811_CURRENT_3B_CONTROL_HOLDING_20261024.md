# D811：当前3B控制与物理持卡诊断

2026-10-24。复用D752已关闭的4,000请求、压缩原始控制日志与物理账本；
没有GPU回放、生产机制修改、指标变更、远端操作或共享Kubernetes修改。
采用analyze-results区分观测、解释与下一检验。完整PLAN、V1及单轮V2已重读。

## 直接观测

| 观测 | D752 |
|---|---:|
| 共同联合SLO | 3831/4000，95.775% |
| 生命周期物理GPU-s | 14860.687045 |
| 四个互斥窗：准备前段/到达窗/drain/清理 | 36.229302 / 14781.362242 / 1.720890 / 41.374611 |
| 业务期间控制采样 | 1911 |
| 4副本ready采样数 | 1714 |
| 4副本时活跃请求 mean / median / P95 | 2.305718 / 2 / 5 |
| 4副本时队列 mean / median / P95 | 0.140023 / 0 / 1 |
| 低队列、低活跃度且无pending采样 | 1548 |
| 上述采样rolling P95 TTFT median / P95（ms） | 1514.737560 / 2184.214731 |
| 满足所有缩容低阈值的采样 | 0 |
| 自然扩容/缩容 | 3 / 0 |

三次扩容在业务开始后350.399、360.596、362.642秒，均由active分量超过
上限触发，当时仍只有1个ready副本，pending分别0/1/2；不是D726中启动
首请求TTFT触发扩容的相同原因。pending依法计总副本上限而不冒充可服务容量，
当前日志与IEEE max-score/all-low算术逐项一致，不把合法扩容直接归为bug。

初始ready在业务开始后0.225秒，首请求TTFT2370.917ms；不是11秒首请求
异常的重现。所有物理区间均真实释放，四窗之和与区间积分均等于总U。
控制采样有时间相关性；活跃请求占比不是GPU utilization，亦不是三卡容量证明。

## 对等待的解释范围

到达至admission平均289.940ms，native engine queue仅1.658ms；GPU命中
请求到达至admission仍299.586ms。87条TTFT违约中，60条的三个入口等待项
以内含owner确认的runtime-slot envelope最大；另有12条dispatch-window、
15条arrival-release最大。不能把这60条一概归因于GPU槽耗尽或纯IPC。
43条TTFT违约来自非Remote层，说明远端传输并非全部尾延迟来源。Remote
获取133条请求与132次实际fetch不同，保留各自定义，不人为对齐。

当前控制仍使用219.518ms的source-only历史下限，而信号为overall TTFT。
这解释没有all-low，但不足以证明提高下限会改善系统。D712/D713已用同路径
标定下限2311.815ms验证，联合仅833/1000，6次扩/缩容，已拒绝；本轮不重试。
D750/D751/D752已降低控制校验开销并使共同联合从94.675%升至95.775%，
却没有降低U，因此下一步不继续堆叠同类微优化来声称解决生命周期问题。

## 下一检验及边界

登记D812：只把3B模型级max_instances由4改3，min1及全部控制阈值、窗口、
profile、批量、缓存、admission、物理四卡可见范围保持不变。假设是更少的
管理副本可在当前负载下保留共同SLO并减少真实持卡；反例是缓存/准备竞争与
排队增加导致联合不达标或排空使资源收益消失。不能由本日志计算反事实TTFT
或直接减去一张卡区间冒充新结果。7B三副本成功只提供历史经验，不证明3B。

联网核查：

- [Llumnix OSDI 2024 §6.5](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)
  同时评价尾延迟与实例使用，并比较不同扩缩容设置的质量—资源关系；不提供
  本项目三副本必然达标的依据。
- [Ray Serve官方说明](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html)
  将min/max边界与延迟、策略分别配置；
  [ray-2.59.0源码](https://raw.githubusercontent.com/ray-project/ray/ray-2.59.0/python/ray/serve/autoscaling_policy.py)
  将观测策略和持续方向延迟处理分层。这里只核对部署边界语义，不移植Ray
  控制律，也不把max=3称为自动学习结果。文档源码入口429后使用官方固定tag。
- 当前IEEEReplicaControl最近相关接通提交faf5c85及D614/D632已核对；论文
  允许configured replica limit。现有3B配置目录未见max3点，不声称全机历史无此点。

## 产物与资源

`paper_results/ieee_tc/p2_backend/d811_current_control_holding/diagnosis.json`
包含来源SHA、动作、原生阶段分组及物理区间，manifest不覆盖旧分析。
复用D727解析器并流式读取既有gzip，不恢复1.25GB解压副本；未重建数据。
辅助域峰值930242560B，high/max/OOM/swap均0，递归空后已关闭。
147项投稿保护及三个用户dirty SHA不变。数值正确性未知、单轮已知负载身份
继续保留；后续W1/W2、全部消融/敏感性、审稿说明和最终交付不删减。
