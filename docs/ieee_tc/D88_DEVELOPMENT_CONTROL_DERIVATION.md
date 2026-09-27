# Development Control Derivation Package — D88

## Target

为首次 IEEE Full 集成明确一套有来源的开发起点；保持论文九式、max-score
扩缩容、EWMA 更新式和服务时间分桶不变。本次不是推导全局最优策略，也不是
重新选择共同 SLO。结果以显式配置交给现有控制器，不新增在线预测器。

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**：容量关系可以在下述假设下
推导；启动时间、GPU-ready token 间隔和校准轮数用于选择参数属于开发启发式，
不是由公式唯一决定的最优值。正式 G1/G2 仍需完整共同协议比较。

## Invariant Object

相同正确工作负载、共同服务要求下的物理 GPU 生命周期，以及同预算下的尾延迟。
这里仅处理使第一次完整实现可检验的控制配置，不能用参数自洽替代上述目标。

## Assumptions

- 当前每模型副本同构、TP=1，接纳容量 C 固定且 C≥2，最少副本 m≥1。
- 缩容只回收已排空副本；在该瞬间未改变其余请求数和每副本接纳容量。
- D84 是已完成的开发测量；身份仍为 source-only，不重标成新准入配置。
- 仅使用代表性测量轮次，排除 warmup；原生首末 token、同一时钟和保护状态
  必须可校验。候选配置、原始日志和生成器都有 SHA；缺失值不补零。
- 首次控制上界沿用历史开发 TTFT 目标 5000 ms。这不是未来共同 warm-reference
  标定，也不是对全部 baseline 规定的最终 SLO。
- EWMA 权重平方和的代数关系不要求独立性；把它解释成方差等效样本量需要
  独立同方差近似。当前三轮在同一运行内，不能声称该统计假设已经满足。

## Notation

- C：实际单副本接纳容量；m：最少副本；n：当前副本数。
- A：当前 admitted 请求数；u=A/(nC)：总体接纳饱和度。
- W：共同 demand/completion window，沿用已声明开发候选 5 s。
- Δ：历史控制评估间隔，两模型均为 2 s。
- τ：D84 实测单副本启动秒数；c：开发缩容 cooldown；V：TTFT 观测窗口。
- R：预先声明且实际覆盖 GPU 样本的校准轮数，当前为 3，不是独立运行数。
- δ：路由服务时间分桶宽度；β：既有 EWMA 式的固定更新系数。

## Derivation Strategy

先按容量单位明确上下界，再以实测启动和 token 间隔确定时间尺度，最后将
EWMA 的记忆深度与现有小规模初始化的支持量对应。所有规则在完整回放前固定；
不读取未来到达、热点或实际生成长度，不按 W0/W1/W2 名称切换。

## Derivation Map

1. 接纳容量与空队列条件 → queue 和 active 的显式界限。
2. 回收一个已排空副本、A 不变 → 回收前后饱和度关系。
3. 既有启动测量和控制采样 → cooldown 与观测保留长度的开发选择。
4. 既有 GPU-ready 服务间隔 → TTFT 下界与路由时间尺度的开发选择。
5. EWMA 权重平方和 → 记忆深度的一种解释；不构成性能最优性证明。

## Main Derivation

**Step 1 — interpretation / design choice.** 令 queue upper=C，表示允许排队量
以一个副本接纳容量为单位；queue lower=1，与控制器的严格小于结合，缩容时
必须无等待请求。并不声称 C 个请求必然在一个 GPU iteration 内处理完。

active upper=(C−1)/C，以每副本一个接纳位置为容量余量。在一个副本时，严格
超出该界意味着接纳容量全满。此界限不是已证明的吞吐拐点。

**Step 2 — conditional algebra.** 选择 active lower=active upper×m/(m+1)。
若 n≥m+1，且 u 小于该下界，则回收一个已排空副本后：

\[
u'=\frac{A}{(n-1)C}=u\frac{n}{n-1}
<u_{upper}\frac{m}{m+1}\frac{n}{n-1}\le u_{upper}.
\]

这只证明当前总体接纳容量比率没有立即越过上界。它不保证 adapter 分布均衡、
未来到达稳定、KV 足够、TTFT 达标或控制器全局稳定；物理 admission 仍独立检查。

**Step 3 — development heuristic.** 采用
c=Δ ceil(max(W,τ)/Δ)，V=c+W+Δ。保留旧系统以启动成本考虑回收等待的动机，
但只使用已经观测到的启动时间，不读取未来 idle gap。多出的窗口长度让一条
已知低 TTFT 观测可以覆盖整个 cooldown，避免 V<c 导致纯粹因过期无法完成
全低判定。旧高 TTFT 样本仍会延迟回收；缺观测仍是 unknown，不改写成零。

τ 不是未来冷启动上界；该简单单副本观察也不是完整经济 break-even 定理。
本候选不为了 W1 的 30 s 间隔把 c 压到 30 s 以下。

**Step 4 — development heuristic.** TTFT lower 取 D84 代表性 GPU-ready 请求
acquisition→first 的 Type-1 P95；必须严格低于历史开发 upper，否则拒绝本候选。
δ 取相同 GPU-ready 样本的原生 TPOT Type-1 中位数。它表示参考的一个输出
token 时间尺度，不称 kernel iteration 时长、估计误差上界或最佳分桶宽度。
原 floor 分桶可能把相差小于 δ 的两值分到相邻桶；此边界效应不被隐藏。

**Step 5 — identity plus heuristic.** 固定 EWMA 的稳态权重 w_j=β(1−β)^j
满足 Σw_j²=β/(2−β)，因此倒数为 (2−β)/β。选择该记忆支持量为 R，得到
β=2/(R+1)=0.5，用于 service 与 preparation 的原更新式。将轮数用作记忆
支持量是开发选择，不把三个校准轮次当作三个独立重复，也不据此计算 CI。
该权重与方差解释参考 [NIST EWMA 说明](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm)。

## Remarks and Interpretation

| 开发配置 / 实测依据 | 3B | 7B |
|---|---:|---:|
| C / GPU-ready 样本 | 8 / 72 | 2 / 18 |
| 原始 τ (s) | 47.324936 | 37.521824 |
| c / V (s) | 48 / 55 | 38 / 45 |
| queue upper / lower | 8 / 1 | 2 / 1 |
| active upper / lower | 0.875 / 0.4375 | 0.5 / 0.25 |
| TTFT upper / lower (ms) | 5000 / 219.518056 | 5000 / 281.278192 |
| δ (ms) | 28.114718 | 27.433561 |
| service / preparation β | 0.5 / 0.5 | 0.5 / 0.5 |

[Knative 的 concurrency 说明](https://knative.dev/docs/serving/autoscaling/concurrency/)
区分硬容量与软目标。这里借鉴这种区分，不照搬其默认百分比、不声称实现了
Knative，也不以它证明 Prime 的控制规律最优。IEEE 的三个量 max-score 与
全低 cooldown 语义没有改变。历史 RPS EWMA、instance-busy fraction 不当作
admitted-request saturation 使用。

## Boundaries and Non-Claims

这次完成的是参数选择、来源记录和实际控制器的确定性检查，不是新的性能
实验。新 contract 的 model_config 与 D87 共同组装结果严格相同；新 source
spec 加入已完成长度初始化和 contract SHA。source collector 不运行 Full
autoscaler，记录控制 contract 只是避免下一次集成再换配置身份。

不得据此越过 Full 资格检查或将 D84 性能贴上新标签。新的准入路径测量后，
还需匹配 profile 导出、完整 activation/lifecycle、开发回放和共同 SLO 标定。
若开发期证据要求改变控制规律，另建明确版本并完整验证，不能筛正式结果。

## Open Risks

- 48/38 s cooldown 可能让 30 s phase gap 内没有自然回收，影响 G1；如实测量。
- source-only 初始化时延与启用准入后的动态性不同；新测量不能省略。
- 低 TTFT 下界可能过严、旧高样本可能使回收滞后；必须看完整回放。
- β 与 δ 的启发式可能响应过快或对某些观测类不理想，不宣称已优化。
- 零权重集合的数值区分性限制仍在，不用参数检查替代正确 adapter 的证明。
