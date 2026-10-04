# D214 — confirmed-route identity recheck with conditional footprint reuse

日期：2026-10-04。范围：PrimeLoRA 7B，单一开发候选；尚未进行 GPU
性能回放。

## 动机与可证伪假设

D210 的请求级投影显示，`routing_decision_us` 平均约 325.9 ms、P95 约
1,075.3 ms；这段时间包含每个副本一次完整的
`request_source_snapshot`，而该请求级路径只需要确认“当前副本是否仍是同一
native source/epoch，以及所选 copy 的身份”。D209 已把选中低层副本复核改为
身份接口，但首次 IEEE 路由仍每次构建带 footprint 的完整 scoped observation。

可证伪假设是：若每个请求仍先读取所有副本的实时 owner identity，而在
`owner_id/epoch/clock_id/slot/source_id/path/rank/tier` 完全不变时复用该副本
上一次已确认的 scoped footprint，则可以减少路由控制路径的物理图构建时间，且
不会把 last-known tier 当作当前确认状态。若身份读取失败、epoch 改变、native
条目未知/未确认、目标 footprint 不完整，必须退回原始完整 scoped observation。

## 语义不变量

1. `source_identity_snapshot` 是每个请求的新 owner 读取；没有 TTL、轮询预测、
   最近一次层级直接路由或 demand cache。
2. 只缓存不可变的 `NativeSourceSnapshot` footprint 描述。缓存命中会以本次
   identity observation 的 owner、epoch、clock、slot 集合和 capture time 生成
   新对象，再交给现有 `commit_native_sources`/`confirmed_source_class`。
3. native owner 的 epoch 在 source/slot/reference 状态刷新时推进；epoch 或
   source identity 任何变化都会导致 cache miss。未知或未确认 native ID 不得被
   identity-only 状态解释为 Remote/Host。
4. 文件 tier 仍由每次 `owner.source_snapshot(adapter_id)` 与原有
   `confirmed_source_class` 重新校验；缓存不绕过 selected-copy revalidation、
   admission、lease、budget、load 或 execution commit。
5. 九个 IEEE 公式、service/preparation profile、GPU admission 和生命周期记账
   均不变。开关默认关闭，未冻结为正式配置。

## 实现证据

`ScenarioRunner._ieee_collect_route_identity_cache` 是
`resource_coordination.ieee_route_identity_recheck_cache` 显式配置键下的唯一
候选路径。该键属于控制器，不参与冻结的模型/service profile 身份。它为每个副本并发调用
`ieee_source_identities()`；命中时复用同一 `(instance_id, requested_scope)` 的
完整 scoped 状态，未命中时调用现有 `ieee_request_sources()`。身份 RPC 异常也
会对当前成员集合执行完整回退。新结果只写入 `_ieee_route_full_cache`，不写旧
结果目录，也不更改历史 D210 数据。

CPU 资格测试覆盖：相同 epoch 的 footprint 命中、epoch 改变强制完整读取；既有
请求 footprint/lifecycle 测试集仍全部通过。CPU 测试不是性能结论；下一步必须
在 7B 同一 W0、同一 D157 profile、同一真实远端发布工件和同一资源隔离协议下做
一次 100-request validation replay。只有当 native contract、dispatch tier、
identity/full fallback 计数和清理均通过，才允许进入一次完整 4,000-request
普通 Full 复核；若 TTFT 或正确性不改善，则撤回候选，不继续局部调参。

## 统计与解释边界

运行时记录 `hits`、`identity_only`、`misses`、`full_fallbacks`、
`identity_failures` 和 `membership_rejections`，使图表能区分“减少物理图构建”
与“改变路由语义”。任何候选回放在共同 Resident/reference、G1/G2、数值
adapter 正确性冻结前都不是 baseline 排名证据。
