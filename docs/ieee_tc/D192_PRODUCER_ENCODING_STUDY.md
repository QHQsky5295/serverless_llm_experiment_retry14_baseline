# D192：生产端 RPC 编码小验证——不采用当前原型

2026-10-03。CPU 组件研究，不是新增模型性能运行。D191 已备份为
`ba299b41cfb359e3f772c65954a5bf02b2896bef`；生产代码仍为 D190 所用版本，
九个公式、配置、冻结 V1、工件、trace、远端服务与交付缓存均未改变。

## 1. 问题与可证伪假设

D191 的当前请求分解显示引擎前平均等待 2.101 秒，但不能把它全部归因于
某个 RPC。D187 观察到编码调用链；D172 只优化解码，未验证生产端编码。
本次假设是：高效 JSON 编码在保留必要类型/有限值检查后，仍能减少控制
线程的同步工作。若检查成本抵消收益，或当前消息已很小，就不推进线上替换。

复用 D172 的三轮交替测量与 D169 已保存四份真实来源状态；复用当前
`NativeSourceSnapshot` 的 routing/request/identity 投影。请求投影取各快照
首个已有 adapter，只用于消息表示测量，不生成新请求、权重或 trace。
这些是历史状态的当前格式投影，不冒称 D190 所有 RPC 的实测大小分布。
历史完整 native graph 单列，不能当成当前紧凑 frontend 回复。

## 2. 官方依据与语义边界

[Python JSON 文档](https://docs.python.org/3.12/library/json.html) 的默认路径
与 [msgspec 类型说明](https://msgspec.dev/supported-types) 在非有限浮点、
扩展类型及 Unicode 表示上存在区别。msgspec 将非有限浮点编码为 null，
因此不直接替换。研究原型先遍历并验证 exact builtins、字符串键、有限
浮点和无循环结构，再编码；全部验证成本都计入，未使用 fallback 或 null 替代。
[vLLM 0.30.0 的序列化实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/serial_utils.py)
只提供高效表示的参考，其 MessagePack/张量缓冲并未移植到本项目。

四项有效 fixture 与 17 种消息的两种解码结果，在类型、键顺序、嵌套字段、
整数及浮点位上逐项一致。但这不是全域协议等价：13 个边界检查明确显示，
非有限数和 lone surrogate 的拒绝阶段改变，整数键会由接受变为拒绝。
8 MiB 仍约束实际 body bytes；紧凑空格/UTF-8 表示使部分原先超限的对象
可以装入，故不能宣称接受/拒绝集合完全一致。真实生产端取消与所有权边界
亦未做新资格；本次不将原型安装进服务。

## 3. 完整组件结果

单位 ms/次；每项三轮交替顺序，大 graph 每轮 20 次，其余每轮 200 次。
原型列包含全部递归验证。下降比例为 `(原−原型)/原`，负值表示变慢。
这是组件测量，不是三个独立 workload/seed，不生成服务 CI 或 TTFT 外推。

| 快照 | 消息形态 | 原 bytes | 原型 bytes | 原 ms | 校验＋原型 ms | 下降比例 |
|---:|---|---:|---:|---:|---:|---:|
| -1 | existing_small_primitive_fixture | 305 | 262 | 0.008337 | 0.006725 | 19.333% |
| 0 | current_identity_projection | 3259 | 3088 | 0.032324 | 0.029949 | 7.346% |
| 0 | current_request_projection | 3578 | 3393 | 0.034334 | 0.034153 | 0.527% |
| 0 | current_routing_projection | 5249 | 4994 | 0.045552 | 0.046059 | -1.114% |
| 0 | historical_full_native_graph | 1518630 | 1388829 | 23.211254 | 33.187543 | -42.980% |
| 1 | current_identity_projection | 4554 | 4315 | 0.044102 | 0.042241 | 4.220% |
| 1 | current_request_projection | 4873 | 4620 | 0.047435 | 0.046036 | 2.950% |
| 1 | current_routing_projection | 7496 | 7133 | 0.065522 | 0.064365 | 1.766% |
| 1 | historical_full_native_graph | 2215094 | 2025713 | 26.546090 | 44.937627 | -69.282% |
| 2 | current_identity_projection | 7143 | 6768 | 0.056782 | 0.056885 | -0.182% |
| 2 | current_request_projection | 7462 | 7073 | 0.058807 | 0.059338 | -0.903% |
| 2 | current_routing_projection | 11989 | 11410 | 0.089873 | 0.091591 | -1.911% |
| 2 | historical_full_native_graph | 3609914 | 3301361 | 40.924184 | 71.698431 | -75.198% |
| 3 | current_identity_projection | 8433 | 7990 | 0.065369 | 0.062185 | 4.870% |
| 3 | current_request_projection | 8750 | 8293 | 0.067025 | 0.066686 | 0.507% |
| 3 | current_routing_projection | 14229 | 13542 | 0.104243 | 0.104764 | -0.500% |
| 3 | historical_full_native_graph | 4304329 | 3936194 | 47.799010 | 82.341437 | -72.266% |

小 fixture 和部分紧凑消息有微小观测下降，但方向不一致，不能支持系统收益。
四份完整 graph 的原型均更慢；当前格式投影的原编码仅约 0.032–0.104 ms。
不能用旧 1.5–4.3 MB graph 的体积宣称当前每个路由 RPC 都有大编码成本，
也不能把不到毫秒的组件差直接解释成数百毫秒等待的原因。

## 4. 决定、资源和下一步

**不采用这个“Python 全遍历校验＋msgspec 编码”原型；不启动相应 Full。**
这不证明一切编码优化都不可能，只说明本次候选没有相应证据。不去掉检查
重新追求好看的速度，也不反复微调这一原型；一次有界测量即归档。

任务实际身份 `1032ef3910434c38a2c299eef201b087`；CPU2,3,26,27，
MemoryHigh/Max=3/4 GiB，swap0；msgspec 0.21.1，native Python 环境。
测量命令 wall 47.12 s、RSS 峰值 1,159,388 KiB；退出0，保存的内存事件与
swap 为零，scope/实际身份/路径已释放。17 形态、102 原始计时行、13 边界
检查完整保留；147 项旧结果前后保护通过。无 GPU、远端或生产代码操作。

下一步返回 D191 已有控制阶段与调用依赖，检查真正位于关键路径上的等待，
尤其是请求推进、资源释放通知与准备工作之间的依赖；先核对历史和官方实现，
再决定一个新假设，不立即重跑或选新配置。当前不支持任何新性能优化结论。

7B 的数值正确性、130 条输出 hash 差异、共同 warm/Resident、旧 Prime
新指标及 G1/G2 仍未闭口；3B、外部基线、主比较、消融和敏感性全部保留，
按原先顺序待执行。新重型任务仍须通过原 150 GiB 门槛，不降低安全要求。

依据 `analyze-results` 分开观察与解释；依据 `academic-plotting` 和计划 §11
交付完整精确表，不制作没有系统性能含义的优胜图。
原始记录：`results/ieee_tc/p2_backend_qualification/d192_20261003/`。
整理数据：`paper_results/ieee_tc/p2_backend/20261003_d192_producer_encoding*`。
