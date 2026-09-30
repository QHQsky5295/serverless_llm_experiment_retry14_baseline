# 共同 warm reference：既有入口的复用边界

本文件是 D119 7B 完整回放期间的只读源码审计，不是新的性能测量、配置选择
或指标协议修订。当前参考数值仍未冻结。执行顺序不变：完整 Full 验证后再标定。
依据为已批准计划 §4.6 及 `METRIC_PROTOCOL_FROZEN_V1.md` §5。

## 已能复用的部分

| 既有入口 / 证据 | 可复用内容 | 不能直接充当什么 |
|---|---|---|
| `scripts/ieee_tc_preflight.py:backend_model_check` | 已验证环境、原工件与原 trace 的装配、native token 计量、真实资源域 | 已完成的共同 warm-SLO 实验 |
| 同文件 `qualify_native_source_matrix` | 显式 GPU-ready 建立、加载与引用回执、独占工作区收尾 | 不经改造即可符合共同参考协议的整套 driver |
| 同文件 `collect_native_source_wave` | 原 prompt/token 准备、原生首末 token 事件和完成证据 | 直接使用其中 Prime 选择/准备路径的 wall time 作为纯 warm-vLLM 参考 |
| D88/D89 两模型实测结果与 profile | 系统内部服务/准备时间初始化 | 长度四分位、每组 256 请求、三轮共同参考结果 |
| 基线仓库 `replay_vllm_engine_trace.py` | 既有 vLLM 回放实现可供后续适配参考 | 冻结指标 V1 已合格入口；当前百分位函数仍为插值定义 |

## 需要明确保留的差异

1. `collect_native_source_wave` 要求每 wave 的 adapter ID 不重复，并检查
   `min(runtime_forward_capacity, max_active_loras)`；它为分层 profiling 设计，
   不是共同 reference 的 batch 可行性结论。不能因此直接宣布 batch 8 不可行，
   也不能静默绕过实际 slot/KV 容量。参考 batch 仍按既定协议实际验证后冻结。
2. 该函数进入 `_register_ieee_pending_admission`、所选来源保护和准备路径。
   其外围耗时不是无 Prime 控制逻辑的 warm backend 时间；后续复用原生事件和
   正确的测量边界，不能把 Full 的排队/观测成本塞进参考值以放宽门槛。
3. 现有 matrix 包含多种 tier、显式 setup/eviction 和不同 wave role；已完成
   数据没有证明满足共同参考的长度分组、每组样本数、三轮和 batch 合同。
   不重新标记这些历史数据为 SLO 标定，不重复 D88/D89 初始化实验。
4. 后续选择既有请求索引、按实际执行输入长度分组；不生成新请求或新权重。
   保留真实 GPU-ready 与原生输出证据，固定 prefix-cache，并在每 batch 完成后
   再进入下一 batch，避免继承前一 batch 的排队。
5. SLO 数值继续采用各轮请求均值、三轮等权平均，再使用已冻结的 5×TTFT、
   2×TPOT 倍率。不能用当前开发阈值 5,000 ms 或 D118/D119 的排队均值代替。

本次没有修改上述入口、baseline 仓库、模型配置或运行中源码；没有运行新测量。
后续实现应最小扩展既有入口，仍需实际测量、来源 SHA、资源释放与资格检查，
本审计不证明其已经完成。内部完成事件与客户端响应边界的现有限制继续保留。
