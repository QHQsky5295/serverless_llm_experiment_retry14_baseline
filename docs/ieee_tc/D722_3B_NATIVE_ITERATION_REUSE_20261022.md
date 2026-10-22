# D722：复用3B原生迭代记录，区分调度形状与控制开销

2026-10-22。接续D721；没有GPU回放、生产改动、配置选择或SLO修订。
复用D684的4份封存原生日志、312160个span与1000条请求窗口，沿用D517／D691
的离线流程和现有严格analyzer，恢复89915次step。45项观察器／分析测试通过。
这些是带观察开销的历史前缀，早于D703／D719，不是当前Full性能结果。

## 为什么先复用

D721只证明完整回放的短输出违约与同副本prefill包络存在重叠；该包络含排队、
控制和执行，不能据此宣布GPU prefill是主要原因。D684其实已记录每次schedule
的请求数、总token数及多token请求数，无需重复推理即可进一步区分迭代形状。
本次确认原D684窗口采用D648冻结共同阈值，SHA
`d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`，
不是在线5000ms。所有窗口和原生日志SHA、覆盖与分解恒等式重新核对。

## 原始分组结果

| 原生分配形状 | step数 | step平均wall / 线程CPU（ms） | step P95（ms） | scheduler平均（ms） |
|---|---:|---:|---:|---:|
| 每个请求分配1 token | 87527 | 12.013 / 4.656 | 20.682 | 0.156 |
| 至少一个请求分配多token | 974 | 64.547 / 64.214 | 77.465 | 0.305 |
| 分配0 token | 941 | 3.166 / 0.902 | 10.067 | 0.076 |
| 本step没有schedule | 473 | 0.244 / 0.242 | 0.508 | 0 |
| schedule或step抛错 | 0 | N/A | N/A | N/A |

多token组的总分配token数中位数760、P95为764、最大3040。474次只有1个
被调度请求，平均step63.278ms；这说明大跨度不只发生在多请求混合batch，
但不能由分配形状推断GPU执行阶段。线程CPU接近wall不等于纯Python计算，
驱动忙等和本地库也可以消耗该线程CPU。

下表是每条decode窗口与互斥主机区间的交叠均值，单位ms。不是反事实匹配，
不能将组间差解释为删去某组件后的收益。

| 输出2–64 tokens的分组 | 请求数 | decode窗口 | 控制client | 多token分配step | 单token分配step | 其他合计 |
|---|---:|---:|---:|---:|---:|---:|
| TPOT达标 | 353 | 374.751 | 69.515 | 23.332 | 269.602 | 12.302 |
| TPOT违约 | 61 | 682.919 | 387.257 | 95.670 | 180.821 | 19.171 |

“其他”包含input非client、zero-token／no-schedule step及未被主区间覆盖的
时间。HOST清点是client内的嵌套子集：上述两组分别49.410／282.116ms，
不再加入总量。2–32与2–64分层均完整保留，但它们有交集，不能当独立重复。

## 判断与有界下一步

1. 本次确实观察到多token分配step更长，但原生scheduler自身仅约0.305ms；
   不能把64.547ms解释为调度算法计算，也不支持直接照抄512／2048预算。
2. vLLM异步batch队列可能在本step回收前一batch，当前schedule的分配形状
   不能标定该跨度全部GPU工作。只减token预算还可能增加输入准备和调用次数。
3. 控制client交叠仍大于多token step交叠。只做prefill切分没有证据可以独自
   达到共同95%；D719自身TTFT边际也尚未达到3800/4000。
4. 下一项只允许一次当前3B的**原有前100请求组件诊断**，复用既有
   `v2_components_v1`观察入口，区分execute_model、输入／attention准备、
   LoRA切换、graph主机调用和输出处理。不新增同步或GPU事件，不改Full。
   100条用于有界定位，不用于选最佳配置或外推4000；避免重现D519的64MiB
   截断。若仍无法定位则保留缺口，不重复同配置，也不恢复已拒绝D529等候选。

本轮核对[D519结果](D519_STEP_DIAGNOSTIC_RESULT.md)、
[D520元数据依赖](D520_NATIVE_METADATA_DEPENDENCY.md)及D529拒绝记录；
已经失败的7B元数据候选不是本次重新采用的优化。当前3B完整点仍为D719：
joint91.60%、U13508.300779 GPU-s，独立数值正确性未知，未满足G1。

版本固定的外部依据：
[vLLM 0.30优化说明](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/#chunked-prefill)
说明token预算存在TTFT／ITL取舍，不保证通用数值最优；
[EngineCore源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的step_with_batch_queue明确包含异步发射及旧batch结果回收。本机测量不能
仅由这些设计说明推导。

## 仪器变化与安全

| 文件 | 本轮变化 |
|---|---|
| raw `d722_20261022/audit_iterations.py` | 复用D517/D691流程，改为D684封存输入及D648阈值核查 |
| raw `d722_20261022/finalize.py` | 验证本轮空资源域、保存小型汇总和源码快照 |
| 本文、EXECUTION_STATUS及curated目录 | 保存来源与结论 |
| 生产、配置、原生观察器、通用analyzer | 零修改 |

分析实际CPU为2/3/26/27，内存峰值503541760B，high/max/OOM/swap均0。
144个旧投稿文件及用户manifest SHA前后不变；新原始step／请求表保留在
gitignored目录。没有远端服务操作、共享k3s操作、工件／trace复制或新图。
全部后续3B场景／baseline／dLoRA、消融、敏感性、CE和审稿交付范围保留。
