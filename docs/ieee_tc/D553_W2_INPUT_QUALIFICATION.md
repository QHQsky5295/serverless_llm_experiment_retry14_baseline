# D553：W2 快速轮换的共同输入合同

2026-10-15。主线位置：已封存 7B W1 六系统结果及 D551 PrimeLoRA 新版本观测后，
准备同一冻结模型配置的 W2。此项是输入/计量资格，不是新性能结果或算法优化。
全部 A1–A5、S1–S13、3B、G2 与最终交付仍保留。

## 原始负载事实和受控变化

完整 PLAN §4.2 / 冻结指标 V1 要求 W2 保持 W0 arrival、prompt 和 token 工作量，
只将 adapter 轮换间隔从 500 请求改为 100。现有源 trace SHA：
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。

原配置 `llama2_7b_auto500_formal4000_s8` 与 `AzureTraceReplayer.replay` 的
历史实现实际使用 48-ID 热点窗口，每次前移 12 个位置，即相邻窗口 75% 重叠。
这不是每次完全替换一个不相交热点集合。源文件的 500-ID 顺序和逐请求实际
局部 rank 均可恢复；4,000/4,000 原始 rank 落在 0–47。

对于从零开始的请求序号 i、源 adapter 在原选择清单中的位置 p_i：

\[
k_i=(p_i-12\lfloor i/500\rfloor)\bmod500,\qquad
p'_i=(k_i+12\lfloor i/100\rfloor)\bmod500.
\]

保留每请求已实现的局部 rank，不重新抽样 RNG，不重建到达或 prompt，保留
48/12 的窗口宽度与步长，只改变轮换频率。有限样本下不声称每段精确服从
理论 Zipf；它继承源回放的实际 rank 序列。

离线映射核验：4,000 请求、458,224 目标输出 token；实际 unique adapter
从 132 变成 497，3,900 条请求改变 adapter。保留既有 500 个逻辑工件及真实
内容关系；497 个出现 ID 不代表 497 组独立训练权重。

索引 `paper_results/ieee_tc/inputs/20261015_7b_w2_adapter_map_v1.json`：
SHA `48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`，
1,033,078 字节。只新增索引，不复制完整 trace/工件池。映射使用已保留源文件的
canonical 路径；基线此前的另一 trace 路径内容 SHA 相同，W2 统一指向 canonical
路径，不改其字节。

## 共同路径与防止错误对比

- `FrozenReplayPlan` 在加载时核对 map SHA、完整 source/view SHA、500-ID
  顺序、pool SHA、全部原始行 SHA，并重新验证每条 rank 与目标公式。
- W2 只允许完整 4,000 请求、rate_scale=1；缺 map、短回放、覆盖视图或其他
  profile 混用直接拒绝。W0/W1 的原身份序列化不增加无关字段。
- 原生服务入口、外置 publisher、HTTP 基线入口和离线 population 校验共用
  该视图。请求 `adapter_id` 与 body 中重复的 LoRA 名称同步改变；domain 更新
  为目标静态元数据；原 domain、adapter 与行 SHA 保存在派生注释中。
- prompt 内容不按新 adapter domain 改写。因此这是受控的 churn 压力视图，
  不是新语义任务质量数据。请求生成正确性/adapter 绑定仍逐条核验。
- 映射只供实验输入与核验使用，不向在线 hotness/planner 提供未来热点；
  九公式、调度、缓存预算、控制阈值、后端与 baseline 原生策略均不改变。
- W2 的请求 prompt/native input IDs/target 必须与 W0 逐条相同；PrimeLoRA
  与 HTTP 基线的 adapter、派生行 SHA、到达计划必须相同。不能用旧 W0/W1
  输出参考直接充当不同 adapter 下的数值正确性证明。

参考过现有 D531 W1、D451 coverage 接入及其 Git 历史；官方 vLLM 文档区分
保留 trace 时间戳与重新按速率采样，也提供不同 LoRA assignment 规则。
本项目保留自身冻结映射，不切换其 random/round-robin 默认分配。
[vLLM benchmark 官方文档](https://docs.vllm.ai/en/latest/cli/bench/serve/)。

## 验证与资源状态

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/main_comparison/d553_20261015`。
`audit.json` 已通过，原源文件及 147 个投稿保护文件未变；其实际资源域已递归
确认无人、关闭且路径消失，峰值 183,365,632 字节，high/max/OOM/swap 均为零。

共同输入资格已通过：现有 tokenizer 的 4,000 请求 prompt SHA、native input-ID
SHA、target 与 W0 逐条一致；PrimeLoRA 实际服务读取函数与 HTTP/population
绑定同一 W2 映射。W2 view SHA：
`5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7`。
`qualification_final.json` SHA：
`1e5f62dac7b1cb9e94f9b37a8b56e232a58069ac47513e36441751921a5fdeb1`。

386 项受影响 CPU 检查完成，无跳过：380 项在此前合格的 CPU wrapper 环境
通过；4 项 watchdog 检查使用实际具备 PID-handle signaling 的系统 Python；
2 项离线解析检查使用已有 ijson 分析环境。后 6 项只重查环境不兼容项，不重新
跑 380 项，也未安装依赖、改断言或改生产代码。与 D537 的已有分环境资格方式一致。
完整主检查资源域峰值 2,898,542,592 字节；六项补查峰值 75,853,824 字节；
均 high/max/OOM/swap 为零，实际递归空、关闭、路径消失。

失败记录不删除：第一次 driver 缺 multiprocessing main guard，子进程重复
写独占回执，导致 FileExistsError 和后续测试无法结束。第二次 guard 修复后，
推理环境的子进程导入启动 humming NVRTC/launcher 辅助进程，发生 memory.high
回收压力；峰值 3,589,148,672 字节、high 48,866、max/OOM/swap 零，未当作通过。
两次只停止身份确认的测试资源域；20 秒 stop 等待超时后又核验实际 inactive/dead
及路径消失，未把超时当作释放。第二次保存 helper 的过宽文件名替换错误及部分
归档也保留，改为明确文件名后完成保存。第三次主检查的原始 2 failures/8 errors
来自上述六个环境相关测试方法，保留原日志，并由六项独立补查闭合。

这些是测试运行器/环境问题，不是 GPU 性能失败或调参轮次；不修改共同 3/4 GiB
辅助资源包络去强行通过。未启动 GPU 推理、远端服务或新的基线性能运行；
native execution、独立数值正确性和 W2 性能仍待实际回放，原有 k3s 保持不动。

下一步先推送两仓库的受测修改与小型证据，再为同配置 W2 建立独立运行键、
私有缓存和资源准入。既有 shell wrapper 还需显式转发 W2 map 参数，注册新
config/case 并检查原生调用路径，不能因离线资格通过直接冒称可启动全矩阵。
不会重复已经封存的 W0/W1 性能点。
