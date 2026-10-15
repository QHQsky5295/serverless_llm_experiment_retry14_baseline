# D554：冻结配置下的 7B 快速 adapter 轮换

2026-10-15。实验主线：已封存 W1 六系统观测及 PrimeLoRA D551 候选后，
检验同一模型配置在 W2 的表现。不是另一轮 W1、基线调参或新的算法候选。

## 科学问题与不变条件

W2 保留原 4,000 条请求的 arrival、prompt 和 458,224 目标输出 token，只将
热点轮换间隔从 500 请求改成 100。沿用 D553 已验证的逐请求 rank 映射，
窗口宽度 48、步长 12、相邻热点 75% 重叠；实际访问 497 个逻辑 adapter ID。
不能称为 497 组独立训练权重，也不把更换 adapter 后的旧输出当作数值参考。

与 D551 相比，配置文件仅变结果目录、私有 NVMe/HOST 路径；其余解析后完全
相同。vLLM 0.30.0、FP16、TP=1、14 slots/seqs、rank16、batch token1024、
初始1/最多4副本、observed_arrivals_v1、confirmed routing、handoff、hierarchy、
admission、profile、控制阈值和九个公式全部保留。没有按 W2 更改参数。

原生服务代码自 D551 起仅有 D553 已验证的共同输入读取变化；本轮以 AST
核对其余代码完全一致。本轮生产改动只在现有 shell 入口转发 map 路径/SHA，
缺索引、误用 profile 或未启用受控外置回放时拒绝启动，安全包络不变。

源 trace SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`；
map SHA `48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`；
完整 W2 view SHA `5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7`。
静态工件池、源 trace 不重建、不复制。在线策略不接收未来轮换计划。

复用 D551 的启动、远端、释放和分析流程，以及 D553 的共同输入检查。
再次查阅 [vLLM 官方 benchmark 文档](https://docs.vllm.ai/en/latest/cli/bench/serve/)，
保留本项目 trace 时间和 adapter 映射，不切换随机/round-robin 赋值或速率采样。

## 资格与失败保留

D553 的 386 项输入/计量检查复用；源码 SHA 不变项逐一验证。测试文件唯一
增加的类之外，旧测试 AST 不变。本轮新增 4 项 shell 边界测试已通过，覆盖
W2 双参数转发、缺字段拒绝、非受控/错场景拒绝、W0/W1/coverage 兼容。
没有重复原生机制资格、基础编译或已有性能运行。

首次测试替身误以为名为 systemd-run 的 shell 函数可拦截 `exec systemd-run`。
实际创建了四个受限辅助资源域；真实入口在模型启动前拒绝不存在的测试路径
或不完整 coverage 参数。未启动模型。失败日志、原测试源码和 driver 完整
保留；先确认每个资源域身份、递归无进程，再关闭并核对路径消失。不能由
`--collect` 或命令退出假定已释放。首次保存 helper 的这一错误假设也保留。
随后只将测试替身改为拦截 shell exec 边界，生产代码不变，四项检查通过。
所有检查/保存资源域均未发生 high/max/OOM/swap 事件。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d554_20261015`。
使用独立 `7b_initial_control_full_w2` 结果、`7b_nvme_initial_control_w2` 缓存和
`/dev/shm/tc-d554-7b-initialw2`，不覆盖 D551。推理前必须再通过资源准入并
备份；复用 D78/D80 只读交付对象，真实传输计入、请求中不打包。运行期间
不维护远端、不清理缓存、不执行编译/测试/扫描/备份。

## 评价与后续

按冻结 V1 与单轮 V2，报告完整生成与 adapter 绑定、联合 SLO、原生 TTFT/TPOT、
实际 GPU-s、dispatch tier、远端次数/字节以及 40 个轮换区间的描述性结果。
SLO 使用 D394，不使用运行器显示的开发 5000ms。独立数值正确性仍待证据，
不由 HTTP 成功或 token 数量自动宣布。W2 预算参考尚未完成，不虚构 G2 资格。

本文件当前是运行前登记，不包含性能结果。首次有效结果永久封存，不补运行间
CI、不重复同键争取胜出。后续 W2 基线、共同预算、同配置 W0、3B、A1–A5、
S1–S13、两轮审稿设计文档及最终图表均未删减。
