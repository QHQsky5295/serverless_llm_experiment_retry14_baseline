# D295：同后端受控数值重复（执行前登记）

2026-10-06。下一主线证据是解释 D290/D293 同后端输出差异；不是再次比较
HF/PEFT，也不是新 Full 性能样本。7B 未验收，3B/外部基线保持暂停。

## 假设与边界

在输入、权重、FP16/attention/graph/生成合同不变时，单请求受控重复可稳定，
而并发批组合可能改变接近并列 token 的选择。若单请求仍变化，或内容/映射
不一致，则这一解释不成立或不足。不能把默认不可复现视为免责理由。
[vLLM 0.30 官方复现说明](https://docs.vllm.ai/en/v0.30.0/usage/reproducibility/)
明确默认不保证复现；[batch invariance](https://docs.vllm.ai/en/v0.30.0/features/batch_invariance/)
仍为 beta，本测量不启用它、不改精度、不替换 kernel。

历史依据：D225 受控重复尚未完成；D246 只有一次 native 回放；D247 strict
跨后端 exact 为 19/22，保留不修改；D294 全 200 行保存的绑定记录一致，
但没有完整序列/批成员，不能恢复首分歧。

## 固定设计

- 复用 D293 的 W0 源 trace，SHA
  `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。
- 既有索引 0–21，加 59/74/77/84/91，共 27 条；包含全部六条 D290/D293
  差异请求及既有四种权重内容覆盖。不是新抽样负载，不代表 500 独立训练 adapter。
- 三个预登记 block 各启动一个新 TP=1/GPU0 runtime；不复用上一 block 引擎。
  每个先单请求逐条，再按同顺序四条一波（最后三条），每臂 27 次。
  运行内请求不是独立统计重复；本项不生成性能 CI。
- 配置复用 D293 数值执行条件：vLLM 0.30、FP16、FLASH_ATTN、原 graph 策略、
  prefix cache off、4 slots/4 seqs、1024 context、759/256 输入输出边界。
  使用原始 fixed-length greedy 目标，不缩短长请求。
- 用既有 backend-model-check/dedicated runtime/reference acquire/generate/retire
  路径；控制发起次序，不执行 Prime 的路由/弹性/admission 策略，不声称 Full 资格。
  使用既有 local frozen 工件隔离数值问题；没有 remote 性能点，也不启动远端服务。
- 同时保留原生 token IDs、top-2 加 sampled-token logprobs、精确请求/owner/lease/
  physical slot 执行记录及每波前后 checkpoint-to-slot 内容读回。
  所有观察、同步、logprob 获取开销只属于诊断，不能进入延迟或 GPU-s 排名。
- 三次独立启动不等于所有系统状态严格相同；实际批成员、graph mode 和环境
  回执保留。single/concurrent4 是受控发起方式，必须用真实执行记录验证。

## 判据与下一步

逐 token 相等才标 exact。比较前强制相同 source/prompt/native-prompt/adapter/
target、完整数量和序列 SHA。差异保留首个分歧位置、两侧 token/logprob/margin。
首分歧之前输入 token 前缀相同，之后的轨迹不能当同条件 logit 比较。
不设事后 tolerance，不从 top-2 之外编造概率，不以相似文本宣称正确。

完成观测合同不自动代表 `N_correct`、完整池数值正确或因果解释通过。
单请求重复稳定且仅并发变化，支持“批处理敏感性”的有限证据；仍需结合
实际 slot/执行映射与既有独立 reference，不把相关性写成已定位某个 kernel。
若任一 block 出现测量/资源错误，保存失败，先解释，不盲目重试。

先 1 个 block → 清理 → 校验状态表，再 2/3；不用未检查循环连续跑三个。
内存 72/80 GiB、aux 3/4 GiB、swap、150/100-GiB 磁盘门不变。
原始输出唯一 `results/ieee_tc/p2_backend_qualification/d295_20261006/block_N/`；
相同目录已存在即拒绝执行。旧 147 项保护清单的已知 baseline replay 脚本差异
单列，不把 146/147 写成完整通过；不改变或 stage 该用户文件。

状态：CPU/启动前检查完成；未生成 GPU 结果。首组 177 项通过。扩大回归
539 通过/4 失败；四项均因把要求 PID-handle signaling 的 watchdog 测试误放进
不提供该 API 的模型 Python，而不是新观测路径失败。改用启动器本来使用的
`/usr/bin/python3` 后，完整 75 项 preflight 测试通过；没有模拟补齐 API、跳过
断言或改安全门。其余 468 项（含 basic smoke）全部通过。三测试资源域均
high/max/OOM=0。最终表在各 block 后追加，不覆盖登记。

配置 SHA：`b329e9f6ad6eb517227c0aa974a03ad69d1c9fa25d06f3c76b7885405e62fd3c`。
启动器 SHA：`c5c1f058ed79c2d5c073a5bd07a4299f4df9ee479b692d8f70b51aba92237497`。

### Block1 首次启动未进入模型

原启动器用了可读资源域名称，而既有 guard 要求 UUID-owned auxiliary scope。
guard 正确拒绝；无模型启动、无 GPU 请求、无性能样本。保留原 `block_1/`
日志和提交 `4bba237`。修正启动器使用原有 UUID 资源域规范，并强制显式 attempt
形成新目录 `block_N_attempt_K/`；不放宽 guard，不覆盖已存在目录。该修正不改变
测量配置或请求合同，首轮重启登记为 block1/attempt2。
