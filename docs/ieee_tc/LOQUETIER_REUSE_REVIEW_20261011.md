# Loquetier：历史复用与共同协议缺口（只读审查）

日期：2026-10-11。状态：历史/源码审查完成；新的 GPU 资格尚未启动。
本记录不替代完整 PrimeLoRA-PLAN.md，不新增性能参数搜索，不宣称复现完成。
审查时 D435 S-LoRA 完整 W0 仍在运行；没有并行测试、构建、模型加载、
远端管理或数据池扫描。历史收据中的 SHA 在下次空闲期核验，不冒称已重新封存。

## 1. 结论与证据范围

- 复用现有源码、环境、编译好的 SMLM、兼容补丁和原工件/trace。
- 历史 7B/128-adapter、3B/256-adapter 成功结果属于 **R2**：支持机器和
  模型兼容性，不满足本轮完整工作量、生成、远程准备和资源计量合同。
- 3B/500-adapter 单卡失败发生在权重准备、尚未执行请求时；不能解释成
  推理吞吐不足，也不能推广为 7B 或全部静态分片部署都不可行。
- 已批准的比较身份是 `Loquetier + static-sharded deployment`，不是其原生
  集群调度功能。只允许必要机器适配；不为性能扫描 1/2/3/4 卡、batch 或路由策略。
- 下一步按安全/正确性证据确定一个可运行的静态分片配置；如有不可行性，
  保留实际失败，不缩小 500-ID 工件池来伪装完整资格，不修改原生计算实现救排名。

## 2. 可复用的明确身份

| 对象 | 已有身份/记录 |
|---|---|
| 官方源码 | `https://github.com/NJUDeepEngine/Loquetier` |
| 本地 commit | `aae33baeeb19777129c1ccbff99a898d4a0e2c63`（本轮读取 git HEAD） |
| 源码路径 | `vendor_new_baselines/Loquetier_main_20260520` |
| 环境 | `/home/qhq/anaconda3/envs/loquetier_20260520` |
| 历史环境记录 | Python 3.9.25、torch 2.1.2+cu121、Transformers 4.45.2、PEFT 0.17.1、SM86 |
| 兼容补丁 | `Loquetier_project/patches/loquetier_local_compat_20260520.patch` |
| 补丁历史 SHA | `135412206787ccc8894fb311eae46899ddfc4085bf07c311f7728bcb5e84f618` |
| 成功收据 | `Loquetier_project/evidence/real_adapt_2026-05-20.json` |
| OOM 收据 | `Loquetier_project/evidence/formal_preflight_2026-05-20.json` |
| 现有入口 | `Loquetier_project/scripts/run_loquetier_trace_gate.py` |

本轮 `git status` 仍有三个历史修改文件：`kernel_src/src/loquetier/utils/kvcache.py`、
`src/loquetier_src/models/mixed_lora.py`、`src/loquetier_src/virtual_model.py`。
未修改或还原它们。历史补丁解决 Python/PEFT API、实际 rank shape 和构建兼容性；
正式使用前须核对实际 diff 与收据，不把 HEAD 单独当成实际执行源码身份。

历史 7B 结果路径：
`results/paper_experiments/15_new_serverless_baselines_remote_v1/gates/loquetier/loquetier_llama2_7b_limit128_filtered256_gate_20260520.json`。
记录 SHA `f9d687494bfff476ba57d9cbd8539921010f0fbaa0af6465a1513a491f03ab43`。
本轮读取到 256/256 完成、507 个输出 token；配置为 arrival_scale=0、
max_batch_size=2、输入上限512、输出上限2、FP16。
这些不是 4,000 请求 fixed-length 结果，不能用离线重算补出缺失的工作量。

历史 3B 最大成功点为 256 adapters/1,024 请求/2,048 token，记录 SHA
`f60586f61fb42f7191f165e28e1a7cda25071003d8dbbc51241adb18e155d2a6`。
500-adapter 单卡日志记录申请2 MiB失败，进程已使用23.54 GiB，GPU余量1.38 MiB。
原日志位于 `results/logs/new_serverless_baselines_remote_v1/loquetier/gates/loquetier_llama32_3b_limit500_filtered16_preflight_20260520.log`，
历史 SHA `a62808271377e3e4d683073892ff99d4dca88c1a4808a27f25f89fde3ab814c2`。

## 3. 为什么旧性能不能直接进入新主表

| 共同合同 | 现有 gate 的实际行为 | 处理 |
|---|---|---|
| 原 arrival/open-loop | `_select_requests` 按已选 adapter 过滤，并平移首请求、乘 arrival_scale；旧点为0 | 复用原 trace 索引，不复用该到达统计 |
| canonical prompt/原目标 | 自有 chat-template/fallback、截断512、优先 body.max_tokens、再 cap2 | 接现有共同输入合同，不重新生成负载 |
| fixed-length greedy | 旧点保留模型 EOS，可提前结束；256请求仅507 token | 显式固定生成配置并核验原生计数 |
| 原生首/末 token | 旧文件只保存首 token 和完成；循环读取原生 IDs，但未把 IDs/末 token/绑定摘要存入结果 | 观察原生事件，不能把完成通知冒充末 token |
| 共同准备起点 | gate 在 backbone、adapter 和输入 GPU tensor 都已准备后才开始相对计时 | 共同通知后准备、业务固定通知后60秒 |
| 真实远程 | 从本地 adapter_root 读取；trace 文件名含 remote 不证明该次发生 HTTP | 复用 D78/D80，保存本次 HTTP/内容收据 |
| 物理 GPU-s | 历史 gate 无完整真实 allocation/release 积分 | 接现有资源所有权与收尾计量 |
| Type-1 分位数 | 旧 `_percentile` 使用 round-index，空集填0 | 新分析用冻结算法；这并不能修复其他执行差异 |

## 4. 原生源码给出的适配边界

本轮读取的是上述本地 pinned checkout；网络查阅官方仓库/论文只用于核对系统
定位，不声称网络抓取已证明每个本地文件字节一致。

1. `virtual_model.py:load_base_model/load_model` 使用字符串 backbone 路径共享
   基座，`is_trainable=False` 为默认。旧入口没有建立训练 optimizer。
   需核验 adapter 配置路径与注册基座的同一性及实际共享关系；不能将重复基座
   物化误称算法开销，也不能未经证据把 OOM 归因于训练状态。
2. `mixed_lora.py:MixedLoraModel.__init__/_prepare_weight` 取所含模型最大 rank，
   按目标模块准备/补齐权重。静态分片的容量审查应按实际 tensor/rank 和该原生
   表示推导，包含基座、原始 LoRA 与准备后表示共存、KV/工作区及 HOST 峰值。
   不能仅用压缩传输字节判断是否装得下，也不改变 padding、精度或 SMLM 求快。
3. `_get_lora_id` 对未知名称返回0。共同接口必须拒绝未知/错误分片 adapter，
   并核对实际 `lora_name_to_id`、InputRequest.lora_id 与 forward 批次绑定，
   否则“HTTP成功/目标长度正确”不能证明使用了正确 adapter。
4. `model_generation.py:ModelGeneration` 使用原生请求队列和 greedy argmax；
   `post_process` 已通过 `.item()` 取得整数 ID。计量应紧邻这个已有边界记录
   首/末事件，避免额外 CUDA synchronize 或以网络 chunk 反推 kernel 时间。
   原生停止条件含 EOS 和 max_length；固定长度可通过 generation 配置和
   每请求长度表达，但空 EOS 配置与目标1/256等边界尚待测试，未声称已通过。
5. `model_workflow.py:run_inference` 是原生 offline workflow，按队列容量推进
   生成和入队；`max_serve_wait` 可选，示例设6秒。不能把这个示例值悄悄当成
   本项目 deadline。新接口使用已冻结的共同超时，保留原生批处理/计算语义，
   只提供必要在线到达与静态映射，不移植 PrimeLoRA 调度或驻留机制。
6. 已有 gate batch=2 是历史兼容工作点，官方示例默认16，两者明确区分。
   不把旧 gate 点称为官方默认或最优。依据最新用户规则不开展性能搜索。

## 5. 恢复主线时的必要工作（尚未执行）

先完成并封存正在运行的 D435，然后在空闲和保护门槛通过后：

1. 核验上述小型源/结果收据、实际环境及已编译内核身份，保留历史失败。
2. 对完整既有500-ID清单做一次静态容量与映射审查；固定物理GPU映射、
   不依赖未来热点的分片、完整覆盖/无未知ID回退，记录部署适配身份。
3. 小幅扩展已有 gate/common HTTP、remote resolver、资源 owner 和 analyzer；
   保留原生 virtual model、MixedLoraModel、ModelGeneration 和 SMLM。
   不另造负载/整池、不重新构建实验框架、不强行合并不同后端。
4. 先做无需 GPU 的输入/映射/计量/失败与超时检查，再按原计划100请求、
   500-ID功能覆盖、原1000兼容、完整4000顺序资格。每键一次性能观测，
   不为了未达95%而搜索 baseline 配置。任何阶段安全不可行先保存原因。
5. 只有完整本次请求、原生 token、实际 adapter、远程来源和物理生命周期
   合同通过后才按共同 D394 SLO评价7B；Loquetier数值正确性须有自身证据，
   不借 S-LoRA 的 witness 代替。未知字段不填造，不要求跨后端逐token相同。
6. 3B仍在7B比较后推进；不因旧3B gate存在就跳过7B主线。完整 A1–A5、
   S1–S13、W1/W2/G2 和其他 baseline 任务均保留。

官方定位依据：[Loquetier 仓库](https://github.com/NJUDeepEngine/Loquetier)
与 [论文](https://arxiv.org/abs/2511.00101)（2026-10-11查阅）：核心为虚拟化模块
和统一微调/推理计算路径。本次 inference-only、静态分片实验不覆盖其全部贡献。
