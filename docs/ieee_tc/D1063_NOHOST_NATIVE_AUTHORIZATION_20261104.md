# D1063：NoHOST 原生来源授权与私有对象计量

2026-11-04，接续 `D1063_NOHOST_SOURCE_RELEASE_20261104.md`，父提交
`1555ce130dfd8f53c4268e1125115ac3bb0b1c08`。已完整读取最新目标、AGENTS 和
原计划。本页作为增量交接，保留用户修改的 `EXECUTION_STATUS.md`。

**3B A1 性能点仍为 10/11；本轮没有 GPU 性能回放。NoHOST 模式仍未接线启用。**
共享 k3s 不属于本项目，保持不动；HydraServe 已完成隔离适配及 3B W0/W1/W2，
复用已存证据，不重复部署。

## 本轮问题和依据

仅把路由中的 HOST 标为不可复用，并不能阻止原生加载入口使用失活 GPU 的
CPU backing。消融必须在消费入口消除这种独立缓存服务，而不是只更换 tier 标签。
同时，真实 GPU backing、实际借用引用和进行中的构造对象必须受到保护。

核对 D1063 历史实现及既有失败边界，在线复核 vLLM 0.30.0 的
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
与 [model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)。
CPU 注册与 GPU 激活为不同生命周期；上游缓存可以保留 CPU 对象，NoHOST 需要
显式区分独立复用与当前操作必需的临时对象。本轮没有更换原生推理算法、公式、
模型配置或 GPU/NVMe 预算，没有重新尝试零 HOST 预算。

使用 `ablation-planner` 的既有 reviewer 进行同系列 provisional 审查，并使用
vLLM 技能核对原生边界；这不是独立外审或性能验收。审查指出的四个来源入口
问题，以及真实 inventory 对重复逻辑 ID 的限制，均落实到本轮实现/测试。

## 实现边界

1. 新的独立 `hold_host_source`（含 file-fallback）在 NoHOST 下拒绝。
   已有 staging 引用仍由原 lease 幂等释放；GPU 后继必须在后续 controller
   衔接中使用真实 GPU lease，不能把 HOST 引用改名。
2. NoHOST cold demand 必须持有非 HOST 的真实文件 witness。失活 CPU backing
   有相同 name/content 才可替换；实际引用阻塞返回精确目标的
   `nohost_backing_still_referenced`，等待已有 CPU release ACK。无引用时，
   fresh capacity/budget 检查通过后才实际移除旧对象并加载文件，记录
   `nohost_reloaded_inactive_backing`。预算检查不虚构提前释放的字节。
3. file-to-CPU preparation 不再使用旧 registered backing。同步构造及私有
   构造结算保留新旧两个对象直至真实 allocation 检查通过，再退休旧 copy、
   发布新 staging。既有 GPU copy 已赢得加载时仍保留 GPU，不改写其来源。
4. proactive 只允许有效 pending plan、构造回执、内容 SHA 绑定的 staging。
   同一序列化调用内把 `(model object, source ID, content)` 作为私有绑定传给
   激活步骤；无公开 RPC 绕过选项。结果标为 staging，而非独立 HOST 命中。
   同路径错误/缺失 content，以及不同物理路径，在修改缓存前拒绝。
5. private build 的新对象与同 ID 旧 registered copy 不是同一份存储。移除
   旧 backing 不再被单纯 build ID 阻挡；构造对象、线程 join 和峰值 reservation
   的生命周期保持。真实 pin/ref 仍禁止退休，不制造没有 release owner 的等待。
6. 原生 inventory 对正常 registered/staged 仍禁止 ID 重叠。仅对短暂的
   `private_model` allocation 观测使用显式 object-role 遍历：同 ID 两个对象
   都保留，按真实 storage pointer 求并集，区分 registered/staged/private。
   不覆盖字典旧项、不伪造 adapter ID、不把该图用于路由/驱逐容量 credit。
   普通图仍走原 descriptor；private 图显式使用参考遍历，不是异常后 fallback。
   同步 loader 和私有 settlement 的实际 worker closure 均接入此计量。

正常 Full 的缓存/路由/规划/预算策略不变。NoHOST 的必要 CPU staging 和 page
cache 并未假装消失；总主机内存安全限制继续适用。没有修改论文九公式、旧结果、
trace、工件池或远端 D78/D80 交付缓存。

## 有限检查与保留的失败

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261104/native_authorization/`。

| 检查 | 结果 |
|---|---|
| native_tests1 | 109 项通过；随后审查发现未覆盖真实同-ID private inventory，不能凭该轮宣布完成 |
| native_tests2 | 128 项，1 个 wrapper 类名错误：应为 NativeFileHostPreparation |
| native_tests3 | 136 项，2 个 fixture 错误：同一实际 worker 测试的两种 allocator 路径缺 list_adapters 方法 |
| native_tests4 | **136 项通过，0 失败/错误**；源码 SHA 前后一致 |

后两类错误修正的是测试接入，不放宽生产合同。所有尝试保留。最终耗时
9.259 秒（含导入），内存峰值 609,636,352 bytes，high/max/OOM/kill/swap 均零。
CUDA 不可见；CPU 2/3/26/27；memory high/max 3/4 GiB、swap 0。

新增重点包括：真实文件 reload（含等价异路径）、引用释放前拒绝、capacity/
budget 拒绝不删旧 backing、禁止 cold 无 witness 和 HOST 文件、proactive
同-turn/content 绑定、私有构造与旧副本独立生命周期。真实小 CPU 张量检查证明
旧 768 bytes＋新增 256 bytes 的 union 为 1,024 bytes，共享 512 bytes 不重复
计费，registered-exclusive credit 仅 256 bytes。实际 worker 测试验证两个
closure 传递 private 对象，同时保留 registered 对象。

这些是 CPU/native-owner 合同检查；构造和 CUDA 操作部分使用明确 test doubles，
不能证明 GPU 数值正确性、实际性能或完整 NoHOST 执行资格。3B `n_correct`
仍未知，不能从 HTTP/原生 token 数量推造数值正确数。

## 下一步：仅继续未完成的 controller/file 衔接

1. `prepare_gpu_source` 不能无条件接受物理 native CPU 行；NoHOST 失活 copy
   转向真实 NVMe/remote 来源。CPU preparation 返回本轮精确引用冲突时，使用
   已有 CPU movement release 证据，不等整个 GPU plan、不忙等。
2. 每次 NoHOST GPU replacement 取得实际 NVMe/remote fallback；文件零损失
   后继使用真实 GPU lease。默认 Full 路径保持。
3. `execute` 已能读到 actual staged source，但 command 仍使用 frozen row path。
   必须先核对 name/content，再使用实际 staged path，同时保存 frozen/consumed
   两种路径；不改 plan hash。否则等价异路径会持续被本轮原生 guard 拒绝。
4. 完成 `experiment_stack`/runner 模式安装与读回，登记唯一 3B W1 NoHOST，
   复用 D1047 Full；不额外重跑 Full，不按输赢改配置、阈值或删除负面消融。

没有再开性能资格矩阵或改变既定 10/11 计数。LastKnown、ElasticLocality 等
已观察到的反例保留，不能声称 Full 已在所有消融上领先。

## 数据保护、备份和完整余项

按 `github-sync` 做 tested checkpoint，显式文件清单、diff/secrets/SHA 检查，
旧投稿 147 项 seal 和两项用户 dirty 必须不变。提交与远端核验以 RAW
`backup.json` 为准。origin V2 与 faaslora_origin 独立 backup ref 保存；实际
FaaSLoRA V2 分叉尚未集成，不强推，不把备份当作最终集成。

未完成范围完整保留：NoHOST 3B、7B A1 和两个累积点、受控 A2–A5、M2、S1–S13、
CE/价格/Pareto、B1/B2/C1/C2/C3、公式推导/新颖性/两轮审稿文档、IEEE 图表、
同版本 canonical 来源及 FaaSLoRA V2 非强制集成。整体目标保持 active。
