# D437：Loquetier 完整池静态容量与原生生成规则检查

日期：2026-10-12。阶段：7B 剩余基线资格；不是性能实验，不进入 M1/M2 排名。
本记录接续 `LOQUETIER_REUSE_REVIEW_20261011.md`，不替代完整计划。
仅作必要兼容、正确性、计量和安全准备，不开展基线性能参数搜索。

## 1. 本次得到什么证据

完整既有 500-ID 工件池可按实际静态字节确定四个不相交分片，每片125个。
但每卡已计入的基座、原始/准备后 LoRA 和 batch=2 满上下文 KV 约22.5 GiB，
尚未包含 CUDA context、工作区、临时张量等，**不能据此宣布 GPU 容量可行**。
原生生成的四项 CPU 规则检查通过，亦不等于 GPU 数值或完整请求合同通过。

没有加载 GPU 模型、发起 HTTP、管理远端、重跑已封存性能点或新建数据池。
用户再次批准的只读 gzip 交付缓存仍复用 D78/D80；没有重复发布、请求中打包
或创建第二套解压工件池。真实读取、传输、竞争和本地物化的计量规则不变。

## 2. 复用和实际身份

原生源码路径：`vendor_new_baselines/Loquetier_main_20260520`。
HEAD：`aae33baeeb19777129c1ccbff99a898d4a0e2c63`。
三个历史修改文件保持不变；实际默认 `git diff` SHA：
`de476d06c318ba2be7f65dd9dba9bfb30a2f2c7e14e8bd6e35f30206ab041a39`。
既有兼容补丁文件 SHA：
`135412206787ccc8894fb311eae46899ddfc4085bf07c311f7728bcb5e84f618`。
补丁使用不同 unified context，不能要求其 SHA 与默认 diff 相同。

本次核对现有 `loquetier_20260520` 环境：Python3.9.25、torch2.1.2+cu121、
Transformers4.45.2、PEFT0.17.1、safetensors0.7.0；已有 SMLM 构建记录为SM86。
没有安装、重新编译、升级后端或修改驱动。

| 对象 | 本次核验 SHA256 |
|---|---|
| 原生 `models/mixed_lora.py` | `93be379685693ca16dd57b4400bea9e446ba8ed4108f97f5ba7b2c311d6e4b1e` |
| 原生 `model_generation.py` | `31fb751c0a0b04608b427e84c765735883f37128af4e12379dd8860b135a6cb2` |
| 已编译 `_kernels.cpython-39-x86_64-linux-gnu.so` | `100efcf29d3f8a53d8b2fe31b877d1589634024c742c8fff6ff30110f5a57ec9` |
| 旧7B/128-ID/256请求结果 | `f9d687494bfff476ba57d9cbd8539921010f0fbaa0af6465a1513a491f03ab43` |
| 旧3B/256-ID/1024请求结果 | `f60586f61fb42f7191f165e28e1a7cda25071003d8dbbc51241adb18e155d2a6` |
| 旧3B/500-ID准备期 OOM 日志 | `a62808271377e3e4d683073892ff99d4dca88c1a4808a27f25f89fde3ab814c2` |

旧7B点输出总计507tokens、输入cap512、输出cap2、arrival_scale=0、batch=2，
仍为 R2 兼容证据，不能重算成共同 fixed-output W0。
旧3B OOM 不证明所有7B静态分片不可行；亦不能删去该失败。

## 3. 静态容量方法和结果

只扩展现有 `Loquetier_project/scripts/run_loquetier_trace_gate.py`：
新增显式 `--static-capacity-only`；不导入 torch、不读 trace/subset、不读全权重。
验证500个ID、配置SHA、权重文件大小、safetensors header的形状/类型/完整区间，
以及 backbone index/header 和基座路径同一性。
这里的 header SHA **不是** payload SHA；运行时仍须核验实际物化内容。

输入清单为主仓库
`paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json`，SHA
`e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c`。
原工件目录为
`/home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix`。
500个逻辑ID不等于500个独立训练权重；既有内容多样性审计保持有效。

实际rank：350个rank8、150个rank16；同一128-target-module布局。
按静态 footprint 降序、ID稳定排序，再分给累计字节最少分片；并列按数量和
分片index决胜。没有使用未来访问次数、热点、到达或性能结果。
每片原生最大rank16。异构目标模块布局、未知ID和重复ID不会被静默接受。

计量包括：

- FP16基座：13,476,835,328 B/卡。
- PEFT原始LoRA：默认autocast产生FP32；与另备的FP16权重分别计入。
- 原生 `_prepare_weight` 按该分片最大rank补齐的FP16表示。
- 原生A/B指针表。
- KV：`2 × layers × kv_heads × head_dim × 2 bytes × batch × ceil(context/16) × 16`。

batch=2沿用已有兼容点，不称官方默认或最优；context=1024覆盖共同输入合同。
四卡是已批准静态分片部署的容量资格候选，不是搜索后的性能最优配置。

| 分片 | ID数 | 原始LoRA / B | 准备后LoRA / B | 指针 / B | 满上下文KV / B | 已计入合计 / B | 合计 / GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 125 | 5,469,372,416 | 4,194,304,000 | 256,000 | 1,073,741,824 | 24,214,509,568 | 22.55152 |
| 1 | 125 | 5,469,372,416 | 4,194,304,000 | 256,000 | 1,073,741,824 | 24,214,509,568 | 22.55152 |
| 2 | 125 | 5,435,817,984 | 4,194,304,000 | 256,000 | 1,073,741,824 | 24,180,955,136 | 22.52027 |
| 3 | 125 | 5,435,817,984 | 4,194,304,000 | 256,000 | 1,073,741,824 | 24,180,955,136 | 22.52027 |

明确未计：CUDA context、allocator碎片、logits/activation、临时padding/cast、
原生workspace、HOST峰值。实际dtype和共享storage仍需运行观察；
`runtime_qualified=false`、`safe_to_launch=false` 保留。

结果：`paper_results/ieee_tc/loquetier/d437_7b_static_capacity.json`，SHA
`049441391b67e500328546c3c8e1001fd1ce197f0c1a93e150fc26e8bdfe4a77`。
该结果所用脚本SHA：
`b0f679614432c135197867007c3fa8db62bceca4b671d88cd809a66486c1268f`。
约392KiB元数据，没有复制权重或完整trace。

## 4. CPU 验证及其边界

`tests/test_loquetier_static_capacity.py`：5项通过，0.021秒。
覆盖实际rank与配置rank区别、A/B一致性、未知表示拒绝、确定性完整静态映射、
重复/异构布局拒绝、header区间和截断检查。合成小fixture不是性能数据。

`tests/test_loquetier_native_generation.py`：在原生现有环境中显式开启
`TC_TEST_LOQUETIER_NATIVE=1`，4项通过，22.855秒，无skip。
直接执行未修改的原生 `ModelGeneration.post_process` 与映射逻辑，
使用CPU tensor/mock KV，不执行模型算术：

1. 禁用EOS后，目标1/2/256均恰好终止；即使原生整数token为EOS也不提前停。
2. batch内不同请求完成/排序后，mark和LoRA ID保留，KV恰好释放一次。
3. `do_sample=false, num_beams=1, eos_token_id=None` 对应GREEDY_SEARCH。
4. 实证未知名称在原生 `_get_lora_id` 返回0；后续公共接口必须主动拒绝未知/
   错误分片ID，并核验实际输入与forward绑定，不能把该回退当正确执行。

这些检查不证明GPU内核数值、native token时间戳、HTTP请求合同或显存可行。
不能以CPU模拟 token替代正式原生模型输出。

三次检查在GPU隐藏、CPU2,3,26,27、memory.high3GiB/max4GiB/swap0域内完成：

| scope 后缀 | InvocationID | peak / B | high/max/OOM/swap |
|---|---|---:|---|
| d4370000000000000000000000000001 | `cd1d535689a742258ebee84fda1e2a36` | 12,460,032 | 0 |
| d4370000000000000000000000000002 | `b968f2e3e59544e6bc895e967d808fba` | 32,608,256 | 0 |
| d4370000000000000000000000000003 | `6976b153528e46a7ad974d3be8b77b6d` | 536,424,448 | 0 |

三组均已按准确身份和真实空进程子树结束；没有存活的D437检查任务。

提交前在同一原生环境重跑9项CPU回归，全部通过（2.219秒，无skip）；
这不是性能重复。aux0004的InvocationID为
`370ccc61689744b78416cfdb596f0eca`，peak248,307,712B；
147项旧投稿资产校验通过，aux0005的InvocationID为
`478687e8a52a4d35b28e1f4c57d7a589`，peak12,562,432B。
两组high/max/OOM/swap均0，均按身份和真实空进程子树结束。
保护回执位于主仓库
`paper_results/ieee_tc/baseline_audit/d437_loquetier/protected_verification.json`，SHA
`1ff66513bbe7dc141d9bd14e43a7ce4f111cee38aeb215db2db8eebe141fb657`。
三个受保护用户文件和S-LoRA/Loquetier历史原生补丁保持原SHA。
收尾检查四卡仅各15MiB图形占用、无tmux实验；MemAvailable约105GiB，
磁盘可用162,920,607,744B（约151.73GiB）。这不是下一重型实验的准入许可，
下一次仍须完整新检查，尤其新物化数据峰值和HOST内存。

## 5. 下一步仍须验证

按完整计划，复用既有common HTTP/remote resolver/资源所有权/分析入口，
只补静态映射、原生计时和共同输入/失败/取消接口；保持 virtual model、
MixedLoraModel、ModelGeneration、SMLM和原生批处理，不移植PrimeLoRA机制。

- 原生 `.item()` 已有整数输出边界附近观察首/末token，不额外同步GPU，
  不把旧gate的step结束或HTTP chunk时间冒称原生生成时刻。
- 共同通知后开始准备，业务固定通知后60秒；实际GPU申请至释放全部计量。
- 完整500-ID远端内容、真实共享基座/LoRA表示和受保护实际容量需验证；
  500个逻辑ID不利用重复/零权重折叠，不缩池。
- 先完成无GPU接口测试及受保护实际容量/100请求资格，再依序做500-ID覆盖、
  原1000兼容、单轮完整4000；失败保留，必要安全处理不变成性能搜索。
- 独立核验Loquetier adapter数值证据，不借S-LoRA证据代替；未测字段仍未知。

上述尚未完成；目前不能将Loquetier加入合格W0主表。
然后继续HydraServe原生资格、7B比较/必要PrimeLoRA优化，再推进3B；
W1/W2/G2、A1–A5、S1–S13及审稿文档均未因此取消。

官方定位与适配边界参考：[Loquetier 官方仓库](https://github.com/NJUDeepEngine/Loquetier)、
[论文](https://arxiv.org/abs/2511.00101)。本次只评估其原生推理加静态分片部署，
不冒称覆盖统一微调/推理全部贡献或原生集群调度。
