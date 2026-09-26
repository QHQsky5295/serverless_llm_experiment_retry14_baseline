# 既有 7B/3B 工件内容核查：2026-09-26

## 结论与范围

按计划 §4.3、§6.7、§12 检查当前本地 frozen 工件的全部 500 个逻辑 ID。
这是原始工件的只读 CPU 检查，不生成、训练、修复或替换任何权重，不是性能实验。
清单、文件内容、大小和修改身份在扫描期间均保持不变。

| 当前工件池 | 逻辑 ID | 不同权重文件 SHA | A/B 全零 ID | 含非零 A/B 的 ID |
|---|---:|---:|---:|---:|
| Llama-3.2-3B | 500 | 2 | 500 | 0 |
| Llama-2-7B | 500 | 4 | 498 | 2 |

3B：rank 8 的 350 份与 rank 16 的 150 份分别具有相同权重 SHA；全部为 FP16，
没有 NaN/Inf。7B：348 份 rank 8 和 150 份 rank 16 全零；`finance_lora` 与
`medical_lora` 的 A、B 含非零值且 SHA 不同。非零不等于已经验证领域训练质量，
也不凭 A、B 分别非零就宣称其乘积在每层都非零。

检查的是当前本地文件。历史运行或远端副本是否具有完全相同内容，要用各自
保存的 SHA 进一步核对，不能把当前扫描直接回填成历史测量。

## 数学意义与论文证据边界

标准 LoRA 的附加项为缩放后的 `BAx`。任一操作数为零即可证明该层更新为零。
这也解释了 PEFT 默认初始化为何可保持原模型行为；官方说明是 A 随机初始化、
B 为零，而不是本次观察到的 A、B 都零。
[PEFT 官方 LoRA 初始化说明](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#initialization)。

- 不能将这些 ID 表述为 500 个独立训练或领域专用模型。
- 零权重仍可能走实际 LoRA 注册、内存、传输和计算路径；因此发现零权重并不
  自动使过去所有计时失效。但必须证明路径确实执行，并限定为结构性系统负载。
- 全零适配器的输出不能区分“加载了正确 ID”“加载了另一个零适配器”及
  “根本没有应用 LoRA”。原生 ID/slot/加载回执证明控制路径，不替代这一
  数值正确性证据。不能用同一份被测输出自己充当正确性的独立答案。
- fixed-output 协议减少了生成长度差异，但不能消除工件内容/压缩代表性问题。
- GPU-s、SLO 和缓存收益的主张仍需完整公平实测；不因本次发现先验断言 Prime
  或某个 baseline 胜负，更不为获得非零输出修改工件。

## 刚完成的 3B 原生输出对照

原 run tag：`llama32_3b_nativeadapterref2_attempt1`。
同一个既有 `req_00003` prompt，先后使用 `finance_lora` 和 `code_lora_0015`。
后者按原始前 100 请求中首次出现的不同权重 SHA 选择，没有按输出筛选。

| 检查 | 结果 |
|---|---|
| 原生后端 | stock vLLM 0.30.0 AsyncLLM |
| prompt / 原生 input IDs | 两次相同 |
| 实际/目标 token | 各 217/217 |
| 两次 output IDs SHA | 相同：`2c2300cd16afc6c4e1295a8695da2899ffc2e9dace654f7389d97f4cfb25148c` |
| 权重内容 | rank/file SHA 不同，但两者 A、B 全零 |
| 结论 | API、长度、清理通过；语义可区分的负对照未建立 |
| 外部资源证据 | 64 次采样，服务峰值 4,926,136,320 bytes，high/max/OOM 均零 |
| 收尾 | 原生 scheduler/cache 空，GPU context 已释放、服务资源域已移除 |

原始文件中的 `pass=true` 仅覆盖其实际断言：生成完成、token 数和清理。
没有不同输出断言，不可升级为适配器应用正确性通过。原始 `input_mode`
沿用了 concurrent-pairs 标签，实际 requests 和代码路径明确是两个顺序对照；
原始结果不改写，curated summary 已显式指出两项限制。

该发现解释本次“切换不同 SHA 仍同输出”的不可区分性，**不解释**此前 3B
取消/批次改变输出的具体数值根因。后者已由 stock AsyncLLM 独立复现，
不能归因于 Prime 的引用/驱逐，也不能未经证明断言某个 kernel 有错。

## 字节与来源：不能把填充当 LoRA 参数

| 内容（500 个逻辑目录求和） | 3B bytes | 7B bytes |
|---|---:|---:|
| safetensors 权重文件 | 5,978,564,400 | 10,922,202,000 |
| `adapter_data.bin` | 8,932,962,246 | 0 |
| 全部目录文件 | 19,482,573,296 | 12,985,984,450 |

3B 填充文件逐字节检查均为零。这些数是**逻辑文件字节**，不是独占物理磁盘、
HOST tensor、GPU pool 或网络线上字节。共享 inode、page cache、压缩均需分开。
现有 artifact 服务使用 tar gzip (`remote_artifact_node/server.py:123`)；全零权重
及填充高度可压缩。因此不能把本地按目录大小注入的等待视为真实远端等价，
也不能未经新证据假定远程路径对所有方法影响相同。按原计划分别记线上字节、
打包/传输/解包，保留实际部署的路径语义。

来源审计事实：

1. 3B `.publicmix_generation_manifest.json` 记录 500 个 ID、rank/名义大小，
   没有足以证明已训练的来源信息；当前目录无 `faaslora_artifact_spec.json`。
2. 7B manifest 的 `pool_type=publicmix_v2`，但 `public_count=0`、
   `generated_fill_count=500`。目录名 publicmix 不能充当真实公开工件证明。
3. 当前 `generate_adapter_synthetic` 明确使用 `torch.zeros` 写 A/B，并按名义大小
   填充零字节；函数注释中的 randomly initialized 与实际代码不一致。
   当前内容与这类路径相容，但未恢复确切生成命令，故不声称已经证明历史来源。
4. 既有 sanitized 脚本可将 NaN/Inf 替换为零，但其路径不是当前被测路径；
   没有证据就不能把本次全零归因于 sanitization。
5. 对 public_candidates、archive、remote 和两个 remote 备份目录的 2,521 份
   adapter config 做限定范围检索，未发现 base 指向 Llama-3.2-3B 的候选。
   这不是整台服务器所有文件都没有合适工件的证明，也没有下载替代品。

## 主线处置与待决策

已完成：本次原生对照制表、两池全量内容审计及可复核 CSV/JSON。

可以继续：7B 现有非零工件的独立数值正确性验证、Full 公式与生命周期接入、
Serverless 历史审计和其他不依赖新增权重的工作。两个原池保持原样。

暂不准入：把 3B 当前全零池视为“适配器语义已正确验证”，或把两池写成
500 个独立训练模型。本次资格不能放行完整 TC 主性能结论。

需要用户确认的范围变更：计划禁止新增权重，而已有查找范围内没有非零 3B。
建议仅允许补充少量明确来源、兼容的已训练非零 3B 工件，用于独立正确性对照，
不替换/复制原 500 池、不重建 trace、不删除旧测量；是否升级代表性实验范围
另行决定。若不允许，则保留结构负载限定及该语义证据缺口，不能强行通过。

不要重复运行相同的零权重负对照；它在数学上不可区分，不是多跑 seed 能解决。
M1/M2、核心消融和敏感性仍未开始；本检查不是这些实验的替代品。

## 复核与产物

沿用 `scripts/ieee_tc_preflight.py artifact-audit --path <existing-pool>
--expected-adapters 500 --output <new-json>`；只读输入、拒绝覆盖输出。
按 SHA 复用张量统计，所有逻辑文件均独立哈希，扫描前后检查文件身份。
七项小型测试覆盖零/非零操作数、NaN、缺失、重复 ID、扫描期间变化及零修改。
测试中的小数组只是临时数学夹具，不是模型工件或实验负载。

- `paper_results/ieee_tc/p2_backend/20260926_artifact_tensor_audit.{csv,json}`
- `paper_results/ieee_tc/p2_backend/20260926_3b_nativeadapterref2.{csv,json}`
- 原始审计：`results/ieee_tc/p2_backend_qualification/model_20260926/artifact_tensor_audit_3b7b_attempt1.json`
- 原始 SHA：`224b9337ecaba2590ac888e9d0e3b9eb41a17074ce77b952e98e935259adf60f`

CPU 扫描使用独立 1/2 GiB high/max、swap=0、CPU 3/27，峰值 1,074,528,256 bytes。
产生 43,008 次 high 事件（文件缓存回收/节流）；max/OOM/OOM-kill 均零。
因此不从扫描耗时推断性能。结束后进程为空，仅停止该已空的专属 scope。
没有并行模型运行、整机 page-cache 清理、权重修复或历史结果覆盖。

## D62：把既有内容审计接入完整 HTTP / profile 输入

执行前核查：当前Full使用逐文件`artifact_content_v1`，现存生成清单只有来源/
逻辑大小，D18以来已有消费接口，但还没有两模型的真实完整内容索引。D26的
GPU/HOST串行前缀也不覆盖当前Full所需的文件HOST、NVMe、Remote及并发
服务类别，不能重贴配置标签后当作冻结profile。本轮先补齐实际输入身份。

复用既有`ieee_tc_preflight.py`增加`artifact-index`操作：读取原审计、两池
现有全部文件，输出可由原HTTP client直接消费的小型JSON。相同且扫描中不变
的inode只读一次；不同inode不凭相同名字/大小/旧SHA猜测相等。权重、PEFT
配置和填充SHA必须与原审计相同；其他文件获得当前身份，不反推历史一致性。
最初扫描前后检查目录和文件身份，链接/设备/池外ID/途中改变均拒绝。不存在输出
时才写入，不复制、解压、训练、重新生成工件或负载，不重新做tensor统计。

顺序为3B索引→校验/状态表→7B索引→校验/状态表；单个受限CPU任务，无GPU
或真实远程服务，结果不作为性能测量。每个索引保存原审计、生成清单、脚本、
计划SHA，全部逐文件大小/SHA、完整目录内容类数及inode去重读取量。
这服务于正式输入校验和待测类别确定，不改变当前保守`exact_content_v1`分区，
不把两种权重SHA等同于两种远程文件树，也不据此冻结准备时间。

依据原HTTP消费实现、D18/D22/D36历史及重新核查的
[Python3.12 tarfile文档](https://docs.python.org/3.12/library/tarfile.html)、
[vLLM0.30 loader源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)。
文件传输完整性与正确adapter的数值应用是两项独立检查；本索引不替代后者。

### 首次索引检查与有证据的边界修正

| 步骤 | 结果 | 判定 |
|---|---|---|
| `index3b`首次扫描 | 遇到辅助文件symlink时拒绝，没有生成索引 | 本地文件树假设过严；当时尚未核对远端文件类型，不是模型失败 |
| 本地只读枚举 | 500目录各有5个指向基座模型的辅助文件链接 | 权重/PEFT配置/填充本体不是这些链接 |
| 174实际服务源码核对 | 明确跳过池外或dangling链接；SHA `a365072244512f4880432d7f4198cf3e45897ff19063a2b25eb141c8ca4e2a02` | 只证明源码行为，尚不能推断远端目录的文件类型 |
| 原server真实loopback响应测试 | 两个临时目录的archive普通文件和索引大小/SHA逐项一致 | 仅证明payload选择合同，不是174实际下载资格 |

第二次采用`regular_files_skip_external_symlinks_v1`，明确记录每条被省略链接、
可读本地字节以及来源目录总量。仅索引本地服务会发送的普通文件；受管权重/
PEFT配置/填充不可通过省略链接逃过旧审计核对。内部链接展开尚不合格则明确
拒绝，不引入静默兼容。所有原文件、服务及其打包行为保持不变；首次日志保留。
该修正不是修改被测系统以提高结果，而是避免计量合同把未传输内容算作payload。

### 远端目录实查：区分本地打包与实际远端文件集合

上述服务源码只能说明如何处理链接，不能证明远端目录本身仍然有链接。
随后对174两池各500目录做只读文件类型/大小枚举，发现3B的五种辅助文件
已经全部物化为普通文件，没有symlink；7B也全部为普通文件。**因此不能将
本地跳过链接的索引直接用作174的3B下载清单。**不改写任何本地/远端工件。

| 对象 | 文件条目 | 逻辑payload bytes | 当前证据范围 |
|---|---:|---:|---|
| 3B本地打包索引 | 1,500 | 14,911,799,296 | 500目录、24个完整内容类；仅本地打包诊断 |
| 174的3B普通文件 | 4,000 | 19,482,573,296 | 只读类型/大小枚举；尚未逐文件远端SHA验证 |
| 174的7B普通文件 | 5,000 | 12,985,984,450 | 同上；本地链接情况以随后的lstat扫描为准 |

本地3B省略的2,500条链接共4,570,774,000逻辑字节。首次成功索引保存在
`paper_results/ieee_tc/inputs/20260927_3b_content_index.json`，SHA
`ba57a73918851669c28af0e383df9cacb38d2217fbbbaa9d8dd16662c00a6d16`，
**只作local-pack诊断，不放行真实remote实验**。该扫描scope进程已空并停止，
峰值1,074,524,160B，high23,683、max/OOM/OOM-kill均0；不能将耗时当传输性能。

下一次索引采用显式`regular_files_materialized_support_v1`：只允许解析既有
`models/LLM-Research--Llama-3.2-3B-Instruct`内的辅助目标，按远端普通文件的
相对路径记录当前本地SHA，既不复制也不改写链接。未授权目标/内部展开拒绝，
共享目标从首次观察至扫描结束必须不变。实际下载仍须逐项匹配；文件数/大小
一致不是远端SHA相同的证明。此清单是待验证的输入合同，不是已完成远程资格。
沿用CPU3/27、1/2GiB high/max、swap0；3B通过并制表后再顺序扫描7B。

3B显式物化索引已完成：500ID、4,000文件、19,482,573,296逻辑字节、24个
完整文件树类；实际只哈希1,505个不同inode/14,920,940,844B，不复制共享辅助
文件。原HTTP client成功冻结索引，并逐一校验500个PEFT配置和routing identity。
这仍是本地可信输入的检查，`remote_content_verified=false`。

| 3B物化索引收尾 | 观测值 |
|---|---|
| 索引文件 | `paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json` |
| 文件SHA | `bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275` |
| HTTP规范化合同SHA | `ac8e9b36c328376a9e1ecec6a4d928dd684f4d94e3fa4d2e08f531b166da145e` |
| 索引器SHA | `99ccb362cbb5640c2dfb27b5746f96ab08bfb578f4eb9d018ecfb1da105be700` |
| 资源 | peak1,074,520,064B；high22,769；max/OOM/OOM-kill0 |
| 收尾 | 实际cgroup.procs空、TasksCurrent0，专属scope停止 |

本地打包与远端物化索引的24类均由实际完整文件SHA计算；不是把24类当作
独立训练模型，也不把上述读取耗时当作加载profile。68项最终安全/索引/回放
检查通过；其中包括共享辅助目标在不同adapter间改变时必须拒绝的测试。

### 7B第一轮扫描与同一物化规则

此前按可读文件枚举的5,000条不能证明本地没有链接。完整lstat扫描实际发现
`finance_lora/config.json`和`generation_config.json`是指向既有7B基座目录的
链接，共797B；其他4,998条是普通文件。因此默认skip索引只覆盖12,985,983,653B，
不能冒充174的完整文件集合。500个PEFT/routing身份校验通过只说明该索引内部
一致，不消除这项远端文件集合差异。

| 7B默认skip检查 | 观测值 |
|---|---|
| 输出（仅local-pack诊断） | `paper_results/ieee_tc/inputs/20260927_7b_remote_content_index.json` |
| SHA | `f53c8e4e71e189ff064aa76cb4d133900d62cee43f71b820cd28d27c90fc8bc3` |
| 完整文件树类 | 6；不是独立训练权重数 |
| 资源 | peak1,074,266,112B；high18,811；max/OOM/OOM-kill0 |
| 收尾 | 实际进程空、scope停止；GPU全部15MiB/0% |

沿用已通过测试的同一索引器，显式允许既有
`/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf`辅助目标，
另存7B物化索引，不覆盖首轮索引。重新读取只是验证现有字节，不生成新工件。
最终两模型的下载候选索引仍须实际HTTP校验才能获得remote资格。

对上述7B两文件另做一次严格SSH只读抽查：远端均为普通文件，609/188B，
各自SHA与本地允许的基座目标相同：
`9242e7db1bc2a17873e66084c3b1c6ed10883076e156b338fd6a7775748e2e3c`、
`11e70f5fd0a1a47346b5a941ff0049d226b353a1647735d9ec171a2b1521e881`。
只读取这797B，不启动服务；此局部一致性不能替代两池完整下载资格。

### D62最终输入交付（不是性能或完整remote资格）

| 预期远端文件集合索引 | 逻辑ID | 文件数 | 逻辑字节 | 完整内容类 |
|---|---:|---:|---:|---:|
| 3B显式物化 | 500 | 4,000 | 19,482,573,296 | 24 |
| 7B显式物化 | 500 | 5,000 | 12,985,984,450 | 6 |

7B物化索引`20260927_7b_materialized_content_index.json`的文件SHA为
`e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c`，
HTTP规范化合同SHA为
`684c7ab113b6a51b694066753b340fce4eb0b26f565e1f9b7ebbd97d3b8ea050`。
scope峰值1,074,266,112B，high17,056，max/OOM/OOM-kill0；实际进程空后停止。
最终两索引均由原HTTP consumer读取，逐项校验全部500个PEFT/routing身份，
无遗漏链接，逻辑总量与旧tensor审计一致。3B/7B的权重/config/padding哈希
全部维持原审计；辅助文件的当前身份新增记录。使用说明和全部文件SHA见
`paper_results/ieee_tc/inputs/README.md`，避免误用早先local-pack诊断索引。

68项安全/索引/回放检查、288项基础回归通过；全部本轮scope停止，无模型
或远端服务运行。147个保护条目及计划SHA不变。每份JSON明确保留
`remote_content_verified=false`、`serving_qualified=false`。内容类用于确定
待测覆盖，不是已经测得的准备时间；不合并旧profile或解除Full守卫。

下一步回到代表性实测服务/准备时间与Full资格。不要重复本次索引或旧内存/
输出前缀微测；独立数值正确性仍需补齐。174磁盘和3B非零正确性工件的范围
选择仍待用户确认；没有据此开展baseline/M1/M2/消融或宣称系统优越性。
