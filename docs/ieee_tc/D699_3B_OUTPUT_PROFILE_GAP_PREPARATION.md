# D699：3B 声明输出长度画像缺类准备（CPU-only）

日期：2026-10-19

## 结论

D698 的 30 个缺失项已经被转换为一个可审计的、仅引用既有输入的 native
source-matrix 波次计划。D699 目前只是准备结果，不是新的服务测量，也不是
新的 3B 性能结果：`launchable=false`、`gpu_runs=0`、`n_correct=null`。在
动态 native collector 完成并通过独立 coverage/export 审计以前，不能把它当作
3B profile、SLO 标定或 PrimeLoRA 排名证据。

机器可读产物：

- `paper_results/ieee_tc/p2_backend/d699_3b_output_profile_gaps/source_spec.json`
- `paper_results/ieee_tc/p2_backend/d699_3b_output_profile_gaps/target_manifest.json`
- `paper_results/ieee_tc/p2_backend/d699_3b_output_profile_gaps/prepare.json`

准备脚本为 `scripts/ieee_tc/prepare_d699_3b_output_profile_gaps.py`。它只做静态
解析和 fail-closed 校验，没有连接远端、启动 GPU、下载文件、构造请求或修改
既有 D644/D645/D698 文件。

## 缺类的严格含义

D698 先以 D644 已完成的 360 条 native service interval 计算声明输出长度的
Type-1 四分位边界 `[89, 152, 179, 256]`，再与既有非输出维度交叉。缺类是
具体的：

`(tier, representation, prompt_bin, output_limit_bin, rank_bin, footprint_bin, admitted_bin)`。

因此这里不能用邻近长度、均值、默认值或“同一 adapter 的其他 source”填补。
只有实际 collector 产生的 `admitted_after_accept`、source representation 和
原生 token interval 才能关闭对应缺类。

## D699 波次构造

- 数据只来自既有 1,000-request trace prefix、既有 500-adapter frozen pool 和
  已发布内容索引；没有新 trace、新权重或新 adapter。
- 对每个 D698 缺类，按已有请求的真实 `expected_output_tokens` 和实际
  `adapter_config.json` rank 选择一个目标请求。目标 request ID、adapter ID、
  rank、声明输出和 source SHA 都被写入 `target_manifest.json`。
- admission 边界采用已经冻结的 `[1,2,4,8]`。目标请求放在自然并发 wave 的
  最后位置，wave 大小分别为 1、2 或 4；此前的 0、1 或 3 个请求是同一既有
  trace 前缀中的真实背景请求。没有 sleep、barrier、延迟注入或持有已完成请求。
- 每个 wave 内 adapter 不重复；不同 wave 可复用同一既有 adapter。collector
  必须记录所有背景请求，不能只保留目标请求。
- 另保留一个不超过 8 lane 的 GPU warmup。warmup 不计入 representative
  coverage，且不能被解释为性能样本。
- 5 个 source 各有 6 个 measurement waves，共 30 个缺类 wave；总计划记录
  61 个 request references（包括 warmup）。静态解析确认最大 wave 为 6，低于
  3B 当前 runtime 的 8-lane 上限。

## 静态门禁结果

`source_profile_inputs` 使用与 D644/D645 同一个 resolver，验证了：

- trace SHA：`4ea5d026…eb26018d`；
- 500-adapter content index SHA：`bd1826c5…a38f275`；
- pool 为既有 `llama32_3b_a500_v1_modelscope`，没有越界路径；
- 31 个 wave 均引用已存在 request/adapter，wave 内没有重复 adapter；
- `bins.declared_output_tokens=[89,152,179,256]`，rank 和 admission 边界未改变；
- 5 个 source（GPU、native HOST、file HOST、NVMe、remote）均出现 6 个 wave；
- resolver 解析到 31 个 wave、6 个 distinct adapter identity；
- preparation 阶段没有远端连接、没有 GPU run、没有 profile 默认填充。

注意：静态 resolver 只能证明“输入与波次合法”，不能证明目标请求在运行中
确实落入目标 source 或 admission bin。source 改变、请求顺序变化、native
terminal 不完整、adapter/token 绑定不一致时，动态 collector 必须失败关闭。

## 后续动态运行与验收

后续应以新的 D699 execution key 运行既有 `native_source_matrix` collector，
使用 D699 `source_spec.json` 和 D641 3B native 配置。启动前仍执行推理机
150 GiB 磁盘起点、内存增长量、swap、GPU idle、远端服务健康和只读发布缓存
门禁；运行期间不扫描、压缩、打包、重启远端服务或改配置。

动态运行需要：

1. 31/31 waves 完成，所有 planned request 有 native terminal、固定输出合同、
   adapter binding、完整 source admission 和 release receipt；
2. `remote_pack_performed=false`，真实请求触发的 remote read/transfer/materialize
   保留在原生时间线；
3. 用实际 `class_features` 重新计算覆盖，30 个 D698 缺类全部出现；未出现的
   类不能插值或以 target manifest 代替；
4. 通过 `export_d298_profiles.py` 的来源、释放、时间恒等式和完整性检查，生成
   新的 development-only profile；
5. 将 D644/D645 的历史 development samples 与 D699 新 samples 按 source SHA
   合并，再重新执行 coverage。直到合并 coverage 完整，3B source profile 才
   能进入后续 runtime qualification；这仍不等于 3B W0/G1/G2 已完成。

## 解释边界

D699 只解决 D698 明确识别的画像覆盖缺口。它不改变 PrimeLoRA 的九个论文
公式、G1/G2、共同 SLO、75% 资源预算、远端计量口径或单轮封存规则；也不证明
3B 当前候选优于 D682/D693。D695/D697 的失败和 D698 的 coverage-only 身份
继续保留，不能因 D699 静态准备成功而改写。
