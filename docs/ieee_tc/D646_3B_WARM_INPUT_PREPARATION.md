# D646：当前3B共同warm输入核查

2026-10-17。只准备既有请求索引，不是GPU测量、SLO冻结或Full性能结果。
沿D645→共同warm→Resident→Full W0/W1/W2主线，不重跑D644来源测量。

## 输入与选择规则

复用D138完整3B记录中的4,000条输入身份，并用当前原生环境的tokenizer逐条
核对canonical prompt、原生token IDs SHA和输入长度，全部一致。旧延迟不进入
选择或阈值。原trace、500逻辑adapter池和权重不变；未生成新负载、未复制工件。
调用现有`warm_reference_input_index`与`validate_warm_reference_index`，
按Type-1输入长度四分位划分，重复及最大边界合并，不另设计选样逻辑。

| 组 | 原生输入tokens | 原输入数 | 选中请求 | 选中逻辑adapter |
|---|---|---:|---:|---:|
| 0 | ≤506 | 1,023 | 256 | 92 |
| 1 | >506，实际至760 | 2,977 | 256 | 87 |

原四分位为506/760/760；760包含原生特殊token，正文内容上限仍为759。
选择不依赖响应快慢。每组先探测最大完整KV需求与最大不同adapter数的batch，
共4个batch、32请求；通过后才运行194个batch：16条预热和1,536条测量
（2组×256条×3轮）。这三轮是冻结V1已约定的共同标定，不是重复主性能点。

## 不变合同与边界

- 使用D641当前3B配置、同vLLM 0.30.0和CUDA工具链；batch/slot/并发为8，
  prefix cache关闭。已选batch最多8个不同adapter，最大总上下文8,128tokens。
  真实KV block与驻留可行性尚未测出，保持null，不由输入计数推定通过。
- 复用既有`native_warm_reference`收集器与严格原生计时/引用释放检查。
  原生adapter全部GPU-ready后才计时，不经过PrimeLoRA路由/planner/admission。
  warm隔离掉网络，仅用既有本地工件准备；它不是本地模拟remote主实验。
- D647为容量探测，D648为正式warm标定；只扩展现有runner入口，其历史来源
  测量分支参数不变。不用7B阈值替代3B，不按某个系统较慢表现放宽标准。
- 此索引不证明完整池数值区分性；D645已披露2个独立权重SHA等范围保持不变。

## 验证与来源

`tests.test_ieee_tc_warm_reference` 15项CPU测试全部通过；当前tokenizer核验
4,000/4,000。qualification session30918 exit0，无CUDA初始化。管理scope
递归空且已关闭；内存峰值741,072,896B，high/max/OOM/swap均0。
147个旧投稿保护文件及用户manifest SHA不变，未动无关k3s或远端服务。

小型索引及回执：`paper_results/ieee_tc/p2_backend/d646_3b_warm_inputs/`。
索引SHA：`5406f0f13dc489a91ae1f2e199217b5ca471a66acd043de474e301d7d5b135a4`。
原始核查：`results/ieee_tc/p2_backend_qualification/d646_20261017/`。
准备文件记录完整PLAN、V1/V2、旧输入来源和当前配置SHA；旧测量字节不改。

下一步：备份后一次D647 batch8探测，再D648共同warm；随后Resident资源参考、
3B Full及完整baseline/dLoRA比较。A1–A5、S1–S13、审稿文档与最终图表仍待完成。
