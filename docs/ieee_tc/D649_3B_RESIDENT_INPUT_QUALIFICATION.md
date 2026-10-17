# D649：3B Resident 输入与模型身份准备

2026-10-18。主线是取得当前3B共同Resident资源参考，再评估PrimeLoRA Full，
不是重新开展HydraServe平台适配。本节点没有GPU性能运行，不产生资源预算或排名。

## 为什么需要这一步

D648已完成当前3B共同warm测量并冻结TTFT/TPOT数值；不能用7B的SLO或
Resident资源代替3B。现有原生vLLM四卡入口与真实GPU持有计量可复用，但
共享分析器的目录、远端health索引和SLO模型身份此前固定为7B。

本次只把分析模型身份显式化：`--model-key 3b`选择3B路径、远端记录与阈值，
默认仍为7B。backbone解析后的实际路径必须与模型身份一致，冻结SLO的模型
也必须匹配；错误模型、人口集合或调度合同明确拒绝。原生事件、生成数量、
物理GPU积分、成功/失败分母、SLO计算公式不变。三个原生系统的receipt路径
接受显式model_root，既有默认路径保留。

## 配置复用及边界

新增baseline配置`configs/ieee_tc/vllm_3b_resident_native_compiler_v1.json`，
复用D641/D138的模型相关参数，而不是把7B并发直接复制给3B：

| 项目 | 3B配置 |
|---|---|
| 后端与精度 | 既有vLLM 0.30.0，FP16，TP=1 |
| 常驻部署 | 四张物理GPU，各一个原生实例，原round-robin |
| 序列/批token/上下文 | 8 / 4096 / 1024 |
| GPU/CPU LoRA槽与最大rank | 8 / 32 / 16 |
| 显存比例 | 0.72 |
| prefix / chunked prefill / eager | false / true / false |
| 编译工具链 | 既有共同CUDA13合同SHA 328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c |

未修改原生服务、请求分配或计量源码；未加入PrimeLoRA的readiness routing、
handoff、层级planner或admission。未安装环境、构建内核、搜索baseline参数。
这不是宣称该配置为vLLM全局最优，GPU部署仍需按原公共资格顺序验证。

官方0.30.0文档支持逐请求LoRARequest绑定及显式槽/rank配置，并建议rank上限
匹配工件实际最大rank；本文的16来自既有池审计，不来自性能筛选。
[vLLM LoRA文档](https://docs.vllm.ai/en/v0.30.0/features/lora/)。

## 已核验的输入（只读复用）

| 视图 | 请求数 | 实际adapter ID数 | 目标输出token总数 |
|---|---:|---:|---:|
| 原序列前100 | 100 | 29 | 17,369 |
| 原序列前1000 | 1000 | 60 | 122,510 |
| 完整W0 | 4000 | 132 | 458,224 |
| 500-ID功能视图 | 500 | 500 | 仅作全池功能资格，不作W0性能 |

source SHA `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`；
完整W0 view SHA `0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37`；
pool index SHA `bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275`。
没有新生成trace、权重或第二套工件池。500逻辑ID不写成500独立权重；W0实际
出现132个ID，须与功能覆盖分开报告。保留source seed42与development身份。

D648冻结SLO SHA `d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`：
原生prompt长度≤506时TTFT1466.050801ms/TPOT29.921872ms；较长组
2539.648668ms/36.215515ms。这里只核验身份，不重新标定或调整阈值。

## CPU资格与失败保留

现有分析/证据测试73项、原生vLLM服务测试16项，共89项通过；包括新增模型/
路径/阈值错配拒绝与3B配置对照。全部为CPU合成或输入检查，不是原生GPU
正确性及性能证据。

第一次driver从baseline仓库运行测试，被其同名scripts包遮挡了主仓库工具链
模块；第二次纠正scripts导入后，又暴露tests常规包/命名空间包冲突。两次
日志和空进程资源释放均保留，未运行GPU、未改服务代码或测试断言。
随后复用D534已验证的显式文件测试加载方式，执行全部16项服务测试，并与
73项分析回归共同通过。不是绕过失败检查或修改被测实现来通过。

受限CPU域3/4GiB、swap=0、CPU2,3,26,27；实际峰值及关闭证据见
`paper_results/ieee_tc/vllm/d649_3b_resident_inputs/`。147个投稿保护文件与三处
用户修改校验不变。无远端服务操作、无Kubernetes配置或权限修改、无旧结果覆盖。

## 下一主线与未完成范围

1. 备份本节点两个仓库的显式代码/配置及小型证据；不提交原始大日志或凭据。
2. 沿既有D557 native-http-service和受限launch/remote/cleanup路径，使用本次
   3B输入先100请求smoke，再500-ID功能覆盖、1000兼容和完整4000 W0/W1/W2。
   只补缺少的当前合同资格；若有完全匹配历史证据则先审计复用。
3. 合格完整Resident运行同时作为vLLM基线与同场景U_ref，不为两个用途重复
   运行。只有完整正确及资源计量合格后才能冻结0.75U_ref预算。
4. 随后当前3B PrimeLoRA Full W0/W1/W2、其他3B基线与dLoRA限定比较，再
   同版本主表/Full消融一致性、A1–A5、S1–S13、成本/审稿文档和最终图表。

不回到已完成的7B/HydraServe基础资格或HF逐token支线。原k3s属于其他项目，
保持不动；后续HydraServe复用已有隔离适配及明确局限，不宣称未触发机制已验证。
完整目标保持ACTIVE；本节点不是3B SLO合格、G1/G2最优或论文实验全部完成。
