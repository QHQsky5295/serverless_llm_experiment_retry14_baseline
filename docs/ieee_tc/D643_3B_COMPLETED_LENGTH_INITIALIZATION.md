# D643：复用当前 3B 完成长度，准备准入启用后的测量

2026-10-17；属于3B主线的CPU初始化资格，不是Full性能结果。

## 问题和决定

D642在当前vLLM 0.30.0、实际3B池rank≤16、8并发和现行控制执行路径下，
完成368个原生来源用例。但它是source-only：不能把其时间样本改名为启用
admission后的profile。D642中完整exporter拒绝缺失pending-KV信息是正确的，
该拒绝及原始记录保留，不削弱校验。

本轮复用既有`prepare_d298_capacity.py`、`measured_admission_initializer`
和main runtime组装函数。新增D643入口，严格绑定D642原始SHA和D641配置；
不改算法、九公式、输入、权重、现有校验器或旧运行身份。
下一D644复用原46 waves/368 cases，仅启用当前完成长度初始化和对应内容索引，
在既有gated source runner中测量其实际时延与pending-KV路径。

## 离线结果

| 项目 | 结果 |
|---|---:|
| D642原生完成用例 | 368 |
| 用于统计的测量用例（不含预热） | 360 |
| 按原始请求ID去重后完成样本 | 21 |
| 原生prompt长度≤759的样本/平均完成输出 | 8 / 98.375 tokens |
| 原生prompt长度>759的样本/平均完成输出 | 13 / 171.69230769230768 tokens |
| 沿用用例/代表adapter/静态逻辑adapter | 368 / 24 / 500 |
| 既有严格初始化CPU测试 | 13通过 |
| CPU资格峰值内存 | 786112512 B |
| high/max/OOM/OOMkill/swap | 均0 |
| 原投稿保护文件 | 147未变化 |

重复来源/轮次不当作独立长度样本；21个原请求不等于24个代表adapter，也不
代表500种独立训练权重。原生prompt计数包括实际特殊token，内容截断759
不能被误用为原生prompt永远不超过759。两桶都有真实样本，没有缺失值填补。

这两个长度均值恰与早期3B相同，原因是复用相同固定输出案例；本轮仍要求
当前原生完成、prompt SHA、trace、后端和模型身份逐项一致。它不是“新旧
时延相同”的证据，也不是未来请求长度预测准确率证据。

当前官方[采样参数文档](https://docs.vllm.ai/en/v0.30.0/api/vllm/sampling_params/)
分别定义输出上限、忽略EOS和停止条件。本项目固定输出仍必须用原生完成
计数核验，不能仅凭配置上限推定已经完成。此次没有更换生成协议。

## 保持和未完成

- 原有3B 5秒窗口、3个movement容量及D88控制参数保留，并明确标记为历史
  开发控制设置；未宣称根据D642重新标定或得到最终最优工作点。source
  collector不执行autoscaler，保留控制参数不是本轮性能选择。
- 13测试验证现有严格拒绝条件；真实D642数据再次通过strict initializer和
  main child配置核对。CPU验证没有初始化CUDA，没有新的GPU运行。
- D644使用D642相同编译合同、600秒worker启动保护和日志保留；旧入口保持
  原样。启动保护不是SLO，正式共同3B warm/Resident仍待测，不套7B阈值。
- 旧D89时延不换标签，D642 source-only不冒充Full，未生成图表。
- 主线继续：D644准入启用测量→当前3B共同warm/Resident→Full W0/W1/W2及
  全部既定基线/dLoRA。A1–A5、S1–S13、成本/CE、两轮审稿文档和最终图表
  均保留。非本项目k3s不变。

产物在`paper_results/ieee_tc/p2_backend/d643_3b_initializer/`；原始CPU资格在
`results/ieee_tc/p2_backend_qualification/d643_20261017/`。所有新文件独占创建，
不覆盖旧结果、用户manifest或源trace。
