# 同后端输出哈希差异：版本限定的证据边界

2026-10-17。D624 W0 性能回放期间的有界只读核查；不是新性能实验或
数值资格验证，不修改在线配置、生产代码、历史结果、SLO 或九公式。
主线仍为完成 D619/D621 同版本工作点的 W0，再选择有证据的优化方向。

## 1. 已观测与尚未证明

[D623](D623_W2_SLO_SHORTFALL_DIAGNOSIS.md) 已匹配 D615/D621 的全部4000条
请求输入、adapter、到达、prompt hash、目标长度及原生计数；142条完整
输出 token hash 不同。该计数是已有结果，本次没有重新扫描大日志。

输出序列不同不自动等于权重绑定错误；反过来，后端存在非确定性也不自动
证明这142条正确。绑定/内容/计数证据与独立数值验证分开，继续保留
`N_correct=null`、最终排名资格未通过的边界，不更改 D623 原记录。

## 2. 官方版本资料能支持什么

本次直接读取 v0.30.0 标签下的官方文档和源码，不以 latest 中的新实现
替代本机版本。使用 research-lookup 的原始来源核验规则；默认 Parallel
服务未认证，改用可用网页工具，没有安装依赖、登录或上传项目数据。

- [可重复性说明](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/usage/reproducibility.md)：默认模式不保证可重复性；在线服务的建议路径是批次不变性。随机种子的设置与批次调度无关性并非同一保证。
- [批次不变性说明](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/features/batch_invariance.md)：该功能为 beta，通过确定性计算路径等措施实现，可能损失性能。列出的已测模型包含 Llama 3 系列，但不能据此宣称本项目 Llama-2-7B 的全部动态 Multi-LoRA 路径已经验证。
- [LoRA shrink 实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/ops/triton_ops/lora_shrink_op.py)：kernel 配置选择接收 batch、rank、hidden size 等输入。
- [LoRA kernel 工具](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/ops/triton_ops/kernel_utils.py)：shrink 在 `SPLIT_K > 1` 时存在原子累加路径。这只是待核查的数值路径，未证明142条请求实际采用此路径，更未证明它是差异原因。

因此，不能仅因 temperature=0 就要求所有不同批次轨迹产生完全相同序列；
也不能仅设置一个环境变量就声称旧结果已被证明正确或现有 LoRA 路径
具有跨批次确定性。没有进行新的 GPU 算术测试，不作这两类外推。

## 3. 本机有限代码核验

只读取既有独立环境中的三个小文件，没有导入推理后端、遍历模型缓存或
进行全环境哈希扫描。环境根目录为
`/home/qhq/.venvs/primelora_vllm0300_tc_20260925`，包目录为
`lib/python3.12/site-packages/vllm/`。

| 相对文件 | SHA256 |
|---|---|
| `envs.py` | `fbd370b2f56ff798d373e85705c9e044ccae893ef974f2800eec4ef6f2b4fb7f` |
| `lora/ops/triton_ops/lora_shrink_op.py` | `036dfc7f2216de0e9e5da12698fa9c921d076cbc2fda053409ff94a2c4779fa4` |
| `lora/ops/triton_ops/kernel_utils.py` | `0c30484ae5c097800589aab000b96c226c9ec275ab6c4a462838f76569b62e68` |

本机 `envs.py` 中 `VLLM_BATCH_INVARIANT` 的缺省值为 false；这不等于已经
读取了每个历史 worker 的实际环境。本机 shrink 配置同样接收 batch，
kernel_utils 的对应分支也有原子累加。这里只确认代码路径存在，不推断
特定请求的 kernel 配置或错误概率。最初尝试的 python3.11 路径不存在，
随后核实实际为 python3.12；没有在不存在的路径上作版本判断。

## 4. 对后续工作的约束

- D624 继续原配置，运行中不切换确定性、批处理或后端环境，不新增推理。
- 完整运行结束、释放资源后，优先复用已有数值资格与绑定证据。若需补检，
  区分相同受控输入/批次下的数值验证、批次变化影响及 adapter 错配；不能
  用普通运行成功、文本可读或原生 token 数正确来填补独立数值证据。
- 不修改失败判据或数值容差追随结果，不把新诊断配置的性能拼入原配置。
- 这一核查没有关闭142条差异，也没有形成性能优化候选。W2 的94.95%
  联合SLO短缺继续保留，3B、消融、机制、敏感性及最终交付范围不缩减。

计划 SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
本小型文档待当前性能运行安全结束后，随正常检查点做差异/敏感信息检查
与备份；不在性能测量期间推送或执行压缩维护。
