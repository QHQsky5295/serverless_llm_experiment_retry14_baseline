# D753：3B 基线复用与 ServerlessLLM 当前路径接续

2026-10-23。D752 的 3B W0 联合时延达成3831/4000=95.775%已封存；本轮
返回比较主线，不再调同一W0配置。没有新的性能实验、生产代码修改或图表。
评价继续使用冻结V1与单轮V2、D648共同阈值；独立数值证明未知不自动升级。

## 已有数据与处理

| 系统与来源 | 已有内容 | 本次裁决 |
|---|---|---|
| ServerlessLLM D74 | 原3B前1000请求；997响应、3失败；本地工件、诊断eager、无完整物理lease账本 | R2：保留诊断及原生环境/模型/负载，不进入新主表、不重跑原版轮询配对 |
| vLLM D653 W0 | 4000条原生合同、联合99.975%、U=16119.239770 GPU-s | 复用完整开发观测及D657参考，不为baseline另跑一次 |
| vLLM D654 W1 | 4000条原生合同、联合100%、U=9006.378087 GPU-s | 同上；不是独立数值正确性证明 |
| vLLM D656 W2 | 4000条原生合同、联合100%、U=16110.751827 GPU-s | 同上；高频轮换的497 IDs不等于497组独立权重 |

D74原始replay、HTTP配置、部署配置SHA与封存汇总相等，1000/997/3不改写。
未重新扫描全部历史或重算其失败前缀。D657三个参考引用的原始完整审计SHA
分别重验通过；因此不因PrimeLoRA代码更新而重跑未变化的vLLM基线。
3B其余系统、限定dLoRA和跨场景PrimeLoRA/G2仍须继续完成。

## 当前ServerlessLLM复用边界

复用D414/D555的原生vLLM0.10.2、store快速加载、RR、已授权ready-before-wait
轮询修复、request-identity scheduler队列修复、pre-ready入口与真实物理GPU
池账本。服务与measurement helper、四个实际原生文件及共享HTTP/clock源码
SHA完全一致，复用既有114项相关CPU检查，不重复资格或新增对手调参搜索。
3B输入沿D649的既有trace和500-ID索引；100/1000/4000及功能覆盖视图完全
一致，7项本次输入核对/负例通过。该核对不是GPU全池功能验证。

原3B checkpoint `v43-sllm-native-smoke-llama32-3b`直接复用，不转换或复制。
静态tensor index均为FP16，索引末边界等于既有文件长度；本次没有以此替代
模型运行或算术正确性证据。3B原min1/max4/target8/keep10和backend seq4/
token1024/GPUfraction0.72/prefix-on设置保留，LoRA槽4/CPU槽4/rank64保留。

唯一相对D74旧诊断模式的公开执行选择是`enforce_eager=false`，沿已验证的
D409/D410图执行修正。旧helper为资格强制eager，不是官方默认；本次不扫描
3B性能参数、不扩大并发、调整阈值或用更慢模式人为压低对手。公开后端默认
false及vLLM混合执行定义已重新联网核查：
[固定ServerlessLLM源码](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/backends/vllm_backend.py)、
[vLLM 0.10.2参数](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html#enforce-eager)。
实际3B捕获/执行和启动开销必须在新运行中观察，不能从7B结果推定。

## 下一项：D754受影响路径100请求

新目录`results/ieee_tc/serverless_qualification/d754_20261023`，只复用现有
runner/源视图/loader安装恢复/远端管理/分析入口，改为明确3B模型与100人口。
100请求来自原W0前缀，29 IDs、17369目标tokens；不造负载。当前只登记，
是否实际启动、结束和通过以回执为准。

- 本轮私有下载目录`/tmp/tc-d754-3br`；D78/D80发布缓存复用，3B端口18080。
- 60秒部署通知、原生固定长度、1800秒资格保护、原CPU/72/80GiB内存和
  物理持卡计量不变。准备与捕获计入，真实获取/竞争/物化不事后扣除。
- 新运行前重新检查150GiB/102GiB余量；辅助3/4GiB、无swap，单重型任务。
- 不改其他项目k3s；HydraServe复用既有rootless隔离适配，不重开权限问题。
- 先检查真实模型/adapter/token/时间/资源/远端绑定；低SLO不是重复优化理由。
  异常只处理有证据的必要适配，保留失败，不自动回退eager或增加超时。
- 完成后真实释放、恢复本轮loader、停止自有远端、分析、封存；再补该后端
  尚缺的500-ID功能/1000兼容与完整场景。现有合格证据不重复。

D753 CPU域峰值225914880B，high/max/OOM/swap均0，已精确关闭；145旧投稿
保护项及三处用户修改SHA不变。`experiment-audit`共享依赖缺失，仅完成上述
项目来源/合同检查，不宣称独立审查通过。使用run-experiment复用既有框架。

全部未完成范围保留：3B其余基线/限定dLoRA、PrimeLoRA W1/W2/G2及必要资源
优化、7B同版本主表/Full、A1–A5、S1–S13、离线CE/Pareto、两轮审稿设计文档
与最终学术图表。W0时延档位达标不是完整G1/G2最优；单轮不声称统计显著。
