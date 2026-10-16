# D564：Loquetier 并发变化的同前缀数值与混合容量诊断

2026-10-16。D562失败/部分结果和D563请求身份修复已封存并推送，不覆盖或重跑旧键。
本次落实Loquetier异常低性能的专项原因核查，保持IEEE九公式、共同协议及其他系统不变。

## 问题和可证伪假设

D561的batch2完整输出与D445一致；D562的8条同时prefill中7条一致，
req_00015/code_lora_0015在零基index131首次由11029变为1024，后续生成分叉。
绑定、输入与数量检查通过不等于独立数值正确。批形状改变引发局部argmax变化是
一个假设，不预先排除原生SMLM/执行错误，也不凭官方文档宣布差异无害。

历史：D437/438静态容量、D445/W0、D544/W1、D560所有权回收、D561工作区、
D562失败和D563诊断ID修复。当前native固定commit aae33baeeb19777129c1ccbff99a898d4a0e2c63，
在线engine、kernel与旧batch2配置不改。D563 helper只修复不同场景复用诊断ID，
不会解释数值变化。本轮原始实现及资料再次联网核查：

- [固定原生生成实现](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/src/loquetier_src/model_generation.py)：原生greedy对next_token_scores执行argmax，随后原生post_process。
- [PyTorch2.1数值说明](https://docs.pytorch.org/docs/2.1/notes/numerical_accuracy.html)：批量计算可能不逐bit相同；这不是本次差异的原因证明。

## 最小诊断，不是性能重复

仍只扩展现有run_loquetier_trace_gate.py。显式绑定D561容量证据和D562差异证据SHA，
核对source-view、静态分片、容量模型及完整输入相同；只接受一个已留存差异，
从其原始输出自动确定观察位置，不按新结果挑选请求或token。

一次加载最大125-ID分片并验证原始对象回收/准备后权重和指针不变后：

1. 同一个generator上用目标请求的两份完整输入作batch2数值参照。
2. 使用D562原有8条不同adapter请求同时prefill，全部生成到256token。
3. 使用同8条输入作4decode＋4prefill混合执行，完整drain。
4. 实际分配8条完整1024上下文KV，验证容量边界。

请求均760输入、256输出；不缩短、不强制前缀、不改变adapter、不屏蔽EOS之外的
既定生成策略。每场景执行身份不同，source_request_id保持原身份。所有场景使用
相同原生模型/SMLM和D561推导的b=8，未作并发扫描或生产配置修改。

## 只观察原生预测分数

在现有generation_strategy接口临时附加诊断观察：先调用原函数取得原始决策，
再读取目标请求已登记的输入、实际生成前缀、adapter/index、分段与原生score，
最终返回原函数产生的同一个token tensor；退出或异常必恢复原函数。

只在既有分歧位置采样，batch2两份、batch8和mixed各一份，预期四份向量。
保存完整little-endian FP32 score字节、SHA、实际前缀、top8、top2间距、最大值
并列数、旧/新候选分数、实际batch顺序和分段。文件只允许新建，不覆盖。
如果前缀已经变化，保留prefix_matches_retained=false；不能把不同上下文的
score差异作为同前缀解释。实际token仍由原生greedy决定，不反馈历史输出。

同前缀且边界候选排序变化可解释本次token分叉的直接决策原因；仅有小差值
不能独自证明所有原生算术正确。需联合绑定/分段/权重证据，公开未覆盖的独立
高精度参考与全500数值身份限制。差异大、错绑定、非有限或缺证据则保持未合格。
诊断中同步/读回影响时序，因此绝不据此报告吞吐、SLO或正式GPU-s收益。

## 资格与保护

36项CPU检查通过、无跳过：19项容量/所有权＋17项原生测量。新增检查覆盖
差异证据/输入绑定、CLI缺少SHA拒绝、原生返回tensor不变、scores不变、并列argmax、
向量字节/SHA、路径不覆盖、前缀错误和NaN拒绝、异常恢复。实际CPU资源域已关闭，
峰值284,471,296B，high/max/OOM/kill/swap均0。D560已合格source-view直接复用。

RAW：/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d564_20261016。
复用D562 prepare/launch/tmux/closure脚本；新键、新目录，不新增模型/工件池/trace。
GPU运行前重新通过150GiB磁盘、102GiB主机余量及真实72/80GiB服务限制检查。
一次重型任务，无远端服务操作；实际释放后才能分析/发布，失败按实际结果保留。
此处明确为本地不可变池容量诊断，不冒充真实远端性能实验。

当前GPU诊断尚未启动、结果未知。证据完成后再决定在线表示生命周期与容量配置，
然后全500资格、新版本W0/W1/W2；不把小诊断成功等同于完整性能复现或G1/G2完成。
全部7B/3B、消融、敏感性、两轮审稿文档与后期图表仍保留。

## 实际结果：容量执行完成，数值解释有边界

一次原生诊断正常完成，服务/watchdog RC0，实际GPU释放、服务资源域消失及
launcher递归空成员关闭均验证。262次外置采样，服务峰值6,966,071,296B，
主机最小可用105,148,612,608B，high/max/OOM/kill/swap均0。无远端操作。

原始FP32表示实际回收5,469,372,416B，基座/准备后内容与指针不变；
batch2、all-prefill8、mixed4/4均完整生成256token/请求，分别257/257/259采样。
混合时观察到4prefill和4decode真实共存，终态KV均归零；随后8条1024上下文KV
实际分配成功。all-prefill峰值allocated22,187,018,240B，mixed22,044,143,616B。
最大活跃KV4,294,967,296B。容量执行通过不等于全500数值/在线性能资格完成。

同一req_00015前131输出token完全一致，index131的原生score如下：

| 场景 | token11029 | token1024 | 实际选择 | 最大值并列数 |
|---|---:|---:|---:|---:|
| batch2，两份相同 | 13.734375 | 13.7109375 | 11029 | 1 |
| all-prefill8 | 13.703125 | 13.703125 | 1024 | 2 |
| mixed4/4 | 13.734375 | 13.71875 | 11029 | 1 |

四份完整32000维FP32向量已保存（每份128000B），包括原生分段、请求顺序和
SHA。all-prefill的tie由原始argmax选择较小index1024，直接解释D562/D564
req_00015分叉。batch2及mixed的该请求完整输出均与D445相同；all-prefill全部
8条输出与D562对应输出相同。这是直接决策证据，不是全算子正确性证明。

另一个独立未闭合项：mixed中的req_00104/ecommerce_lora_0045在index54首次
由10894变为5936，192/256位置不同；前54token相同，其余7条完整输出与D445一致。
本轮预登记只观察req_00015/index131，故没有这个位置的score，不补造“也是tie”
解释、不把全部输出标成相同。下一步对既有差异作有界同前缀核查，不逐次筛赢家；
正式serving仍为原batch2，新容量尚未接入性能主表。独立数值正确性继续未知。

所有原始输出与失败保留，诊断时间不用于吞吐或GPU-s优势。既有计划的全500、
在线生命周期、W0/W1/W2、新旧版本比较以及完整PrimeLoRA后续主线均未删除。
