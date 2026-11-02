# D1027：独立文件并行确认的完整3B W2验证

2026-11-03。当前为预登记，尚未产生结果。接续D1026的94项相关回归和三个
真实工件的组件观察；仅组件发布wall下降10.43–18.93%，累计CPU反而增加，
故必须以完整回放判断净收益，不能由局部值推造SLO。

## 唯一改变与主线

相对D1022，只改变发布事务内独立文件SHA校验的执行并行性，保留owner锁、
全部内容/签名/容量检查、join后发布和失败清理。D628解锁hash等已否决路线
不重开，不叠加capture或控制策略；九公式与核心机制不变。具体历史、原始
资料与CPU证据见D1026，不新增短GPU诊断或后端画像。

配置`configs/ieee_tc/20261103_d1027_3b_parallel_confirmation_w2.yaml` SHA
a5898c34f9abdb33cd3c82e4f3ef944a0c70865598beafe130aa6c4a0e58faa3。
与D1022结构比较后，除三个输出/缓存路径外所有字段逐项相等。32 GPU/CPU
LoRA slots、并发8、native active-LoRA specialization、原D1021服务/准备
画像、原生vLLM0.30、D747 descriptor、stdlib gzip、1–3实例、全部控制参数
保持不变。画像对应的原生后端未变；准备初值保留同一冻结观测，在线仍按原
EWMA更新，不按场景或正式结果另调参数。

输入复用既有4000请求、500-adapter universe、seed42、W2 rotation100、
同arrival/prompt/target与fixed_length_greedy_v1；原source trace已用于开发，
不是独立留出。60秒部署通知、真实已发布远端工件、完整生命周期计量不变。
原W2 map SHA为28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23。
不覆盖D1022，不重跑同key找胜出。

PLAN、V1、单轮V2的SHA与D1026一致；D648共同SLO不改：prompt≤506/>506的
TTFT分别1466.050801102/2539.648668191ms，TPOT分别29.921872367/
36.215515193ms。G1要求全部正确且至少3800联合达标，再比较物理GPU-s；
G2另需U≤12083.063870GPU-s，再比较W2 P95及观测SLO差。独立数值正确性
仍须按真实证据判断，不能用binding/token通过填造n_correct。

参照D1022：3770联合达标（94.25%）、8062.415716 GPU-s、P95 TTFT
1997.601528ms。并列报告TTFT/TPOT/E2E、失败、来源、资源与CPU代价，
不只看比旧候选提升。外部D810基线完整观测保留；达标后立即推进同版本
W0/W1及全基线比较，不无限停留在局部校验微测。

## 入口与保护

```bash
env -u FAASLORA_TC_CORE_TIMING -u FAASLORA_TC_CORE_STEP_TIMING -u FAASLORA_TC_HOST_INVENTORY_TIMING \
  FAASLORA_IEEE_HOST_INVENTORY_MODE=descriptor \
  FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so \
  FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044 \
  FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=stdlib \
  FAASLORA_D1000_RUN_ID=d1027_20261103 \
  FAASLORA_D1000_RUN_TAG=d1027_3b_parallel_confirmation_w2_full \
  FAASLORA_D1000_CONFIG=configs/ieee_tc/20261103_d1027_3b_parallel_confirmation_w2.yaml \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

原入口文件名含W1，实际固定W2；不设prefix。外层tmux保持长运行，已有资源
入口在启动worker前建立服务72/80GiB、swap2、CPU4–23/28–47；辅助3/4GiB、
swap0、CPU2/3/26/27及独立watchdog。空机内存≥102GiB，磁盘≥150GiB且
覆盖32GiB预计新增量的规则，运行时16/100GiB保护。一次一个重型任务，
不与编译、维护、备份、远端配置变更并行。

预查远端：认证health为artifact_timing_v2/prepublished_gzip_v1，clock_id
b7a04b62f38c4ae6adabb4da5c0fd8dc，端口18080 PID1984520，沿用D1016–D1021
已记录进程；它不属于旧failed的primelora-artifact-tc-3b.service。本轮仅
只读确认，不错误重启旧unit。实际管理归属为远端session-8529.scope。
两端sysfs均1000Mb/s，远端full；不把D77历史100Mb/s当当前链路状态。
没有请求中打包、额外复制池、共享k3s操作或远端后台压力测试。

raw独占目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1027_20261103/d1027_3b_parallel_confirmation_w2_full`。
完整运行后先验证实际release，再复用D703 outcome审计入口显式指定本配置和
D1021 manifest，恢复SLO/生命周期/原生生成/双边传输和阶段证据。失败和
退化照实保留，不能删点或换尺子；不提前绘图。

本文件是当前执行交接，不覆盖用户dirty EXECUTION_STATUS；其SHA仍为
41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374。
所有未完范围仍保留：同版本两模型W0/W1/W2及G1/G2、独立数值正确性、
A1–A5、S1–S13、CE/Pareto、B1/B2/C1/C2/C3及两轮审稿回应、后期图表与
最终V2集成。当前目标active，不宣布3B完成或全项目完成。
