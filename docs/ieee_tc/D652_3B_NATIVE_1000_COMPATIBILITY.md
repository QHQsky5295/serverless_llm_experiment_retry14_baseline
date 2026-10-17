# D652：3B 原生 vLLM 原始 1,000 请求兼容检查

2026-10-18。沿计划 §6.7，复用 D650 的100请求路径、D651全部500-ID功能覆盖
和 D649 的89项CPU检查；不重做已完成资格，不修改服务源码或搜索基线性能参数。
下一步为完整4,000请求Resident W0/W1/W2，可按同键同时用作vLLM基线与U_ref。

## 本轮合同

- 原3B trace前1,000请求，60个实际ID，目标122,510原生tokens。
  source SHA `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`；
  view SHA `7b8321203834ea96b9913c3444f99169e6d15ac56878644eaf3617d447d99fd7`。
- 同一既有500逻辑ID工件池，pool SHA
  `bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275`；
  `trace_prefix_v1`、`fixed_length_greedy_v1`，不重新生成trace或权重。
- 原生vLLM0.30.0，D649冻结的3B四卡TP=1/request-ordinal round-robin配置；
  原生缓存，无PrimeLoRA策略；60秒共同部署通知、独立开放到达、1800秒资格保护。
- D648共同3B SLO SHA
  `d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`。
  共同V1/单轮V2不变；本轮保留已知trace的开发/兼容身份，非独立留出测试。
- first-touch由真实远端已发布只读gzip对象获取；本轮LoRA下载缓存为空，
  不清原生编译磁盘缓存、不提前启动后端、不请求内打包、不减掉真实传输时延。
- 执行MAIN `d0068d6d4daf87f8558c28a909cff80cae25a34f`，baseline
  `e16af33aaf41f02a62ecdce59f6f3227813244c3`；启动核对两仓库备份和用户修改SHA。

原始目录：`results/ieee_tc/vllm_qualification/d652_20261018/`。
tmux `tc-d652-vllm-3b1000`，私有缓存 `/tmp/tc-d652-3bv`。沿现有gated-launch/
native-http-service和D651远端管理、收尾、分析入口，只调整campaign、输入视图
与结果名；新增峰值按32GiB保守检查，推理机150/100GiB和防OOM协议不变。

## 验收与边界

1,000/1,000请求、原生token逐条目标一致、请求/adapter/模型/worker/内容绑定、
无fallback、原生计时恒等式与TPOT重算≤1ms、实际物理持卡/释放及资源事件完整。
共享分析器显式 `--model-key 3b --population prefix1000 --scheduler-contract
resident_rr_v1`，不得继承7B阈值。独立数值证明未知保留，不由HTTP200或长度
单独推定LoRA计算正确。不按SLO高低重跑同键或调整基线。

结束后释放→停止本次远端服务→保存/审计→备份，再回收经核验可重建下载副本。
当前步骤不是完整W0/G1/G2，不以短负载GPU-s线性外推U_ref。完整3B Full/
其余baseline/dLoRA、A1–A5、S1–S13、同版本主表/消融与最终交付均仍开放。

## 单轮结果与解释

| 项目 | 观测值 |
|---|---:|
| 计划/到达/提交/终态 | 1,000/1,000/1,000/1,000 |
| 失败 | 0 |
| 原生绑定、目标长度核验 | 1,000/1,000；122,510 tokens |
| 实际逻辑 adapter | 60 |
| TTFT/TPOT/联合达标 | 1,000/999/999 |
| 联合 SLO | 99.9% |
| 平均/P95 TTFT | 146.920263 / 383.093254 ms |
| 平均/P95 TPOT | 10.978950 / 11.658600 ms |
| 平均 E2E | 1,477.281892 ms |
| 生命周期物理 GPU-s | 4,582.450818 |
| 真实远端获取/本地复用请求 | 60 / 940 |
| 线上/物化字节 | 139,254,428 / 2,336,314,960 |
| 请求路径内打包 | 0 |

GPU 时间互斥分项：业务前233.310340、到达窗4317.657120、drain 5.311130、
清理26.172228 GPU-s；总量逐设备按实际持有区间积分，非逻辑副本数或折扣账单。
时间恒等式、TPOT和原生计时重算最大误差均0ms。服务峰值15,737,495,552B，
主机最低可用97,561,296,896B，high/max/OOM/OOMkill与服务swap均0。
GPU真实释放，服务和launcher退出；只停止本轮所属的远端两个服务和monitor。
已有其他项目的k3s未修改，不重复已完成的HydraServe基础适配。

复用D422的逐请求原生序列比较方法，对照D650原100请求：全部request contract
和原生prompt一致，98条输出序列完全相同；req_00072与req_00079（finance_lora）
分别从0-based位置42、40开始不同。保存逐条hash和首差异位置；没有对应独立
HF/PEFT或logit-margin证据，原因保持未知，不笼统归为舍入误差，也不把一致的
98条当作独立数值正确性证明。该离线检查不新增GPU运行、不以exact-match筛选
性能结果，不能外推其余900条。原始n_correct=null与独立数值证明未知继续保留。

这是固定已知源trace的兼容性观测，短前缀与500-ID功能视图的输入分布不同，
不由二者数值差异推断系统优化收益。当前没有修改服务算法或基线参数。
同一冻结配置下一步只运行完整4,000请求Resident W0/W1/W2，每个执行键一次，
符合合同者复用为vLLM基线与资源参考；不重跑本轮、不由短负载外推资源预算。
随后3B PrimeLoRA/其余baseline/dLoRA和全部原计划消融、机制、敏感性继续执行。

共享分析：`audit_prefix1000.json`；离线序列检查：`native_prefix_comparison.json`。
小型发布目录：`paper_results/ieee_tc/vllm/d652_3b_native_compatibility/`。
原始请求和远端获取记录保留；备份后仅安全回收经全SHA/引用核验的下载副本。
