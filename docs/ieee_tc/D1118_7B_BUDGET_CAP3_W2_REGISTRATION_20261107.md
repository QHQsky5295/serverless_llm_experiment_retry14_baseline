# D1118：固定 D1117 三副本配置的 7B W2 验证

## 完成交接：2026-11-07

本次唯一回放已经结束并释放，不再启动。完整结果及下一步见
`D1118_W2_RESULT_AND_BUDGET_WORKING_POINT_20261107.md`。
4000原生合同完整、联合3900=97.5%、U9873.292173262686 GPU-s、
P95 TTFT3053.004479035735ms；W2预算12086.669343876012通过。
与D1117同配置W1构成预算可行观测点，保留资源—尾延迟取舍及数值证明限制。
以下“运行中/尚未运行”均为历史登记。下一主线7B当前实现A1，不重跑基线。

## 当前状态：2026-11-07 18:41 已启动一次

结果/登记检查点`70cdac92606bec01458f53e03169864fac138a47`已推送且remoteHEAD一致。
tmux=`tc-d1118-7b-budget-w2`，panePID=1417227，运行中；不要重复启动。
service scope=`primelora-tc-svc-1396f87cc78845db88722e2b49c6eadd.scope`，
invocation=`167da10880c34e97b3e556382a78177e`，inode41312674；
aux=`primelora-tc-aux-025dcbaae7524e48b474cb47cc0002ae.scope`。
远端本次invocation=`be5ec5e0b6d846c387f4b48d54107c42`、PID949742。
初次health在服务就绪前连接失败，随后合同验证通过，均在部署通知之前。

通知monotonic3555284.928629879，replay_t0=3555344.928629879；
W2 view SHA=`5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7`，
source/map SHA与原登记一致。两端1Gbps/full；初始available121583210496B、
disk173307752448B，GPU无计算进程。sample34未触发告警/中止，内存events全0。
当前无最终性能结果。下面“尚未运行”属于启动前冻结登记。

下一步监控本次完整4000回放，不并行维护、备份、hash扫描或新实验。
结束后先核实资源真实归还，再按实际health clock收集remote journal；
复用既有原生计量与W2基线合同，不改共同阈值或重做对手。

## 启动前登记

2026-11-07，启动前登记，尚未运行。D1117/W1已完整结束、释放并通过预登记
原生合同/时延/预算条件，故依原定顺序执行W2，不新增调参候选。
先读D1117_W1_RESULT_20261107.md，保留其资源改善与时延退化两方面。

## 唯一执行合同

- 完全复用`configs/ieee_tc/20261107_d1117_7b_budget_cap3.yaml`，SHA
  `34562e22c517fc62033da1c3437ef95df616fd54c8ae0dcb20a3e59db289eb53`。
- D1100生产代码、IEEE九公式、vLLM0.30.0、FP16/TP1、14slots/seqs、1024
  batch tokens、min1/max3、38秒cooldown、45秒历史及在线自适应规律不变。
- 四卡物理可见，退役/激活交叠实际计费；上限3不是三卡固定预算。
- 仅通过现有runner切到原W2：相同arrival/prompt/token，rotation100。
  源traceSHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。
  既有adapter map `paper_results/ieee_tc/inputs/20261015_7b_w2_adapter_map_v1.json`，
  SHA `48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`。
- 4000请求、500逻辑adapter、固定greedy输出、共同60秒通知、真实远端已发布
  压缩对象不变；不生成新权重/负载，不继承D1117的HOST/NVMe缓存。
- 新结果/缓存路径由runner明确覆盖；不改YAML，不覆盖旧目录。

## 验收与解释

沿用冻结V1/单轮V2、D394 warm阈值和D594 Resident参考：
4000原生合同完整、共同联合达成至少3800/4000，
U≤12086.669343876012 GPU-s（0.75×D557参考16115.559125168016）。
共同参考P95=1751.2954722624272ms；与W1参考归一化P95等权分析。
原始n_correct未知、同后端输出差异与单轮已知trace限制继续保留。

保留全部失败和首次有效结果。若不达标，不重复同键、筛运行、改阈值或自动
继续max2；依据实际瓶颈返回主线。若达标，封存该跨W1/W2预算工作点并比较
已有五个论文基线，不以本轮成功替代其他待做实验。W0不属于该G2点新增要求，
不因此再跑W0或拼接resource_opt。此前D621旧cap3 W2失败保留，不遗忘其风险。

## 启动及安全

先备份D1117结果/本登记，检查GPU空闲、推理机内存≥102GiB、磁盘≥150GiB
及新增峰值余量；启动前重查两端1Gbps/full和远端服务归属。
采用既有72/80GiB服务、3/4GiB辅助、swap2/0、固定CPU域和外置watchdog。
无新生产修改，不重复无关100/1000请求资格。按run-experiment技能使用tmux：

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W2 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1118_20261107/7b_budget_cap3_w2 \
  d1118_7b_budget_cap3_w2 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1118_20261107/7b_nvme_budget_cap3_w2 \
  /dev/shm/tc-d1118-7b-budget-cap3-w2 \
  configs/ieee_tc/20261107_d1117_7b_budget_cap3.yaml
```

测量期间不并行重型任务、压缩、构建、哈希扫描、Git推送或远端管理变化。
收尾先实际释放，再绑定健康检查clock收集对应远端journal，不用clock拼文件名。
复用D1113 collector；W2分析沿用已经核验的D1100/D1113 W2合同及D593矩阵，
如需扩展现有比较入口，仅改离线显式run/output支持，不重建统计框架。
备份至faaslora_origin/retry14_continuous_queue_v2，不stage用户dirty文件。
全部M1/M2/A1–A5/S1–S13及文档、图表范围保持开放。
