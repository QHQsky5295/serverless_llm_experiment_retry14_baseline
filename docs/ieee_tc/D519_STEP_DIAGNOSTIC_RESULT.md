# D519：一次真实远端迭代诊断的结果与限制

2026-10-14。唯一运行已完成并实际释放，源结果 SHA
`af2c2680e375c3c5d14720a810d30d9e6892852b7951fcdb6e1522083005e7b1`。
不是普通完整4000性能点，不用于主排名，不据此接受新的优化。

## 执行与完整性分开判断

- 原始1000请求，全部原生合同完成，0失败；请求/adapter/内容/lease/slot/
  原生时钟/输出数量审计各1000通过。独立数值正确性仍为未知，n_correct不补写。
- 已恢复D476服务语义，保留D518 opt-in观测。无新权重、负载、缓存阈值或SLO。
- 11个实际allocation全部释放；物理资源3002.6851364709437 GPU-s。
  此数仅用于本次仪器化诊断，不与普通Full据此宣称提高或降低。
- 平均/P95 TTFT为1474.804094/6948.303871ms，平均/P95 TPOT为
  35.259896/63.133829ms。全部观测开销保留，不事后扣除。
- 60次真实remote获取，74,953,014线上字节，按请求打包0。启动前实际链路
  1000Mb/s/full；远端服务在本地真实释放后才关闭，日志SHA已核对。
- 服务峰值20,621,922,304B，1188次资源采样；high/max/OOM/OOMkill/swap均0，
  没有资源告警。旧投稿147文件和用户修改仍受保护。

**完整分阶段诊断门槛未通过。** 主进程在末段触及64MiB日志上限，footer明确
truncated；11份日志中10份完整。原分析器按“整份进程日志完整”计算的请求
完整覆盖为279/1000，不能因此说其余721条请求全无记录。主进程已经保存
31360个完整step，最后完整root结束于1466563626875555ns；其后的1个孤立
schedule没有父step，保持孤立，不虚构父节点或close。

最初要求全覆盖的 analyze_components.py 明确拒绝，失败回执和源码保留。
随后单独生成 components_partial1：严格默认仍拒绝不完整日志，只有显式
partial模式分析实际完整root，结果字段 complete_diagnostic_qualified=false。
没有重跑GPU、放宽原资格或覆盖原始文件，不把这份局部统计写成全负载分布。

## 已观测完整迭代（非独立重复、非纯prefill/decode）

共47038个完整step。下表是同类step的无条件均值，未调用的component计0次；
嵌套时间不可相加。单token/multitoken是原生分配形状，不等于GPU kernel类别。

| 观测项 | 单token分配（45513） | 含multitoken（987） |
|---|---:|---:|
| step wall，ms | 23.669581 | 83.052146 |
| step线程CPU，ms | 7.468172 | 82.809049 |
| scheduler，ms | 0.230154 | 0.343925 |
| execute_model主机调用，ms | 4.324807 | 80.685168 |
| prepare_inputs，ms | 0.919288 | 0.852001 |
| _set_active_loras，ms | 2.100393 | 22.072073 |
| sample_tokens，ms | 1.987577 | 1.289916 |
| 驱动HOST allocation累计增量，ms | 0.125878 | 0.332239 |
| 驱动HOST free累计增量，ms | 10.206529 | 0.357847 |
| HOST allocation次数 | 4.421704 | 12.914894 |

全类别、每个进程、逐step和Type-1分位数保存在同目录CSV/JSON中，不删掉
zero-token/no-schedule类别。单token分配的fullgraph主机调用均值0.243713ms；
这不是GPU执行时间。LoRA激活单token的P50/P95为0.133625/0.403878ms、
最大159.992009ms，均值受到少量长调用影响；不能只用均值代表每次都慢。

## 解释与下一项

1. 直接HOST allocation耗时相对完整step较小，当前不支持把它作为主要瓶颈，
   不因“每步分配多次”就恢复无限锁页缓存或选择一个人工阈值。
2. HOST free计数覆盖后台线程。10.21ms不是可从23.67ms直接减掉的请求代价；
   它可能包含等待/同步并与GPU执行重叠。目前还没有因果干预证据。
3. 多token分配的execute_model及少量LoRA激活长调用值得进一步核查，但它们
   仍是主机调用，不能直接归为Python纯计算或GPU空闲。
4. 下一主线仅核查原生输出/锁页缓冲释放与LoRA激活的真实依赖、现有D56–60
   容量合同及官方实现；复用本次已有逐step数据，不重跑同一诊断、不做缓存
   参数搜索。若证明可避免的重复工作，推导有界复用/生命周期规则，再最小
   正确性验证和一次普通完整4000回放。尚无新候选准备或性能改善结论。

原生计数的单位与含义依据
[PyTorch 2.13实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/cuda/memory.py)，
步骤边界依据[vLLM 0.30 V2 runner](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/model_runner.py)。
本次127项原生/分析CPU测试、后续33项component及完整性测试通过，0 skip；
这些不是新增性能实验。原分析器旧文案“provisional ruler”不适用于本次：
实际采用已冻结D394共同阈值，其SHA和数值随结果保存。

7B W1/W2/G2及共同参考、3B与基线、dLoRA B6、A1–A5、S1–S13、离线成本/CE、
两轮审稿文档、最终图表和交付仍未完成。当前保留D476，不采纳D516；不重新
开始已经完成的HydraServe适配。图表继续延期，先解决目标指标对应的实际瓶颈。
