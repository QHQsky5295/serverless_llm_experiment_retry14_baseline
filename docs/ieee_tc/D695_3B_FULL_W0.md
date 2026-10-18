# D695：同副本 GPU 准备资格的完整3B W0评估

2026-10-18。D693是唯一生产候选：按native owner派发最终GPU准备动作，
同owner最多一个，其他owner及HOST/NVMe仍并行，全局并发3不变。假设、
历史日志/代码和官方vLLM依据见D693文档。本轮不新增优化或参数搜索。

D694已完成100请求真实远端原生资格：100/100生成与绑定、4owner的162次
最终GPU动作重叠0，HOST交接19次；共同联合SLO97/100。该短资格不能替代
完整性能证据，故直接扩到4000，不再追加1000前缀或局部剖析。

## 冻结合同及单轮身份

- 主仓库父checkpoint为`0a8a458a050f42762d6ff3e246e382b2343b3deb`，baseline
  仓库保持`e16af33aaf41f02a62ecdce59f6f3227813244c3`，不改对手。
- 配置逐字等于D682全量配置，仅改D695输出/NVMe/HOST目录。与D694相比
  只有请求数100→4000和路径变化，保持同一生产候选，不按完整结果调阈值。
- vLLM0.30.0、FP16/TP1、8GPU LoRA/32CPU LoRA、rank16、8seq/4096batch、
  chunked prefill、无prefix cache、context1024、输出cap256、prompt759。
- 500个既有逻辑adapter、原seed42 W0 trace、D645 service/preparation profile、
  initial observed-arrivals、confirmed routing、handoff、hierarchy/admission原样。
  IEEE九公式/核心语义和现有HOST租约交接不变。
- 源trace SHA `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`；
  全量view SHA `0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37`。
  不新增权重/trace，不复制完整工件池，不使用未来请求信息。
- 真实HTTP使用D78/D80通知前发布gzip，运行中不打包、维护或扫描远端。
  真实传输、必要读取、物化和请求竞争计入延迟，无模拟等待。
- V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`，
  单轮V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`；
  D648共同SLO和D657资源参考不变。既有已知trace上的开发观察，不伪装独立留出。

## 预先规定的分析

1. 全4000原生合同、458224tokens，无fallback；dispatch与E2E分解完整。
2. 已保留movement日志验证每owner最终GPU动作不重叠、全部订阅/物理引用释放；
   保留GPU准备成功/延后、源确认/epoch/租约和所有旧机制触发证据。
3. 共同联合SLO至少3800/4000为时延可行条件；边际TTFT/TPOT、平均/P95/P99、
   E2E、阶段分解及8个500请求组完整报告。不给单项TPOT退化另设否决阈值。
4. 物理GPU-s来自真实allocation/release，含准备、业务、drain、释放，互斥
   分项之和等于总量。不用idle折扣或旧InfraGPU终端显示替代物理积分。
5. 与父候选D682（3694/92.35%，U15896.253912）及D672（3647/91.175%，
   U14931.314880）等完整历史点并列；D688退化91.125%及已撤回身份继续保留。
   不只挑慢对照，不重复同键求赢家，不计算运行间CI或显著性。
6. 独立数值正确性仍未知时保持n_correct=null，不因原生生成与绑定自动升级
   正式排名。G1不要求75%预算，G2另需W1/W2及共同预算，不由本W0宣告完成。
7. 完整结果决定候选去留；若未达标，先以本轮证据判断具体瓶颈，不堆叠补丁、
   放宽评价尺子或继续重复短前缀。所有失败和退化永久保留。

## 安全、存储和进度

复用D293 runner、D694资格/管理、D682完整流程。服务72/80GiB、swap2GiB；
辅助3/4GiB、swap0、CPU分离；一次一个重型，watchdog与真实整树释放保持。
推理机150GiB启动/100GiB安全停止、102GiB内存启动门槛不变，实时核验。
共享k3s不操作；HydraServe已有隔离适配/实际结果复用，不重启基础适配。

本轮存储仅无损压缩已闭合D684诊断main_outcome，原SHA
`d2235bf4c1c17f7ee2960f126cee898805f0538caa69d2683cfa992738dfcc39`与两次
解压一致，回收259760128B。新映射见
`paper_results/ieee_tc/storage/20261018_d695_closed_outcomes.json`；原审计与
结果身份不改。147个保护文件及用户manifest不变；已回收空间不重复计数。

当前启动前准备，尚无本轮性能结果。完成后按实际资源释放→远端停止→离线
审计→封存/备份推进。原始目录`results/ieee_tc/p2_backend_qualification/d695_20261018/`，
汇总`paper_results/ieee_tc/p2_backend/d695_3b_full_w0/`，不覆盖旧结果、不画图。
其余模型/场景/G2、3B基线/dLoRA、独立数值证据、A1–A5、S1–S13、成本/CE、
审稿文档与最终交付完整保留，整体目标ACTIVE。
