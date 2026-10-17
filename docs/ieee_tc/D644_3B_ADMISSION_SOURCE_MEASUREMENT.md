# D644：当前3B准入启用来源测量完成

2026-10-17。执行版本`143b5ac066a3716ef14dc42d5187a5bfeb4b1f01`。
沿用D641模型参数、D642编译/启动合同，以及原24代表、46waves、368cases；
唯一必要变化是D643当前完成长度初始化和main对应的准入配置组装。
未重复source-only或改变生成/输入/权重，未把旧时间重新标为当前profile。

## 实际完成证据

- 368/368原生用例完成，其中360测量、8保留预热；实际token均等于目标。
- 既有`completed_gap_samples`严格校验通过：原生时钟/事件、工件身份、GPU
  引用释放及pending-KV关闭均完整，没有改动D642曾触发的拒绝条件。
- 覆盖40/40可达服务类别及96准备类别；缺失服务类、内容来源均为0。
- 启动92.776253秒；本次单卡实际生命周期515.976593 GPU-s，包含测量准备
  和收尾。它不是4000请求的GPU-s/request，也不是共同部署窗口主实验。
- 368次受控获取，共854130750线上字节；双端transfer ID、内容、归档SHA和
  字节对应，无请求内打包、临时归档。真实传输保留在路径中。
- 服务峰值5723078656B，主机最低可用105985019904B；high/max/OOM/OOMkill
  与服务swap均0。GPU已释放，launcher递归空/path消失，远端两个artifact
  服务和本轮monitor均已关闭。分析资源域峰值663367680B、事件/swap为0。
- 147个投稿保护文件和用户manifest未变；原始失败和D642记录均保留。

## 描述性来源结果

每种来源72条测量；不是72次独立运行，不计算运行间CI。

| 实际来源表示 | 获取均值ms | admission至首token均值ms | 原生TPOT均值ms |
|---|---:|---:|---:|
| 真实远端已发布归档 | 1385.260 | 1666.763 | 18.465 |
| NVMe文件 | 130.734 | 376.657 | 19.224 |
| HOST文件表示 | 127.056 | 372.943 | 19.110 |
| HOST原生tensor表示 | 56.093 | 411.103 | 17.905 |
| GPU已就绪 | 0.000 | 188.101 | 18.645 |

这些是五种来源表示，不是五级物理硬件。NVMe可能命中page cache，不宣称
测到物理盘读取。GPU获取为0源于admitted与acquired边界一致，服务时间不为0。
admission至首token不是主实验从计划到达计量的TTFT。

远端准备明显高于本地表示，为减少实际fetch提供测量动机；NVMe与文件HOST
接近，不能据此声称每一层都有明显优势。原生HOST获取更短却没有更短的
admission至首token均值，表明获取成本并不等于总服务收益。路由、handoff和
admission对全系统的因果作用仍须既定消融，不以这张描述表代替。

D642 source-only与本轮不是统计重复、也不是完整admission开关的因果消融。
编译/page cache、启动变化继续保留，不将两轮小差值归因于单个机制。
原生日志目录名中的`engine_dead`来自现有停机保留逻辑；本轮pass、释放和
完整用例表明不能仅凭目录名推定崩溃。

## 当前结论与下一主线

当前已取得准入启用后的真实服务样本，未导出或冻结最终服务/准备profile，
也没有共同3B SLO、Resident预算参考或Full性能资格；原始`n_correct`未知状态
不改写为独立数值正确性已经证明。

下一步直接复用本轮`completed_service_samples.json`与source1原始证据，通过
现有exporter完成3B服务/准备profile及runtime绑定，再推进共同warm/Resident
和Full W0/W1/W2。不得再次无变化地运行这368个用例，不套用7B阈值，不用
历史5000ms开发控制值宣布正式达标。D88控制在source collector中未执行，
是否保留到完整回放须按当前证据与既定公式明确记录。

其余3B基线/dLoRA、同版本比较、A1–A5、S1–S13、成本/CE、审稿文档与最终
图表全部ACTIVE。未绘图、未安装、未改论文、未接入无关k3s。

原始证据：`results/ieee_tc/p2_backend_qualification/d644_20261017/`；
小型归档：`paper_results/ieee_tc/p2_backend/d644_3b_admission_source/`。
