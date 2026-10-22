# D729：3B无损backbone流水加载完整W0

状态：准备完成，是否启动以raw/start_receipt.json为准；尚无性能结果。

前提D728：41项CPU检查与4个原生生命周期请求通过，物理资源已释放。
本执行相对D726仅增加`ieee_backbone_loader=fastsafetensors_affinity_preserving_v1`
以及新结果/NVMe/HOST cache路径。不改IEEE九公式、路由、hierarchy、admission、
controller阈值/窗口、CUDA Graph、C8、模型权重、LoRA、trace或生成契约。
160类service与preparation仍为D725同计算配置实测；不重复采样未改路径。

## 执行与判断

- 使用现有run_d293入口的d729 case，一次完整4000 W0，无短前缀/详细observer。
- 同一真实远端D80已发布gzip、共同-60秒通知；不请求打包，不扣传输/启动。
- 同一72/80GiB服务、4GiB外置、固定CPU/allocator/toolchain和保护门槛。
- 共同D648 SLO与冻结V1/单轮V2；至少3800条joint达标，物理U按实际占用。
- 参照D726 joint92.650%/U15884.281128，另保留D71991.600%/U13508.300779
  和D70393.600%/U14864.807562的版本差异；不只胜过最近点就称最终达标。
- 记录初始ready相对业务、首请求TTFT、扩缩容触发/持卡边界及全程SLO；
  不从权重加载阶段的局部差值推造节省。不隐藏原生KV容量的实测变化。
- 4000/4000原生计量/绑定与独立数值正确性分别报告，不追改n_correct未知。
- 本次是已知工作负载开发新执行键，非独立留出；不算运行间CI、不筛胜轮。

运行中不构建/备份/压缩/重哈希/调整远端或启动其他重型任务。不操作共享k3s。
完成后先确认本机物理释放再按InvocationID停止本轮远端和monitor，复用
audit_d703_3b_full与已有阶段分析保存完整/失败结果，再备份并返回主线。
全部3B后续、最终7B一致性、baseline/dLoRA、A1–A5、S1–S13、CE与审稿文档/
后期图表继续保留。此候选W0未达标时不自动启动W1/W2。
