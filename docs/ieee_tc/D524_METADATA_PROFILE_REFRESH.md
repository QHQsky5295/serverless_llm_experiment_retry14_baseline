# D524：只刷新元数据优化影响的初始化时间

2026-10-14。D520–D522 的同一候选，不另加调度策略或更改九公式。

## 为什么需要补测

D523 原100请求 Full 在初始化时拒绝旧 D362 时间 profile：新增
`ieee_lora_metadata_policy=cpu_live_mapping_v1` 改变了精确执行身份。
请求尚未提交，没有新的性能观测。保留失败配置、日志与实际释放回执；
本机服务/辅助资源域已关闭，远端两服务及监控已停止。不更改共享 k3s。

原生元数据构造直接影响执行时间，不能将它当成仅启动时选项从 identity
中删去。D524 复用 D298 collector、prepare/export 入口，保留严格时间身份。
历史 D361→D362 曾对另一执行路径改变采用同样的受影响时间补测原则。

## 复用与变化范围

- 精确复用 D362 已合并的135组、1669次调用：34次保留warmup，1635次代表
  测量，顺序及原请求索引不变。不是1669次独立性能运行。
- 复用原 first1000 trace、500工件清单、已发布只读gzip。无新权重、负载、
  工件池，无请求中打包。
- C14、1024迭代token预算、HOST预算、allocator、控制交错、file-demand
  路径不变；模型配置相对D362仅新增明确metadata策略。
- 仅迁移旧fixed-output已完成长度。新长度绑定显式列出历史容量、control、
  file-demand与metadata差异；不迁移旧时延、SLO或数值正确性。
- 当前Full控制规则来源为D476，source collector并不执行其router/planner。
  新时间profile必须与后续Full的实际model identity完全匹配。
- 共同warm SLO及G1/G2尺子不变；这不是重新标定评价阈值。

## 验收和下一步

已有collector走真实全局HOST预算attach，再执行原生加载与生成。保存该现有
attach返回值及HOST账目，不增加测量请求中的RPC、GPU屏障或profiling。
检查实际publisher安装、99040B固定staging的allocator占用、累计调用、
原生buffer/invalidation保持、无poison/fallback。设备局部算术D522不重跑。

所有调用的原生token/adapter/时间合同、远端双侧字节与零打包、200个service
与24个preparation类覆盖、实际GPU释放及局部资源关闭均须通过；缺类不补零
或借旧时延。此处不证明全模型独立数值正确性，`n_correct`保持未知。

先补受影响的CPU绑定/导出测试及明确源码备份，再做一次测量。使用既有资源
限制、外置保护与tmux；测量期间不构建、压缩、哈希扫描、备份或调整远端。
失败和缺失保留。通过后新执行键原100 Full，再一次完整4000 W0判断收益。
完整7B W1/W2、3B、消融、敏感性和论文证据交付均继续保留。

RAW：`results/ieee_tc/p2_backend_qualification/d524_20261014/`。
汇总：`paper_results/ieee_tc/p2_backend/d524_metadata_profiles/`。
状态：尚未测量，不作收益或G1/G2最优性结论。

沿用D520/D522固定官方源码和设备证据：[vLLM0.30元数据路径](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/punica_wrapper/punica_gpu.py)。
本轮仅调整证据绑定/采集，不引入新的服务优化假设。
