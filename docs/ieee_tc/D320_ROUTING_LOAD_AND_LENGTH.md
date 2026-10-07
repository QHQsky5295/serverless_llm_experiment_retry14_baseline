# D320：排除负载漏计，转向服务时间的长度区分能力

2026-10-07，7B P2/P3 开发诊断。复用 D315、D318 两轮完整 W0；本轮没有
新推理、远端操作、图形或 serving 修改。计划和冻结指标 V1 均未修改。
这是定位下一优化对象的证据，不是性能改进或正式 G1/G2 达标结果。

## 问题、源码与相关工作

D319 已排除“尾延迟差异完全来自两次再扩容”的解释。本轮只检验：路由的
Q_i 是否遗漏 preparing/native unfinished 请求，从而在副本生命周期变化后
错误低估负载。历史 b9975ef 的 confirmed-copy 缓存与更早 request-local
快照改动是核查对象，不预先假定它们存在缺陷。

实际实现：`_ieee_request_snapshot` 在最后一次 native await 后同步读取
file snapshot、active_requests、active_adapters、pending 和 service class；
选中后同步 reserve。准备前 active_requests 已增加；原生 terminal/retirement
与引用释放确认之后才减少。pending-load 在 acquisition ack 后清除。
缓存的源副本信息不是缓存的实时 Q。32 并发 reader 测试覆盖共享 native
collection、逐次重读实时计数。不能从“cached source”推断“cached queue”。

核对的原始资料：

- [vLLM 0.30 scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)
  分别维护 waiting/running/finished 与异步依赖；不能只把 running 当成全部工作。
- [dLoRA，OSDI24](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
  讨论副本内队列与跨副本请求/adapter 负载平衡。其迁移目标不是本论文公式，
  不据此添加 affinity 系数或修改本系统路由排序。
- [Llumnix 官方调度说明](https://docs.llumnix.ai/design/scheduler/instant_accurate_load)
  区分观测及时性与负载度量准确性；这个区分适用于本次诊断，但不移植其
  预测器。自然停止输出未知，与本项目 fixed-output 的已声明长度上限不同。

## 数据方法与局限

所有成功请求的最终 predecision snapshot 全量进入核查，不选有利请求。
对每个候选副本，在 file capture 时刻 t，其他请求满足
`controller_admitted <= t < native_last_token`，则它一定仍属该副本。
其数量是 Q 的保守下界。dispatch 前和 native 执行分别计数；非 GPU 源且
尚未 acquired 的请求另给 pending 下界；核验 active adapter 集合包含这些请求。

历史 schema 没有精确 controller reserve/release 时间，所以这不是精确重建：
admission 前的 reservation、last token 后的 cleanup 仍可能合法占用 Q。
零矛盾只说明没有发现漏计反例，不证明任意时刻的计数绝对正确。
另按冻结 admitted edges [1,2,4,8] 核验 class，按原 bin width 和完整
lexicographic key 重放实际选择；不生成反事实延迟。

| 检查 | D315 | D318 |
|---|---:|---:|
| 完整请求 | 4000 | 4000 |
| 候选副本视图 | 15919 | 15473 |
| 有已知 native 未完成工作的视图 | 8648 | 8099 |
| 有已知 pre-dispatch 工作的视图 | 1549 | 1601 |
| 有已知 pending load 的视图 | 1027 | 1059 |
| Q 小于可证下界 | 0 | 0 |
| pending 小于可证下界 | 0 | 0 |
| 缺失已知 active adapter | 0 | 0 |
| admitted class 不一致 | 0 | 0 |
| 选择偏离原公式 | 0 | 0 |
| 所选 Q 高于可行集最小 Q 的请求 | 2536 | 2807 |

**决定：不支持“漏计负载”这个实现缺陷，不做相应 serving 补丁。** 最后一行
也不是缺陷判据：同公式允许准备/服务时间优势先于 Q tie-break。

## 更具体的下一假设：输出长度类别过粗

D298 service profile 的 declared_output_tokens 唯一有限边界是 256；当前
全部请求的 2–256 token 工作量共享 output bin 0，在线更新也混在该类别。
复用 D319 的已验证 native interval，连接本轮所选副本的预测 O，不读新的
完整日志、不再做 overlap 分组。下面的长度组是已有描述性分组，不是已选定
的新配置。固定输出计数已与 declared target 核验；没有在线使用未来长度。

| 输出 token 组（每轮 n） | D315 预测/实测 O，秒 | D318 预测/实测 O，秒 |
|---|---:|---:|
| 2–32（1114） | 3.093 / 0.493 | 3.090 / 0.521 |
| 33–128（1442） | 3.147 / 2.311 | 3.208 / 2.362 |
| 129–256（1444） | 3.497 / 6.530 | 3.581 / 6.582 |
| 全部（4000） | 3.258 / 3.328 | 3.310 / 3.373 |

平均绝对 O 预测误差分别为 2.459、2.453 秒。整体有符号误差只有
-0.070、-0.063 秒，却掩盖了短请求高估、长请求低估。这支持检查长度类别
的辨别能力，而不是继续检查已经关闭的 Q 漏计假设。

在所选 Q 较高的请求中，1833/1907 条相对一个最轻载候选没有预测 D 优势；
取该集合中 service bin、预测总时间、replica ID 排序的候选作诊断参照。
这不证明所选位置更慢：未选位置的实际 T/O 未执行，不能声称减少多少
TTFT、TPOT 或 GPU-s。输出差异、正式数值资格未知仍按 D318 原记录保留。

## 下一主线与禁止重复

1. 下一轮 7B 只考察 declared-output 类别/初始化这一候选方向。先读 IEEE
   类别语义、D298/D307 历史和已有 profile 导出器，保持 D+T+O、EWMA、
   route tuple、admission、扩缩容、输入工作量和 V1 评价尺子不变。
2. 优先复用过去已完成的开发请求样本，审计 admission-time 特征与类覆盖。
   合法静态/历史长度边界在运行前冻结；不能按正式结果扫描分组，不能把
   缺类补零、池化或邻近类值伪装成实测。仅缺失且必要的测量才补。
3. 构成一个可执行候选后，按既有最小验证 → 完整4000W0 比較决定接受或
   撤销。预测误差变小不是性能改善的替代；不换控制系数来保证胜出。
4. 本轮 Q/长度诊断到此关闭，不继续把现有数据切成更多 overlap/cohort；
   不重开512 token预算、D294/D295数值小重复或旧 profile 全矩阵。
5. 7B 达标前不推进3B、外部baseline或新图。正式 numerical/warm/Resident/
   oldPrime新指标证据仍未完成，不宣布最优 Full。

## 产物、复核与资源

复用 `analyze_d297_full.py`，新增离线 mode 和7项反例测试，未新建实验框架。
`d320_routing_load/analysis1` 保存候选视图、摘要与 SHA 清单；`length1` 保存
每请求预测/观测和分组摘要。analysis1 执行后只新增独立 length mode；其
准确原脚本另存 provenance 快照（非入口），SHA 与原 manifest 完全相同，
避免为脚本扩展重复读取两份完整日志。旧产物不覆盖。

最终334项CPU测试通过3.776秒；此前274项通过3.605秒。完整日志分析资源域
峰值1212862464B，长度连接82448384B，最终测试556945408B；均 high/max/
OOM/swap 为0，实际 procs 空、populated0 后按 InvocationID 关闭。没有新
GPU 实验或远端服务启动。保护、源/输出复核与备份见 validation_checkpoint。

本轮结果分析技能用于分离观察、解释和下一验证；vLLM 技能只辅助定位调度
路径，不采用其通用性能数字。提交前按 github-sync 规范只加入本轮显式路径，
不加入用户 manifest、旧图、raw logs 或凭据。
