# D157：7B 四并发的准入与分层初始化测量

## 结论和边界

2026-10-02，沿用 D155 的单一容量假设和 D156 已通过的四路原生并发能力，
完成一次带真实准入的分层标定。184/184 条原生请求通过；180 条代表性观测
覆盖完整 500-adapter 静态集合所需的 30/30 服务类、24/24 准备类，没有缺类。
新 profile 已通过现有严格加载器；实际 Full 构造入口的无 GPU 检查也已通过。
这允许下一次普通 Full 回放，不代表 7B 已达到 G1/G2，也不是正式 S1。

当前服务代码仍为 `39cc3c1147f1699a4e7cddb54996be2564a895c3`；测量时仓库
HEAD 为证据提交 `26e44a2014a9178148afddf1b18447fd6b6e5ab3`。没有新增权重、
trace、独立解压池或远端缓存。用户批准的一次性只读交付缓存继续原样复用。

## 假设与控制变量

D154 的平均 TTFT 为 103.898 s，其中 98.593% 是 dispatch 前等待；其最多
八个全局执行许可不仅覆盖生成，还覆盖准备和确认清理。D155 的真实 KV
观测支持有界地验证每卡四并发，D156 已证明四路 decode 确实重叠。
本轮仅补容量改变后必须重新测量的服务/准备 profile，不增加第二项优化。

- 配置沿用 `configs/ieee_tc/20261002_d156_7b_cap4.yaml`；max_num_seqs、runtime
  cap 为 4，四 LoRA slot、TP=1、FP16、显存比例 0.70、1024 上下文及 token
  预算、禁用 prefix cache 等保持原值。
- D88 的完整控制配置逐字段保持不变：需求窗口 5 s、扩缩容周期 2 s、service
  bin 27.43356142744625 ms、上下阈值、38 s cooldown、movement=3、EWMA=0.5。
- admission 使用 D156 已完成请求的长度初始化，不使用未来输出长度。
- 原 source42 请求中六个既有代表；46 个四路 wave，其中一个 warmup，45 个
  代表性 wave，交错三轮、五个源。admitted 分桶边界为 [1,2,4]。
- 保留真实远端 1 Gbit/s/full 链路、已发布压缩工件、相同传输协议，无模拟
  等待和请求内打包。没有网络调整、远端哈希扫描或清理与推理重叠。

因此后续是“容量及其新标定初始化”的开发比较；不能说所有时间估计仍与
cap2 完全相同，也不把必要的重新测量冒充另一项独立算法贡献。

## 测量结果

下表每个源有 36 条代表性观测、六个 adapter、三个交错轮次。
这些轮次属于同一运行，不能当三次独立重复，故不提供 t-CI。

| 起点 | 实际加载 d 均值 (ms) | admission→acquired D 均值 (ms) | acquired→首 token T 均值 (ms) | 原生 TPOT 均值 (ms/token) |
|---|---:|---:|---:|---:|
| Remote | 777.455 | 891.445 | 352.032 | 33.757 |
| NVMe | 70.535 | 177.956 | 312.640 | 33.777 |
| HOST 文件 | 71.217 | 192.521 | 304.164 | 33.702 |
| HOST tensor | 20.375 | 142.415 | 427.531 | 33.755 |
| 已保护 GPU | 不发生加载 | 0.000 | 296.295 | 33.626 |

GPU 的 D=0 来自受保护源的 acquired/admitted 定义，不是给缺失加载补零。
NVMe 可能命中页缓存，不能解释为物理磁盘读取时间。原生 TPOT 含真实批次
交错和调度间隔，不是 kernel 逐 token 时长。

观察支持复用本地源减少准备成本，但 HOST tensor 的 T 并非最小；不能用
最快层级单独推断完整请求时延。这里没有完整到达队列、自然扩缩容或新的
SLO 参考，不能把数值直接搬到主表或宣称系统优越。

## 完整性、资源和来源

- 184/184 原生实际输出数等于目标；完整 wave、源身份、时间分解、pending
  admission 关闭、reservation 释放均通过既有校验。六个重复输入组的输出
  hash 在本轮没有变化，但既有权重的数值区分限制仍未解决，`n_correct=null`。
- 30/30 服务类含两个 rank 类、五种源、三个 admitted 类；24 个准备类均有
  六条观测。六个 exact file-content 类不等于六个独立训练权重。
- 184 个远端 UUID 对逐一匹配：995,431,947 线上字节，4,892,930,430 逻辑字节；
  其中 36 个是被测 Remote 起点，其余为受控初态设置。全部内容/归档 SHA
  核验通过，打包次数为零。不能把所有 setup 下载算作自然 Full miss。
- 单卡物理持有 585.044383295 GPU-s，包含启动、设置、warmup 和收尾；这是
  标定占用，不是 G1 的完整工作负载 GPU-s/request。
- 591 个资源采样；服务主存峰值 5,413,052,416 B，主机最低可用
  106,817,986,560 B；观测到的 high/max/OOM/swap 均为零。
- launch 的服务与 watchdog 返回 0，物理租约已释放，原生 PID/cgroup 消失，
  工作目录清理通过。远端两个 artifact 服务及该轮监控按实际身份停止后，
  再复制日志并逐 SHA 核验。147 个历史保护条目未变。

原始结果：`results/ieee_tc/p2_backend_qualification/d157_20261002/7b_cap4_source1.json`，
SHA `542b258cdf2d95035ad86597ae79f597d81475e23333a34ed05ea1ec25bae4b3`。
汇总：`paper_results/ieee_tc/p2_backend/20261002_d157_7b_admission_source.json`。
初始化：`paper_results/ieee_tc/p2_backend/d157_7b_initialization/manifest.json`。
服务 profile SHA `d7b359b9b2ab23658df73e199983d7ed161310a568e962158c23c77c2dd9f8b1`；
准备 profile SHA `57b2d1d51dd5675e1bc16f7429c131c7fbc4d46ca139abe9d78b3e6f315816be`。

## 分析与图表检查

复用现有 `plot_paper_figures.py`、D88 汇总和 D89 profile 导出，不重建分析框架。
分析 attempt1 使用了不含 matplotlib 的模型环境，未生成结果；attempt2 被旧
固定 purpose 名称拒绝，也未改写数据。两次失败保留，不属于 GPU 测量失败。

分析入口新增显式 `--source-profile-purpose`：要求与原始 spec 精确匹配，原
默认仍拒绝 integration pilot；原生 token、时间、源、完整性检查全部保留。
manifest 记录真实和期望 purpose，输出仍声明 development-only、非正式 S1。
未篡改原 spec 或重标原始测量。46 项分析测试通过（包含两项新测试）。

attempt3 依次完成图表、汇总、严格 profile 导出及真实主入口的 CPU-only 构造
检查。后者使用 4000 请求、500 adapter，确认四卡 dispatch 容量为 16；在
任何引擎/网络执行前停止。它不是实际四卡吞吐或资源资格实验。

三个图位于 `figs/ieee_tc/p2_backend/d157_7b_admission_source1/`，附完整 CSV
和 provenance manifest。实际查看三张 PNG，文字无重叠或裁切；PDF 嵌入
Times New Roman regular/bold，宽 3.45 inch，下方加粗 (a)/(b)/(c)。保留完整
样本，无 CI；颜色按源区分，类别位置和标签亦可独立识别。

## 下一主线

先封存、备份本轮和新初始化，再用同一 W0/source42/4000 请求做一次普通
Full 回放。相对 D154 仅变容量及其观测初始化，保持其余控制和计量合同。
不再重复 bootstrap 或本次源标定；不增加 profiler、prefix cache 或超时。
7B 的数值 adapter、共同 warm/预算参考、旧/新 G1/G2 验收仍开放；3B 和外部
baseline 继续暂停。不能以胜过此前的慢候选代替逐模型目标验收。
