# D337：少卡区间的解码代价与启动区间的服务退化

2026-10-08，7B 优化主线。复用 D335/D336 各 4,000 条完整请求和既有
生命周期、worker 日志；没有新 GPU 回放、远端操作、服务代码/配置修改、
工件/负载生成或图形。D336 完整审计和备份已结束，本项不是重复该审计。

## 方法和边界

仅在现有 `analyze_d297_full.py` 注册固定 D335→D336 对照，复用 D319/D327
的同钟区间交集和共同到达集合。新输出目录拒绝覆盖。以两运行实际
startup/release-gap 时间并集定义共同请求组，不按延迟挑窗口。
两边最后 engine-ready 的共同边界为业务开始后 2739.884672 秒。

全部 8,000 条请求区间及 207 个输出 token SHA 差异保留。输入/到达配对
不代表随机因果试验；同一 trace 的一次开发运行/配置不出 CI。原生计数和
绑定不替代数值资格，`N_correct` 仍未知。下表使用 D170 暂定时间阈值，
不是正式联合 SLO，不据此关闭 G1/G2、共同 warm/Resident 或新旧 Prime 验收。

计划 SHA：`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`。
V1 SHA：`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
来源结果 SHA、完整字段和派生产物 SHA 见 analysis1/manifest.json；不重写源测量。

## 共同请求集合：观察结果

| 同一请求集合 | 数量 | D335→D336 P95 TTFT/ms | D335→D336 P95 TPOT/ms | 暂定联合时间超阈值数 |
|---|---:|---:|---:|---:|
| 全部 | 4000 | 2857.905→2912.194 | 52.747→59.083 | 133→163 |
| 任一运行 startup 期间到达 | 288 | 2506.895→5595.268 | 99.417→102.813 | 22→51 |
| 不在上述 startup 期间到达 | 3712 | 2864.482→2678.123 | 49.938→55.409 | 111→112 |
| 任一运行物理释放间隔内到达 | 2033 | 2815.179→2762.798 | 57.089→66.388 | 78→81 |
| 两边最后 engine-ready 后到达 | 946 | 2940.804→3225.216 | 44.406→40.694 | 26→36 |

1. 全体净增 30 条暂定联合超阈值，其中 startup 共同组净增 29 条，非
   startup 组净增 1 条。这是互斥集合的计数恒等式，**不是“启动导致
   96.7% 退化”的因果贡献估计**。startup 组平均 arrival→admission 为
   513.089→1163.377 ms，service TTFT 为 678.574→811.657 ms。
2. 少卡区间的 TTFT 并未整体恶化，P95 TPOT 则增加约 16.29%。全体请求
   decode 期间其他 native 请求的时间加权占用均值为 0.983→1.395。
   native-request 区间不是 GPU kernel 活跃时间；不能由该值推定利用率。
3. 请求关联 native load/acquire 次数 1977→1975，总事务时间
   71.086→71.201 秒，单次均值 35.956→36.051 ms。总量没有明显增加，
   不支持“加载次数暴增”这一简单解释；不排除时序/重叠影响。
4. 与 peer-prefill 包络重叠的 decode 请求为 1950→2288；D336 的 80 条
   暂定 TPOT 超阈值中有 76 条属于该组，但慢 decode 更易发生重叠，
   因而不能把重叠秒数当作可减去的阻塞时间，也不能据此重开已失败的
   512-token 预算扫描。
5. 所有启动结束后的共同 946 请求仍有 TTFT 退化，而 P95 TPOT 改善。
   因此缩短启动值得优先检查，但不会预先宣称它可以解决全部性能问题。
   不同分组的分位数不能相加分解全局 P95。

完整 load/prefill/startup、输出长度交叉组和空组 N/A 全部保存，无选择性删组。

## 已有启动日志：编译不是主要解释

通过真实 native worker PID 将请求 owner 与 11 个 activation 绑定；逐一读取
service.log 指向的保留 workdir。一个重复归档的启动字段一致，去重不计作
额外副本。目录名 `engine_dead` 也出现在正常收尾，不据此宣布请求失败。

| 原有日志中的时间 | 最小/平均/最大（秒） |
|---|---:|
| 完整 activation→engine-ready | 44.314 / 47.371 / 51.429 |
| model loading | 6.660 / 9.379 / 12.049 |
| profile、KV cache、warmup 合计 | 10.440 / 11.501 / 12.850 |
| torch.compile | 0.260 / 0.279 / 0.300 |

11/11 已读取 AOT compile cache、采用已实现 checkpoint prefetch；各记录两轮
graph capture。weights load 嵌套于 model load，compile/capture 又嵌套于
init，不能全部相加。日志值有舍入，不与原生单调时钟跨域相减。

11/11 worker 还记录一次业务期 `_lora_shrink_kernel` JIT 提示。提示没有
持续时间，不证明它是尾部主因；不以移走提示或免费提前执行来“修复”数据。

## 历史、官方依据与下一项问题

已核对 D278/D290/D293（启动/预取）、D304–D307/D314–D315（decode与
512预算取舍）、D319/D327（容量窗口）、D328–D330（当前控制路径与
拒绝不可靠 HOST 缓存）、D333–D336（GPU core-turn 与 soft target）。
不重复预取、缓存命中、低开销快照或较小 token-budget 作为新优化。

本轮再次联网核查：

- [vLLM 0.30 优化说明](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)：启动优化有独立边界；取消 CUDA Graph 会改变稳态性能，不能假定有净收益。
- [同版 engine core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：模型初始化与 KV/compile/warmup 分阶段；utility 与推理循环的关系不等于所有时延都由 RPC 造成。
- [同版 LoRA manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：原生加载/缓存生命周期须保留，不能用少算加载或弱化 confirmed-state 减少开销。

**下一项只检验：fresh frontend/core 的 Python 初始化、启动握手和 ready
发布路径，是否存在可消除的重复工作，使副本在相同后端能力下更早 ready。**
约47秒总启动不能全部归为磁盘或编译，更不能把尚未分解的剩余时间直接
命名为 import 开销。先利用现有启动记录和源码定位；必要时进行一次无 GPU
的有界 import/启动阶段测量，再选一项可验证实现改动。保持新鲜 CUDA 可见
设备身份、单 owner、真实 GPU allocation/release 和全部初始化计费。
不 fork 已初始化 CUDA 的进程、不保留隐形 GPU 常驻、不换 SLO/公式/准备
窗口、不仅为 Prime 免费预热。若无可消除开销，明确归档该假设而非兜底重试。

这次完整日志定位到此结束，不继续无限划分相同数据、不重跑 D336 寻找赢家。
7B 仍未正式达标；随后才是 3B、基线、比较、消融与敏感性。用户最新要求下不画图。

## 验证与保存

73 项 CPU 测试通过（6.232 秒），包含新对照同方法、反向对照和覆盖拒绝。
两个完整运行离线分析退出0，峰值1017884672 B，3/4 GiB、swap0、
CPU2/3/26/27；high/max/OOM/swap 均0。原分析资源域按实际身份和空进程关闭。
启动日志提取复用同包络，未做新的推理。

数据位于 `paper_results/ieee_tc/p2_backend/d337_capacity_lifecycle/`；
原执行回执位于 `results/ieee_tc/p2_backend_qualification/d337_20261008/`。
当前磁盘仍略低于150 GiB，新重型实验前须安全恢复空间；不放宽门槛、
不删除唯一证据。用户 dirty manifest、旧投稿图表和原始结果不覆盖。
