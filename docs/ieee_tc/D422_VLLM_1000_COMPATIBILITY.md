# D422：vLLM 7B 原始 1,000 请求兼容检查（预登记）

2026-10-11。16:42:25 启动，17:01:50 前结束并释放，严格兼容审计通过。唯一原始结果目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d422_20261011`。

目的：完成计划 §6.7 中原始负载的 1,000 请求兼容检查，然后推进一次完整
4,000 请求 W0。复用 D419 原生路径、D421 全 500-ID 功能证据；不重复已通过的
100/500 检查，不进行参数搜索，不使用功能视图性能选择配置。

复用既有 seed42 trace 的前 1,000 行（`trace_prefix_v1`），不改 adapter 映射，
60 个实际 ID、122,510 目标 token；视图 SHA
`0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981`。
源 trace SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
pool index SHA `e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c`。
不生成新 trace/权重，不复制完整池。

原生配置继续为 vLLM 0.30.0，DP4/TP1、原序号 round-robin、native LoRARequest/LRU，
FP16、max-seqs2、max-loras4、CPU-loras16、rank64、1024 context/token budget，
graph、chunked prefill、无 prefix cache、seed0。配置 SHA
`361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5`。
baseline 当前备份 `d9971a14e18eca3391665d9da1244594e69b6041`；本次不改运行代码。

共同 60 秒部署通知、独立 pre-ready open-loop 到达、1,800 秒资格保护；
服务 72/80 GiB、swap2 GiB，外置回放/监控 4 GiB、swap0，同冻结 CPU 包络。
预计新增峰值 8 GiB，启动至少 150 GiB，运行底线 100 GiB。
per-run LoRA cache 为空；现有原生编译磁盘缓存单独登记 SHA，不预启动 GPU 引擎。
复用 D78/D80 只读发布缓存；不重新发布、按请求打包、业务期间操作远端服务。

raw 中 `run7b.sh` 复用既有 gated-launch/native-http-service；远端启动/关闭、
输入冻结和证据保存只作 campaign/视图参数替换。实际执行 commit 由启动前的
`input_identity.json` 固定，不把后续结果备份 commit 冒充运行版本。

验收：1,000 offered/submitted/terminal/native/source/prompt/adapter/content/timing
合同完整、无 fallback、逐请求 actual==target；原生时间重算误差≤1ms；四 worker
来源/资源读回、真实 HTTP 收据与内容相符、物理 GPU 租约完整关闭、旧保护清单零变化。
使用现有 analyzer：`--scheduler-contract resident_rr_v1 --population prefix1000`。
共同 D394 SLO 仅作此短兼容视图描述；不是完整 W0 排名、U_ref 或跨场景最优。
重复/零权重和独立数值证据的限制保留，原始未知正确性字段不补写。

终态顺序：实际 GPU 释放 → 按本次身份关闭远端 → 保存证据 → 严格分析 → 备份。
任何失败保留；不因短视图联合 SLO 较低而重复。之后一次完整 W0，PrimeLoRA D371
和 ServerlessLLM D414 不重跑；U_ref/W1/W2/G2、3B、其他基线和后续完整矩阵仍开放。

实际 main 执行 commit `9f334caab8a2c305f7e29909532a65f7cc8db1d1` 已推送并核对远端；
baseline commit 如上。本次服务 InvocationID `9d9487f825874614acb5d7a94e0f29a7`，
tmux `tc-d422-7b`。628 文件/54387041 字节原生编译磁盘缓存的当前身份已冻结，
与冷 LoRA 缓存分开。启动前 27 项已有分析 CPU 测试通过；没有新后端代码改动。

## 终态结果与证据边界

| 检查/指标 | 单轮观测 |
|---|---:|
| offered / submitted / terminal / 原生绑定合同 |1000 / 1000 / 1000 / 1000|
| 失败 / 实际 adapter ID / 原生输出 token |0 / 60 / 122510|
| E2E / TPOT / 原生边界最大重算误差 |0 / 0 / 0 ms|
| 共同 TTFT / TPOT / 联合达标数 |991 / 1000 / 991|
| 联合 SLO 达成率 |99.1%|
| 平均 / Type-1 P95 TTFT |455.189460 / 2118.832637 ms|
| 平均 / Type-1 P95 TPOT |24.974164 / 27.243896 ms|
| 平均 E2E |3490.662821 ms|
| 生命周期 GPU 占用 |4601.208847534144 GPU-s|
| GPU-s / 绑定合同合格请求 |4.601208847534144|
| 实际 HTTP 获取 / 本地复用 / 请求时打包 |60 / 940 / 0|
| 线上压缩字节 / 核验物化字节 |74953014 / 1474429899|

四张卡的物理租约均关闭，互斥窗口 GPU-s 为：到达前234.1877145266626，
到达窗口4317.657119750977，drain25.14550943300128，清理24.218503823503852。
原生 worker ready 为通知后40.90034856至41.95850344秒，均在共同业务起点前；
原生编译磁盘缓存628文件/54387041字节已在通知前记录，不是免费预启动引擎。
它与 D419 的编译缓存初态不同，不能把不同前缀的延迟变化全部归因于服务策略。

源序号0–99、100–499、500–999的联合达标数分别100/100、395/400、496/500。
九条未达标均为 TTFT；没有从主分母排除启动或较慢请求。分段只作诊断，不是独立重复。

复用 D385 独立 HF/PEFT 原始前100参考：97条完整序列一致，3条差异均具有已有
首分歧位置/top2 margin见证。与 D419 同后端前100相比93条一致、7条变化，全部
逐请求保留于 `bounded_reference.json`。不把差异一概称为舍入，不新增GPU数值实验，
不强加跨后端逐token一致验收；其余900条没有新增独立数值参考，原始unknown字段不改写。
此前 D421 的500-ID执行覆盖复用，不重新跑。工件多数为零/重复权重的限制依旧存在。

服务内存峰值12567740416字节，主机最低可用100156542976字节，磁盘最低
162226290688字节；high/max/OOM/swap均0。实际GPU已释放、服务域已移除、退出码0；
三个远端服务按本次InvocationID关闭，MainPID0且端口无监听。remote源/副本SHA一致，
保护147条零变化；原始结果与下载收据已保存，尚未回收本轮下载objects。

审计 `paper_results/ieee_tc/baseline_audit/d422_7b_compatibility/audit.json` SHA
`43f07985abd12189288597587fc0128939c779e1dad65f6875afbdfe82b81517`；
133文件/67717384字节证据清单SHA
`42f06cbed073ec760b056db59fef4807e57187202f086c78c0db51f5d108abc8`。
离线前100关联脚本初次把到达字段误认为在contract中，失败且未生成输出；改为复用
现成FrozenReplayPlan的源序号后通过。未改变在线运行/原始数据，没有性能重跑。

本次结果证明原1,000请求兼容性和该短视图的99.1%联合达标，不是完整W0排名、
预算参考或跨场景最优。下一步是同原生配置的一次完整4,000 W0，封存后按完整
输入/生成/初态/资源合同与 PrimeLoRA、ServerlessLLM 已有点关联比较。无新参数搜索。

## 备份与缓存回收

主仓库封存 `730ce7cc0c6f47064f7ac8dcf9e160df8037c894`、baseline封存
`e646f9106c8fd6fd15f54b69041bd54234fa47e1` 已推送并核对远端。第一次baseline
推送遇到TLS中断，有限重试成功，没有关闭证书校验或强制推送。
133项证据的字节和SHA逐项验证通过。

随后只回收600个可重建下载对象，实际释放1476624384字节，磁盘可用恢复到
163660283904字节。原始池重新哈希一致、不同inode、元数据副本、挂载/软链接、
可读同UID进程引用与后续配置引用均核验；三个不可读/proc条目如实记录，不宣称
全系统引用检查。原始测量、收据、唯一工件和远端缓存不删除。回执见raw及
`paper_results/ieee_tc/storage/20261011_d422_download_cache.json`；原133文件清单不改写。
没有新GPU实验启动。D423完整W0待预登记与新资源检查，不再运行短资格重复。
