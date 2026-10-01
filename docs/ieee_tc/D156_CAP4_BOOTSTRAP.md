# D156：7B 四并发候选的原生资格检查

阶段 P2/P3；开发资格，不是 Full、共同 SLO 或 G1/G2 结果。
遵循 D155 已登记的单一容量假设；无服务代码修改。
唯一模型差异为 max_num_seqs/runtime_concurrency_cap 从 2 到 4。
其余参数、原工件、原始请求、输出合同、物理保护不变。

复用 D84 六个既有代表请求，GPU-ready 四并发三波：
一波 kernel warmup 保留、两波代表测量。共 12 次观测，不新建 trace。
先测 native 实际重叠、当前 KV、原生数量和所有权释放；完成长度
仅用于新配置自己的初始化，不重标 D89 两并发延迟。

## 启动记录（全部保留）

| 记录 | 状态 | 解释 |
|---|---|---|
| 输入核查 prepare1 | PASS | 仅两容量字段变化；147 保护项无变化；12 个索引合法 |
| Remote health1 | 连接拒绝 | 服务启动命令返回早于监听就绪；没有推理运行 |
| Remote health2 | PASS | 两监听已就绪、认证及已发布交付协议一致；没有重启 |
| bootstrap1 | 加载前拒绝 | auxiliary 标识 31 位，不满足既有 32 位十六进制格式；无 GPU 加载 |
| bootstrap2 | PASS | 12/12 原生数量及身份计量通过；三波真实解码重叠；已释放 |

本状态表遵循 Plan §11 / academic-plotting 的资格表规则，不为启动错误画性能曲线。
远端沿用一次性只读交付缓存；请求路径不打包。两端实测 1000/full。
尚未生成新的时间 profile；数值 adapter、共同参考和整系统目标仍待完成。

## 结果与边界

2026-10-02 05:02–05:04 +08，同一 3090 / TP=1 / FP16 / vLLM 0.30.0。
原始文件 SHA：`befd3161e9ef2cd81b2271ba08777dd61fc623219a4babfeec5f34b6773f84ac`。
初始空闲 KV 为 **278 blocks**，每 block 16 tokens、8 MiB；
四条声明最大上下文所需上界为 4×ceil(1024/16)=256 blocks。
278 不是旧配置的 304；新配置确实重新读取了当前后端容量。
该条件不是连续无 preemption、完整物理 admission 或整轮安全证明。

| 波次 | 用途 | 原生成功 | 四条解码共同重叠 / s | 原生 TTFT 均值 / ms | TPOT 均值 / ms |
|---|---|---:|---:|---:|---:|
| 0 | 保留的 warmup | 4/4 | 5.0070 | 282.740 | 34.138 |
| 1 | 代表测量 | 4/4 | 5.0506 | 250.893 | 34.475 |
| 2 | 代表测量 | 4/4 | 2.8809 | 233.430 | 33.322 |

重叠定义为 min(last-token)−max(first-token)，使用同一 native 时钟。
四条均已产生首 token 且均未结束，不能只用 dispatch 或 task 数证明并发。
这是请求解码区间重叠，不冒称连续 GPU kernel 利用率。
表中原生 TTFT 不含完整开放回放的排队，不与 D154 用户 TTFT 作百分比比较。
单次资格、小量代表请求，不给 CI、不宣称吞吐最优。

全部请求原生输出等于目标、终态事件及释放确认完整；清理后 native sources
与 admitted 为空，临时工作目录移除，物理 allocation/release 完整。
本次总占用 **114.0386 GPU-s**（含启动、准备、warmup、清理），不从测量中扣除；
这不是主工作负载 GPU-s/request，`n_correct=null` 直到数值 adapter 验证完成。

远端 12 次 UUID 配对下载，共 65,204,801 线上字节，内容/归档 SHA 对齐，
请求内打包 0 次；远端停止后复制并逐 SHA 核对日志。
服务峰值内存 5,416,239,104 bytes，监测到的 high/max/OOM/swap 均 0。
观察为零不等于为所有未来并发/输入证明无 OOM。

真实服务 scope InvocationID `aea5686a4ac844bb9c5595b936894bfe`，
auxiliary `5854b37b19e942a4971b1e142ca26cff`；两者自动移除，原生进程已退出。
远端三项 exact-identity stop 均 success，不存在未停 GPU/远端任务。
分析 scope `d63097717f94459fab73b420a9a1be56` 同样已退出。
第一次远端日志定位因远端没有 rg 返回 127；停止操作已成功，随后用 grep 找到
同一 clock 对应的唯一日志，未重启服务或重新生成缓存。

## 决定与下一动作

**接受为容量候选的最小原生资格证据，不是接受为整系统最优配置。**
已建立新的 7B completed-length audit，仍由既有
`measured_admission_initializer` 精确验证 model/backend/trace/native 样本，
两输入桶均有实际完成记录；旧 cap=2 profile 未改写、未挪用。

下一步只补该配置的 admission-enabled 分层测量，按原三轮规则覆盖
Remote/NVMe/file-HOST/native-HOST/GPU、原内容/rank 和 admitted 类域 [1,2,4]。
控制公式不变；只更新因容量变化失效的实测初始化。全域覆盖后才能 Full4000。
不在此另加 CPU 优化、不增加 deadline、不扩大到 cap8、不先做 3B。
共同 warm/reference、数值正确性与新旧 Prime 的 G1/G2 验收仍未完成。

新增表：
`paper_results/ieee_tc/p2_backend/20261002_d156_7b_cap4_bootstrap.csv`。
结果和完成长度 audit 同目录以 D156 命名；原始结果保存在新 d156_20261002
目录，不覆盖任何历史结果。没有服务算法源代码修改。
