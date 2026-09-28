# D101：3B Full4000 W0 第八次完整回放——仍未通过资格

日期：2026-09-28。执行代码 `2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d`。
这是开发期诊断，不是 M1/M2 或基线比较。沿用 D100 的配置，只有三个本轮
独占输出/缓存路径不同；运行代码的目标变化是文件容量计算的临时索引，见
[D101 CPU 验证](D101_FILE_CAPACITY_INDEX.md)。九个公式、输入、profile、生成
合同、1800 秒期限和真实远端交付方式未改。未延长期限或按胜负提前停止。

## 完整执行与资源状态表

| 项目 | 观测值 | 解释 |
|---|---:|---|
| 计划 / 提交 / 终态请求 | 4000 / 4000 / 4000 | 回放已完成，不代表全部成功 |
| 成功且原生生成合同匹配 | 3970，99.25% | 尚未达到全部正确执行要求 |
| TimeoutError / 其他返回失败 | 30 / 0 | 全部保留，不转为单纯 SLO 违约 |
| 激活 | 1 initial + 3 natural_scaleout | 不能统称四个 initial |
| 副本隔离 / 替换副本成功请求 | 0 / 0 | 所有成功均来自原四个 runtime |
| 物理 GPU 租约 / 已释放 | 4 / 4 | 无未释放租约，GPU 计算进程已清空 |
| 生命周期 GPU 占用 | 19874.486281 GPU-s | 失败运行的完整消耗，不进入达标排名 |
| 资源样本数 | 5593 | high/max/OOM/OOM-kill/swap 均零 |
| 服务内存峰值 | 37106450432 bytes | 主机最低可用 81964531712 bytes |
| 远端 UUID 配对 / 内容验证发布 | 132 / 132 | 全部已发布，无未发布获取 |
| 客户端接收 / 服务端 socket 写入 | 306360162 / 306360162 bytes | 配对一致，不冒称纯 NIC 字节 |
| 内容核验的逻辑字节 | 5139892912 bytes | 与线上压缩交付字节分开 |
| 请求中打包 / 临时归档 | 0 / 0 | 共同只读缓存已复用，未重建工件池 |

互斥 GPU-s 窗口：准备 48.358571、到达 15755.358408、drain 3976.913227、
终态至释放 93.856075；未舍入值之和等于总量。GPU 释放后的多分钟 JSON
序列化不再计为持卡，但释放前的 drain 和清理不扣除。以物理账本为准，不用
旧终端 CE、InfraGPU、MaxRep、5000ms SLO 或 loaded 计数替代冻结指标。

## 成功请求的条件时延

全部 4000 个请求 ID 与提交/终态集合一致，3970 条成功的 token 数、prompt SHA
和 token-ID SHA 与原生终态逐条对应。下表只描述这 **3970 条成功请求**；未给
失败补造 TTFT，也不以成功子集代表完整工作负载。分位数为 Type-1，单位秒。

| 阶段 / 指标 | 平均值 | P95 |
|---|---:|---:|
| 用户 TTFT | 734.065505 | 1039.870272 |
| 用户 E2E | 740.126818 | 1046.199589 |
| dispatch/admission 总等待 | 721.191880 | 1024.875461 |
| 其中 dispatch window 等待 | 703.630776 | 1010.261300 |
| 其中 runtime slot 等待 | 15.681931 | 40.459853 |
| 其中计划到达释放迟到 | 1.879173 | 7.572931 |
| service TTFT | 12.873625 | 66.111115 |
| 其中 native vLLM TTFT | 0.484660 | 1.618221 |
| 其中 native 前 service shell | 12.388965 | 65.011780 |
| LoRA I/O 观测 span | 11.117817 | 63.069132 |
| parent RPC overhead | 3.047614 | 13.131296 |
| TPOT | 0.031081 | 0.072670 |

dispatch 的三个分项均值可相加；service shell 与 native TTFT 均值可相加。
P95 不相加；其他嵌套或重叠 span 不重复累计。两个 parent pickup/resume delay
字段仍无请求级样本，保留 n=0/null，不用旧摘要填充。

证据支持“推理前推进/等待是当前主要端到端瓶颈”，尚不能仅凭阶段名称判定
具体锁、状态转换、资源检查或 CPU 函数的因果份额。原生推理较快不证明其
所有 batching/TPOT 已最优；当前长等待也不能作为 Prime 对基线优越的证据。

## 失败、状态观测和归因边界

全部 30 条失败是 TimeoutError。首条为 `req_01685`，controller 观测于业务
开始后 3478.693559 秒，距计划到达 1800.231357 秒；这不是该失败请求的 TTFT。
30 条失败记录的 generation submission / source admission 均未保留，不能把
`instance_id=null` 或缺字段解释为“从未 dispatch”，也不能声称已定位最后等待
阶段。失败细表保留所有 request ID、观测时间和现有证据，不改变样本定义。

source 观测 requests=10952、collections=1049、joined=9903、RPC invocations=4115、
membership rejections=4、stale rejections=6727。6727 是观测拒绝计数，不是
失败请求数或独立样本数。实现中该计数涉及原生 source epoch/采集时间顺序，
不是 TTL、物理内存不足或 admission 容量计数。

residency completed/superseded/cancelled=250/255/1；file preparation=254/255/0；
GPU preparation=36/255/0。255 次 supersession 分为 native_registration 251、
native_gpu_source 3、native_file_fallback 1，三个列表是同一规划传播链的记录，
不能加总为 765 次独立 supersession。没有 quarantine 或替换副本，零事件的
隔离持续时间统计为 N/A，不填零时长。

对比 D100 的单轮完成率与稳定性变化只能作为开发观察：两轮成功子集不同，
不能据此给正式配对 CI、宣称显著改善，或把 CPU helper 的加速倍数推广为 E2E
加速。本轮仍失败，warm SLO/Resident 数值参考和 LoRA 数值可区分性也未通过，
不进入任何达标资源/尾延迟排名。

## 后续方向与数据保护

本轮已完成资源释放、终态核对、远端配对、条件时延和失败状态表。24 项运行前
来源检查及 47 项证据 smoke 已通过（1.586 秒），所有完成的分析资源域已按身份
和空进程检查停止。下一步完成 Git 备份，再针对推理前推进选择一个可证伪假设。
只读代码检查发现共享 RPC 的原始视图仍被每个等待者重复转换和 footprint 校验；
这目前只是待测 CPU 工作，未测其实际耗时或超时因果份额，不据此直接改实现。
不得放松 source 新鲜度/owner/物理容量检查，或盲目启动第九次 GPU 回放。

继续遵循“历史日志与源码 → 原始文献/官方实现 → 最小因果验证 → 完整回放”
的主线。7B Full、warm SLO/Resident、暂停的 baseline、M1/M2、A1–A5/S1–S13
仍待完成；不以这些开发诊断替代论文实验矩阵。

原始目录 `results/ieee_tc/p2_backend_qualification/d101_20260928/`。原始完整
JSON 为 9889140123 bytes，使用未修改的 D96 streaming filter 提取：1210.15 秒，
峰值 RSS 137856 KiB，exit=0。提取与统计均在 3/4 GiB、swap=0 的 CPU 资源域中，
未整体载入原始大文件。147 项旧投稿保护条目零变化，原始结果、失败证据均保留。

Curated：`paper_results/ieee_tc/p2_backend/20260928_d101_3b_full_w0_attempt8.json`，
SHA `86b795c8df33bb1d6984e0c25c455f0cf2645228b821382ef6a54f4ef9d2fdb9`。
失败细表：同目录 `20260928_d101_3b_full_w0_failure_breakdown.json`，
SHA `8a604531df3cae5ef416e5b0b5ed182ca2c8c946be947fa062c21dbad2eb9593`。
两份文件均有原始来源/投影 SHA；curated 包含分析脚本、运行配置、清理和源代码
引用。仅统计未成功运行，本节采用状态表，不制作优势图或覆盖历史图。
