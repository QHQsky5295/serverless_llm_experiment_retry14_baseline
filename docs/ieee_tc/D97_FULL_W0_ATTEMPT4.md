# D97：3B Full4000 W0 第四次完整回放——失败诊断

日期：2026-09-28。执行代码 `d6733aa8607e1fc4f66be3b261f555253c1a1f87`。
这是开发期资格诊断，不是 M1/M2 正式结果，不支持最优性或统计优越性主张。
沿用相同 D88/D89 模型配置、原 4000 请求/500-adapter trace、60 秒通知、真实
远端已发布交付协议和从计划到达计起的 1800 秒期限；本轮未修改这些条件。

## 已完成的结果与资源核验

| 项目 | 观测结果 | 判断 |
|---|---:|---|
| 计划/提交/终态请求 | 4000 / 4000 / 4000 | 完整回放，无全局中断补齐 |
| 成功且原生生成合同匹配 | 1609，40.225% | 未达到全部成功要求 |
| TimeoutError | 2243 | 保留为失败，不作为单纯 SLO 违约 |
| 其他返回失败 | 148 个 RuntimeError | 完整提取已分类，见下文 |
| 实际 GPU allocation/release | 124 / 124 | 无未释放 lease；退出后原生 GPU census 清空 |
| 物理 GPU 生命周期 | 22228.467894 GPU-s | 完整失败运行的消耗，不进入达标资源排名 |
| 资源监控样本 | 6008 | high/max/OOM/OOM-kill/swap 全零，无保护告警 |
| 服务内存峰值 | 27793170432 bytes | 主机最低可用内存 91341680640 bytes |
| 远端传输 UUID 对应 | 133 / 133 | 两端工件身份和线上字节匹配 |
| 线上字节 | 306356296 bytes | 包含未发布尝试，不删其传输消耗 |
| 本地发布 | 131 成功、2 未发布 | 不把未发布记录当内容验证成功 |
| 已验证发布的逻辑字节 | 5105585540 bytes | 仅对内容验证成功记录求和 |
| 请求中打包/临时归档 | 0 / 0 | 使用共同已发布只读交付缓存 |

四个互斥 GPU-s 窗口：准备期 47.883149、到达期 15476.592196、drain
6619.645898、终态后释放 84.346650；未舍入值之和等于总量。
launcher pass=true 仅表示执行与清理通过，不表示全部请求正确或 SLO 达标。
数值 LoRA 可区分性限制仍存在，native-contract-matched 不是数值证明。

## D97 恢复机制的实际证据与边界

- 原始四个 runtime 共完成 1136 条，后续 90 个有成功结果的替代 runtime
  完成 473 条。该分解来自逐请求终态及已知原始 runtime ID，不根据 dashboard
  的实例数猜测是否执行过。
- 122 次 quarantine 事件均记录 released；第一条的隔离、drain、release 时刻
  分别为 62632.111142、62637.203418、62671.998853（本机 monotonic 秒）。
- 124 次物理分配均已释放；替代服务确实发生，不能再描述为“隔离后永远没有
  可服务副本”。但反复隔离/新建仍很频繁，且总体完成率仅 40.225%。
- 这些证据不证明早期队列或原生 RPC 失效原因已解决，也不以单次跨版本差值
  宣称性能改进。下一步应定位最早的冲突/长等待/取消链条，不堆叠恢复补丁。

已记录 source 观测计数：requests=72139、collections=36091、RPC invocations=4270、
joined=36048、stale rejections=2097、membership rejections=226。
collection 不是 native RPC 次数；这些总数尚不能确定单请求的因果阻塞链。

## 完整请求提取后的补充诊断

全部 4000 个请求 ID 已与外置回放、终态账本核对；1609 条成功请求的原生数量、
prompt SHA 和 token IDs SHA 均对应。148 条返回错误中，147 条直接报告原生 RPC
所有权未决、拒绝新生成，1 条通过子进程错误包装报告相同原因。148 条均记录
本次 generation 为 not_submitted。这不意味着其他尚未确认的 RPC 已经结束，
也不能从中直接推断 GPU 释放；实际释放由独立物理记录证明。

最早超时是 req_00513，观察时刻 62193.917817；最早返回上述错误是 req_01159，
观察时刻 62708.447460（同一本机 monotonic 时钟）。超时早于这一返回错误，
不能只处理其后的隔离恢复而忽略前面的长等待。首个超时缺少 source admission
记录，不据此断言从未 dispatch。

以下仅为 **1609 条成功请求的条件诊断**，不代表完整负载性能：

| 分项 | 平均值 |
|---|---:|
| 用户 TTFT | 1084.327598 s |
| dispatch/admission wait | 1081.057257 s |
| service TTFT | 3.270341 s |
| 原生后端 TTFT | 0.198014 s |
| TPOT | 20.852019 ms |

前两阶段满足用户 TTFT = dispatch/admission wait + service TTFT。
原生 TTFT 是 service TTFT 的组成，不再次加和。主导现象在进入服务前的等待；
不能用这些数据支持“GPU 本身每次需要上千秒才生成首 token”。具体等待原因仍
需因果测试。927 条成功请求有已保留的 source 重选尝试，共 11727 次、最大 112；
它们不是所有失败请求或完整 snapshot 拒绝次数。

零 ready 且有队列的控制观测有 457 次：284 次无可用设备、74 次安排 activation、
99 次 no_action。D97 已不再是始终无法恢复的死局，但这种恢复不足以解决长队列。
122 次隔离至释放平均 15.947154 s，P95 28.610816 s；不把这些恢复耗时从用户
等待或生命周期资源中扣除。

## 两次未发布的传输

| 工件 | HTTP transfer ID | 线上字节 | 已知状态 |
|---|---|---:|---|
| translate_lora_0114 | 87ef3eee2235493d81001848b471fa14 | 0 | 未验证/未发布，没有 verified payload 字段 |
| ecommerce_lora_0129 | 47e092e3a1be4854a758da06fed11a3d | 2313559 | 归档已验证，但未完成 payload 验证/发布 |

两条均无请求打包。尚未根据准备/取消时间线确定原因，不预先称为网络故障或
正常取消。第一次收集脚本因沿用“全部传输均成功发布”的假设拒绝输出，第二次
因未发布记录缺 verified payload 字段拒绝输出；诊断记录保留，最终统计显式区分
发布状态和字段缺失，不补零、不放宽成功传输的验证。

## 来源与下一步

原始目录：`results/ieee_tc/p2_backend_qualification/d97_20260928/`。
小型结果：`full_attempt4_preliminary.json`；检查脚本：`collect_full4_preliminary.py`。
保留 request_terminals、physical summary、main_outcome、launcher receipt、
watchdog、远端 journal/监控及本地/远端清理回执。远端两服务与本轮监控均在推理
退出后核验原 invocation/PID 再停止；未停止无关进程，未修改远端配置。

完整输出 JSON 为 3655870499 bytes，禁止直接整体 json.load。下一步复用 D96
既有 jq streaming projection，在 4 GiB 独立 CPU 资源域提取所有 4000 请求及
保留的失败观测，再分类、核查时间线、补齐来源 SHA 和 curated 数据、提交备份。
此后才进行一个可证伪的因果优化。7B、基线、warm SLO/Resident、M1/M2/A/S
均不在本轮被当作完成；不重复已完成的缓存发布/全池下载/初始化 profile。

09:44 更新：流式提取 exit0，耗时 743.62 s，峰值 RSS 72192 KiB；汇总校验通过。
curated 数据为 `paper_results/ieee_tc/p2_backend/20260928_d97_3b_full_w0_attempt4.json`，
SHA256 `9167a9a706b6b24c14b1b7fb6ede4c93fa0ca601e6182db55d53701786ac11fd`，含 27 个
源文件 SHA；旧投稿保护清单 147 项零变化。逐失败分类另见同目录
`20260928_d97_3b_full_w0_failure_breakdown.json`，其 projection SHA 与主汇总一致。
下一步备份本轮证据，随后开展长等待/source 冲突的因果测试，不再重复本轮分析。
