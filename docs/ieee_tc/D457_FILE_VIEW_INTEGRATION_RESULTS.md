# D457：文件视图优化的原始 1000 请求集成验证

2026-10-12。一次开发验证，非完整 W0、非独立留出测试，不宣称 G1/G2 达成。
候选代码 ab495b5fce273bbc1b1a4ab6ec53593082a094dc 已在启动前备份。
保留 D371 配置、cap14、共同 60 秒准备、原生生成合同和真实远端交付；
只有既定文件投影优化与本轮独占结果路径不同。未启用帧观察器。

## 观测结果

| 指标 | 本轮观测 |
|---|---:|
| 请求/原生 token/工件/租约绑定合格 | 1000/1000 |
| 原生输出 token 数 | 122510 |
| 共同阈值联合 SLO | 923/1000，92.3% |
| 平均 / P95 TTFT | 1534.405 / 7104.083 ms |
| 平均排队 / service TTFT | 861.210 / 673.195 ms |
| 平均 / P95 TPOT | 34.630 / 61.620 ms |
| 完整物理 GPU 生命周期 | 3886.440341 GPU-s |
| 初始 activation 到 engine-ready | 52.903724 s |
| 真实远端获取 / 线上字节 | 60 / 74953014 B |
| 请求时打包 | 0 |

同 D434 独立输入记录逐请求比较 request、adapter、target、canonical prompt SHA、
prompt token 数、原生 prompt ID SHA 和源记录 SHA，1000 项全部一致。
输出数量和完整原生绑定不是独立 kernel 数值正确性证明；原始 n_correct=null
保持不变。本轮 60 个逻辑 adapter，不写成 500 个实际出现或独立训练 adapter。

dispatch 前 tier：Remote/GPU/HOST/NVMe 为 60/636/108/196。
初始 activation 1 次并 ready；自然触发 9 次，其中 8 次 ready、1 次未 ready。
触发次数不能当作成功扩容或 first-service 数量。未 ready 的资源不从生命周期扣除。

## 如何解释

CPU 组件等价和实际输入/执行合同均通过。短序列联合 SLO 为 92.3%，不宣布达标；
它也不能直接与原完整 4000 请求的 97.70% 比较。D456 是带帧观察器的诊断运行，
不作为本轮普通执行的性能对照，不据两次不同条件的数值计算优化加速。
目前未建立端到端收益或因果幅度；CPU 微测不能代替完整服务结果。

按预登记顺序，下一步是同一候选/配置的一次普通完整 4000 W0，先通过新的
资源及环境检查，再与已封存、同输入合同的完整结果比较。保留所有旧点和退化，
不重复 D457、不改变共同 SLO、不调 baseline。W1/W2/G2、3B 和后续矩阵仍未完成。

## 环境与资源

1198 次独立资源采样均读回 memory.high=72 GiB、max=80 GiB、swap.max=2 GiB；
峰值 20254359552 B，high/max/OOM/swap 均为零，无保护中止。
最少已记录磁盘可用 160028659712 B，仍高于运行中 100 GiB 底线。
启动时通过 150 GiB 门槛；运行结束后的容量不是下一重型任务的自动许可。
实际 GPU context 释放、服务资源域移除后，才停止本轮所属远端服务和监控，
两端 transfer UUID、内容 SHA、归档 SHA 和线上字节匹配，下载日志 SHA 核验通过。

保留启动时两条 systemd inotify-watch ENOSPC 告警。内核说明该错误可能来自
用户监听上限或内核分配失败，而非直接证明磁盘满：
[Linux inotify_add_watch 手册](https://man7.org/linux/man-pages/man2/inotify_add_watch.2.html)。
事后只读检查：max_user_watches=65536；可见同用户描述符的 watch 行数合计也为
65536，其中一个 MainThread 进程为 63568、codex 为 1909、systemd 为 19。
这与监听配额耗尽一致，但共享描述符可能重复计数，事后快照也不是启动时证明。
未停止其他项目/IDE 进程、未修改全局上限或运行中的环境。
独立资源采样、实际约束和显式释放检查通过；告警对 systemd 自动回收及完整环境
等价性的影响未被证明为零。下一次启动前须保留并核查此条件，不能静默略过。

## 分析复用、失败保留与来源

复用 D452 的 analyze_prefix/cross_contract，仅替换本轮路径和标识，逐字校验。
58 项来源引用、35 项启动前资格引用、147 项旧投稿保护及三份用户文件均通过。
后处理脚本和原始日志独占保存，未画图、未改论文、未重建权重/trace/工件池。

保留两个后处理问题：首次 shell 因 qualification.log 已存在而在执行分析前拒绝；
空 qualification.json 和原日志不覆盖，分析只实际执行一次至新文件名。
首次汇编索引了取消 activation 不存在的 engine_ready_at 而失败，未发布结果；
v2 改用已有严格 partition_activation_readiness，对全部真实请求验证 ready 归属，
不补造 ready 时间、不改实验结果、不重跑 GPU。失败脚本和日志完整保留。

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d457_20261012/`

汇总目录：`paper_results/ieee_tc/p2_backend/d457_file_view/`。
主结果 SHA：6f309c118e7b7eca1f17e55a6865c9a37f2fe9249058d0eece1fe7808f7e8f4a。
冻结 V1、单轮 V2、共同 D394 阈值及原始输入身份均不变。
