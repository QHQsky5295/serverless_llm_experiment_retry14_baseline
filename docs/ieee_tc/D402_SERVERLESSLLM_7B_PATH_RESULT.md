# D402：ServerlessLLM 7B 真实远端路径验证与扩容队列问题

2026-10-11。这是一次 100 请求开发前缀资格运行，不是完整主比较。
已完成运行、释放资源、原生计量重算、双端传输关联和源码故障复现。
结论是**交付与计量路径通过本次覆盖；整体基线仍不能冻结为性能比较点**。

## 1. 范围和结果

复用既有 seed42 的 4,000 请求 W0 中前 100 请求、既有 500-ID 静态清单、
7B checkpoint、官方来源 `9f50241baa5386e06a9321c51f19a9ef5f964c2b`
及历史兼容的 vLLM 0.10.2 环境。无新权重、负载或整池副本。
运行来源：主库 `348c5f3e750fd78a2f877c25bd4c06d85b4806bc`，
基线库 `d0fea52483f2906a5e68a1ec96aed42135498c67`。

min=1/max=4/target=2/keep_alive=10、max_num_seqs=4 是历史机械诊断配置，
尚未选为最优配置。采用已经修复的 ready-first 轮询，保留原生调度、
扩缩容和 backbone fast loader；LoRA 使用已披露的服务内按需 PEFT 交付。

| 项目 | 本轮观测 |
|---|---:|
| 计划／到达／提交／终态／原生合同通过 | 100／100／100／100／100 |
| 失败请求 | 0 |
| 实际原生输出 tokens | 17,369 |
| 本前缀出现的逻辑 adapter | 29 |
| TTFT 均值／Type-1 P95 | 138,039.350／156,904.021 ms |
| 平均路由队列等待 | 135,378.943 ms |
| 平均 service TTFT | 479.332 ms |
| 平均 E2E | 148,696.380 ms |
| E2E／TPOT／原生时间线重算最大误差 | 0／0／0 ms |
| 有实际请求的 backend-ready 来源回执 | 3 |
| 真实远端 HTTP 传输 | 29 |
| 线上字节／核验物化字节 | 50,650,865／724,665,100 |
| 本地交付缓存命中请求／在线打包 | 71／0 |
| 完整物理 GPU 占用 | 1,544.433269 GPU-s |

物理占用的四个互斥时间窗为：到达前 69.747446、到达窗口 734.031593、
drain 719.655325、cleanup 20.998905 GPU-s；总和与设备区间积分一致。
原生 store 在固定四卡池持有上下文，故逻辑实例缩容不能直接当成物理释放。
这描述本次部署，不声称所有 ServerlessLLM 部署都必须常驻四卡。
100 请求的启动占比与 4,000 请求不同，**不计算相对 D371 的性能改善率**。

## 2. 远端与正确性证据

双端以 transfer ID、adapter ID、archive SHA 和实际字节关联；全部 29 次
传输成功，使用 `prepublished_gzip_v1`，没有临时打包或临时归档。
请求的 receipt SHA、冻结工件内容身份与实际下载文件逐一验证。
远端 transfer clock 与本次认证 health 一致；不相减未经校准的跨机时间。
100 次 resolve 均在被测服务资源域内，3 个 resolver，71 次共享缓存命中。
缓存只保存本轮实际请求工件；不是另建完整工件池。

原生 token 数、prompt token hash、请求／adapter 绑定、前置入口时间线和
ready-time worker/source/containment 回执通过重算。**这不新增独立数值
LoRA 效果证明**；原始 `n_correct=null`、数值资格及正式资格字段不追改。
既有静态池的重复／零权重限制仍需披露；29-ID 前缀不是 500-ID 功能覆盖。

本轮远端 `eno1` 实际为 **1000 Mb/s/full**。没有调整网络、重建发布缓存、
在服务期间清缓存或执行哈希扫描。远端独立磁盘门槛按 16 GiB 保留量和
日志增长计算；推理机 150/100 GiB 及内存限制未改变。
请求触发的读、传输、竞争、校验和物化均保留在时延中，没有事后减时。

## 3. 为什么不能立即拿本轮长等待作为基线结论

观测：router 创建四个实例，但第四个
`10bdd813-56b6-4893-9ceb-515d3617e59c` 停在 allocation；实际请求由其余
三个实例执行。scheduler 的实际 worker 日志在 00:14:56 记录同模型
三个 pending 申请（下标 0、1、2），随后只完成两次 allocation；
第三个仍有一张空闲卡及合法 placement option，却没有 allocation 完成。

实际使用的 `storage_aware_scheduler.py` 来源是本轮 source view 指向的
vendor 目录，SHA `b7779e41f451a3328eb73337533d6d9b117aafe6ee6b78f9b28e9b5f5e2b0567`。
其 `_control_loop` 先枚举队列，再在同一轮以旧下标 `pop(idx)`：

1. 三项队列按 0/1/2 枚举。
2. 删除 0 后，后续项下标发生变化。
3. 删除旧 1 会移除原第三项，却完成原第二项的 future。
4. 删除旧 2 抛出 `IndexError`；最后一个 future 未完成，节点元数据更新也未执行。

复用既有 router 测试的 AST 隔离方法，执行**实际 control-loop 源码**，
只替换 Ray 外部依赖为内存对象。三个申请稳定复现
`IndexError: pop index out of range`；future 状态 `[true,true,false]`，
队列却残留第二个已完成 future，metadata update 次数为 0。
这不是人为放慢基线的实验，也没有修改执行过的源码。

归因边界：该源码错误已复现，且与本轮缺失第四次分配的日志一致；
本轮异常堆栈没有被记录，不能声称直接捕获了 live exception，也不能
把全部 135 秒平均队列等待都归于这一错误。启动积压与 target=2 的容量
仍是后续合理配置验证的因素。

在线核查的[官方调度源码](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/sllm/schedulers/storage_aware_scheduler.py)
也保留了相同的枚举／删除形式（2026-10-11 查阅）。官方算法策略与这个
容器操作错误必须区分；不能把错误造成的少卡使用写成固有架构劣势。
联网内容可能变化，本轮判断以保存的实际执行文件及 SHA 为准。

## 4. 安全、保留与复现

- 服务内存峰值 49,332,969,472 B，swap 峰值 0；本轮 cgroup high/max/oom/oom_kill 均为 0。
- 主机最低可用内存 63,025,352,704 B；最低磁盘可用 163,230,203,904 B。
- 启动、回放、监控返回值均为 0；allocation/release 完整闭合；四卡收尾为 15 MiB、0% 利用率。
- 原生 loader-only 环境 overlay 已恢复，既有六个文件 SHA 与 preimage 一致；新增 loader 文件由恢复流程移除。
- 本轮服务资源域已移除，空辅助域已停止；远端三个自有 unit inactive、MainPID=0，HTTP 端口关闭。
- 147 项旧结果保护检查前后完全相同：唯一不符项仍是此前用户修改的 replay 脚本，本轮没有改动或 stage 它。两份保护回执 SHA 均为 `96df8e345419d26b941f99e19578d8dfe6717fe06b35caa6d42e0dae4885e87d`。
- 第一次 health 命令误用无 NumPy 的系统解释器，在 HTTP/模型运行前失败；复用现有环境后通过。没有重跑模型请求来隐藏失败。

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d402_20261011`。
已额外保存 resolver journals/receipts、Ray 小日志和执行 scheduler 源文件，
不复制已下载权重。当前私有物化缓存暂保留供来源检查；不能删除原始记录。

分析表：`paper_results/ieee_tc/baseline_audit/d402_7b_path/audit.json`，
SHA `b1102a8b839ac198d512381ac6b0709fb7ab1bedebe6d38143aa4e1b05275ba8`。
分析配方 `scripts/ieee_tc/audit_d402_serverlessllm_path.py` 直接复用既有
`audit_tc_http_journal`、`validate_http_observation`、`audit_backend_receipts`
和内容指纹函数，不另建回放框架；输出使用独占文件，拒绝覆盖。
其一次 4-GiB/零-swap CPU 审计成功，没有新 GPU 性能运行或图表。

## 5. 主线下一步

先以稳定申请身份修正已确认的队列删除错误，保持原 placement score、
顺序、扩缩容、加载与迁移政策；在独占新 source view 中公开补丁和 SHA，
不改 vendor／历史 view。验证多申请、多个模型、暂不可分配、取消及后来
追加申请等边界，再仅验证受影响的 7B 扩容路径。不是重复 D72/D73 轮询对照。
随后核对既有 500-ID／1,000 请求资格的可复用范围，合理选择公开配置并冻结，
进入一次匹配的完整 W0 回放。不得把 target=2 或缺第四实例的点直接定为基线最优。

PrimeLoRA D371 W0 仍封存不重跑；完整 7B 基线比较、有效 Resident 参考、
W1/W2、3B、核心消融与敏感性仍未完成。当前不画图、不宣称 G1/G2 最优。
