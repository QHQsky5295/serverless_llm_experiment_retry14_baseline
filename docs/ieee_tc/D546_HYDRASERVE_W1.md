# D546：HydraServe 7B W1，复用已合格原生实现

2026-10-15。当前为启动前登记，不是性能结果。接续已封存的 D533、D536、
D537、D542、D544，补齐六系统 W1 比较的最后一项。完整 PLAN 的 W2、G2、
3B、消融、机制与敏感性均继续保留。

## 问题和固定条件

采用同一 4,000 请求、500 逻辑 adapter 集合、原 prompt 与目标 token：仅按
既定 W1 将到达序列分成八段，每段 500 请求、段内时间减半、段间 30 秒。
不等待 drain，不人为强制八次扩容，不重建工件或负载。

- source trace SHA：`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。
- W1 view SHA：`103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3`。
- 458,224 目标输出 token，实际请求涉及 132 个 adapter ID；500 ID 不表示
  500 种独立训练权重，原有 498 零权重／2 非零的身份限制继续披露。
- `fixed_length_greedy_v1`、共同业务前 60 秒部署通知、共同 1,800 秒请求保护。
- D394 长度分组 TTFT/TPOT 阈值及冻结 V1／单轮 V2 不变；全部实际点均保留。
  不要求对手必须达到 95%，不把未达标点当达标资源排名。

## 原生实现复用证据

D510 已完成 W0，不重复基础资格或 W0。逐文件核对结果：15 个 HydraServe
runtime 文件字节相同，重新渲染的 16 份 overlay SHA 全部相同。继续使用官方
修改的 vLLM 0.4.2、原生 Scheduler/placement、30 秒 idle retirement、启动重叠、
pipeline 与 consolidation 路径；不修改策略、不搜索性能参数。

原生 Model.batch=8，`max_loras=batch*PP_SIZE`；model registration 保持
`(13,9000,200,1000000,1)`，USE_CACHE=0、SLOW_EXPR=0。
仅复用 D508/D509 已验证的必要取消所有权与 LoRA 容量接入修复。

与 D510 相比，共享源码仅两处变化：D531 的完整 W1 到达视图支持，以及 D541
将内核退出通知从 select 改为 poll 的高文件描述符计量修复。其余原生 HTTP、
计量、remote resolver 方法保持不变。本次 16 项 CPU 测试通过（2.089 秒），
覆盖 W1、原生 HTTP/token/取消以及物理生命周期连接；不是新的 GPU 性能实验。
旧独立 D541 测试与 D510 GPU 资格复用，不为凑次数重新运行。

第一次只读检查因两个仓库的 `scripts` 导入命名冲突，在读取身份之前退出；
旧检查源码、失败记录及资源关闭回执保留。只调整检查器导入顺序，未改变服务。
成功检查与准备资源域均已实际清空并关闭；无 OOM、swap 或 high 事件。

## 隔离、远端与计量

现有 k3s 属于其他项目，**不访问其 kubeconfig、不改集群或权限**。继续使用
D510 的 rootless Podman 隔离路径、私有 loopback/backbone provider 和只读环境。
单推理节点、四张 RTX 3090；不冒称多计算节点集群。

服务合计 high=72/max=80 GiB、swap=2 GiB、20 个物理核；回放与监控在独立
high=3/max=4 GiB、swap=0 的资源域。记录实际容器 worker，而非只限制 CLI。
启动前再次通过 150 GiB 磁盘及剩余增长内存门槛；运行期间不做压缩、清理、
测试、构建、备份或远端维护。

只复用远端 D78/D80 已发布 gzip 工件。首次获取、必要读取、传输、竞争和
本地物化计入时延；请求路径不打包，不事后减去任何阶段。每轮新冷缓存，
允许 HydraServe 原生请求驱动获取和合法复用。远端服务自动管理，无需用户操作。

完整结果须校验全部请求/原生 token/adapter 绑定、真实 worker 持卡区间并集、
实际释放、两端 transfer join、内存事件与同输入连接。若出现多 rank，必须
补齐实际多 rank 资源边界核查，不能套用 pp1 审计或伪造 pipeline 证据。
异常低性能先查适配缺陷；没有证据不任意调参或重跑同键。

## 结论边界和交付

D510 的 34 个 worker 全为 pp1，不能宣称已证明 pipeline/consolidation 效果；
D546 以实际触发记录为准。独立数值 LoRA 正确性仍未知，`n_correct=null`，
不把 HTTP 成功、固定 token 数或名称绑定当作独立数值证明。单轮已知回放保留
开发身份，不产生 CI 或显著性主张。当前没有 D546 性能数字。

原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d546_20261015`。
配置及小型来源快照备份到 `paper_results/ieee_tc/main_comparison/d546_hydraserve_w1/`。
沿用旧 runner、资源保护、收尾和审计入口；原结果、旧图、用户三个修改文件零覆盖。
