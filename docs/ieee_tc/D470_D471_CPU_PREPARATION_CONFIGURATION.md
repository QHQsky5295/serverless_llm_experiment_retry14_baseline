# D470 资格失败与 D471 配置归属修正

2026-10-12。D469 接入已通过 570 项 CPU 检查和 35 项发布检查，并推送
`68aef590521a3bc3acb0e33a15554871f0715554`。它尚未通过实际模型资格，
不能称为性能优化已经验收。

## D470：保留的实际失败

原 runner 的原 100 请求资格启动在 `ScenarioRunner` 加载既有 service profile
时拒绝：`service profile model/configuration/context mismatch`。
service 退出码 1，watchdog 退出码 0；没有启动 CPU 准备池、GPU allocation
或推理模型。现有保护成功阻止了配置身份不匹配继续执行。

失败不是一个“低延迟/低成本”的性能点，也不是 CPU 准备效果被否定。
source、launch、服务日志和远端监控全部保留。实际 GPU context 已清空、服务
路径移除；启动/监督 scope 按原 InvocationID/inode/实际空进程列表关闭。
随后只停止本轮捕获身份的两个 artifact 服务及监控，确认均 inactive、端口
关闭。远端只读交付物未变化，未改动 Kubernetes。启动前网络为 1000Mb/s/full。

## 原因与修正依据

`FrozenServiceProfiles.model_identity` 对实际模型、LoRA、scheduler 和 ownership
配置严格绑定。D469 将 `ieee_cpu_frontend_policy` 放入 model 字段，尽管它只
决定 CPU 前端何时创建，却改变了模型配置身份。CPU fixture 验证了实际 acquire/
bind/activation 顺序，但没有用这个新增配置走实际 profile loader，因此漏掉
了这一接口不匹配。

修正采用配置职责分离，而不是删除 profile 校验：

- 前端提前准备属于 **resource_coordination 的生命周期策略**；移动到该层。
- 真实模型参数仍逐字段等于 D464/D465；service/preparation profile 原文件、
  SHA、原始测量、均值和严格 loader 均不改变，不重新打标签为新测量。
- 原场景工厂从控制配置取得 opt-in，显式传给准备生产者。缺池、不支持的
  policy、in-process 路径及将 policy 错放进模型配置均拒绝，不静默回退。
- worker 仍以原模型/成本配置 SHA 校验；准备 contract、UUID、birth 和物理
  lease 分别保留。运行执行键和控制配置包含新策略，不从 provenance 中隐去。
- 新增使用实际 D471 YAML 和原 profile loader 的回归，逐字段核对 native
  配置未变；D470 错误配置与失败证据保留，不覆盖后重试。

这不保证服务性能不变：同时进行的 CPU 准备可能影响服务竞争和在线测量。
既有 profile 是同一 native 推理路径的历史初始化值，运行时继续按论文的
观测更新。端到端收益仍由新的完整回放检验，不能由配置等价推定。

## 顺序和边界

D471 首轮回归 571 项中 569 通过、2 项 fixture 错误：新 profile 测试未保存
第一步 service profile，以及既有 no-handoff fixture 未初始化 coord_cfg。
两者按真实构造顺序补齐，未改变生产实现或放松校验；第二轮 571 项全部通过，
无失败、错误或跳过。两轮原始日志与源码快照均保留。CPU 峰值分别为
2456068096/2473689088 B，high/max/OOM/swap 均为零，实际空进程列表与
捕获 scope 身份核对后关闭。147 项历史保护检查通过。

上述仅为正确性回归，不代表实际 native 模型或性能资格通过。

D471 是新身份的原 100 请求资格，仍非正式性能点。通过后才进入一次新键
4000 W0。SLO、九个公式、arrival/token/adapters、CPU/GPU/内存边界、远端
交付和 D463 父版本均不变。已封存基线不重跑、不优化，不画图。
7B G1/G2/W1/W2、HydraServe 原生隔离资格、3B 和全部消融/敏感性仍未完成。
