# D469：将 CPU 提前准备接入原有扩容生命周期

2026-10-12。父版本 `4e8cee275713d0afbdf84051252bcfd4cd699369`。
当前主线仍是 7B PrimeLoRA；这不是新模型启动框架，也不是性能验收结果。

## 可证伪假设与既有证据

D465 完整 W0 的联合达标率为 94.75%，低于共同的 95% 门槛。D466 已复用
原始时间线确认部分积压发生在实际服务容量不足期间；不能简化为一个漏唤醒。
D467 已验证固定实际 UUID 的 CPU 前端导入边界没有创建 CUDA context，
D468 已验证单次激活许可和物理 acquire/bind 的身份合同。

本次假设不变：在部署通知之后、服务的共同 CPU/内存预算内预先导入前端，
能与既有服务重叠，从而减少后续扩容的串行等待。额外 CPU 竞争、内存常驻和
native core 内的初始化可能抵消收益。必须由新执行键的完整 4000 请求判断，
不能从旧延迟中扣除导入耗时。保留 D463 为父版本的一部分，**未接受其性能收益**；
本次不捆绑回滚、并发/阈值/缓存容量变化或后台预构建开关。

已再次在线核查官方
[vLLM 0.30 core 进程管理](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/utils.py)
及 [CUDA platform](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/platforms/cuda.py)。
前端与 native core 分离，并不消除 core 自身的启动成本；设备能力缓存意味着
不能在导入后随意换 GPU mask。网页接口返回 429 后，使用公开官方 raw URL，
不使用二手资料或绕过权限。源码内容及 SHA 随本次证据保存。

## 实际接入

- 原 `SubprocessInferenceEngineProxy` 共用 `_worker_inputs`：模型配置解析、
  allocator、路径、日志和环境生成只有一处。准备路径启动同一 worker 脚本的
  fresh interpreter；固定 UUID 在导入前生效，仅传入指定 socket FD。
- 场景拥有 `DedicatedScenarioEngineFactory`。通过原 Full 资格核查之后，才在
  共同部署通知时间边界内启动 CPU 池；数量受既有合法设备和 max_instances
  限制。CPU-prepared 不进入 instance pool，不构成 ready/路由/admission 容量。
- 正常扩容通过原 `_spawn_dedicated_scenario_engine`，领取对应设备准备进程。
  原 allocator 先取得排他物理 allocation，再 bind 真实早期进程身份并发送
  单次 activation capability。**不等 CPU-ready 后再补写 acquire**；取得 GPU
  之后仍在导入的等待照常计费。native core 仍使用 spawn。
- worker 分开保存 `cpu_prepared.json` 和 `cpu_activated.json`，原 `ready.json`
  仍只表示真正完成引擎初始化。配置/设备/birth/lease 不一致即拒绝，不启动
  第二个 fresh worker 做隐藏重试。
- 在线缩容使用原 native 退出和物理归还证据，辅助进程退出后才补充该设备的
  CPU 准备进程。最终关闭先同步禁止补充，再分别 join CPU 和原 runtime。
  runtime 收尾失败或取消也不能跳过 CPU 收尾。
- Popen 返回后立即登记 handle，之后出生验证失败也保留归属；关闭许可阻止
  未激活进程进入 initialize。CPU helper 退出核查按原 session/group、UID、
  birth、服务子树和 pidfd；无法证明的退出保持失败，由共同安全域保留，不
  宣称资源已释放、不杀未知进程。D467 的 Humming reaper 是线程，不能误称
  已观测到额外子进程；这里覆盖的是潜在辅助子进程的退出边界。
- 最终 outcome 新增准备记录（路径、出生、准备/许可证据、退出状态），用于
  统计准备数量、使用比例和资源代价。物理积分仍由原 acquire/release 决定。

默认无 `ieee_cpu_frontend_policy` 的旧配置不启用准备池。D470 的原 100 请求
资格配置仅增加 `fixed_uuid_cpu_frontend_v1` 和独占输出/cache 路径；其余参数
沿用 D464/D465。原 trace、工件、token 合同、60 秒通知、SLO 和九个公式不变。

## 验证状态和下一步

首轮 567 项 CPU 测试，566 通过、1 错误：旧模拟进程的 wait 返回后 poll 仍
为存活，新增真实退出核验正确拒绝了该 fixture。保留原日志/源码快照；只修正
fixture 的 wait→poll 状态，不放宽生产退出检查。随后补充 runtime 失败仍
join CPU、取消等待和物理归还先于替换的检查；结果另存，不覆盖首轮。
第二轮 570 项全部通过，0 失败/错误/跳过；两轮集合重叠，不累计为独立实验。

CPU fixture 不是实际 CUDA、LoRA 数值或性能资格。后续顺序为：最终回归与
历史保护检查→提交备份→原 runner 的 D470 native100→独立新键完整 W0。
必须检查准备进程在激活前无 GPU context/ready 容量、实际 UUID 和源模型
合同、全部 token、完整生命周期和 CPU/helper 退出；任一缺口不能放行完整回放。
短回放不外推 GPU-s 节省或宣告 G1/G2 成功。

7B W1/W2/G2、HydraServe 独立原生资格、3B、A1–A5、S1–S13、离线成本与
审稿文档全部保留。已封存 baseline 不重跑、不调优；现有非本项目 k3s 不改动。
