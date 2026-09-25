# TC 资源保护资格：执行证据与边界

这是实验有效性与主机安全记录，不是性能比较。以批准计划第 2 节为准。
本机 systemd 249；实际委派 memory/pids，CPU 约束已验证的是进程 affinity，
不是声称存在 cpuset controller 隔离。正式服务仍需 72/80 GiB、swap 2 GiB，
外置回放/监控合计不超过 4 GiB。小型验证不能替代实际模型 worker 验证。

## 外置监控测试（2026-09-25）

沿用 `scripts/ieee_tc_preflight.py` 增加 `watchdog` / `watchdog-test`，
没有启动新的实验框架。监控运行在独立辅助 scope，服务运行在另一个 scope。
两组使用不相交的 CPU，128 MiB 硬上限；没有向主机施加内存压力。

| 检查 | 结果 | 证据 |
|---|---|---|
| 16/24 GiB、PSI 连续十次、磁盘/inode、身份/隔离等合同 | 16 项单元测试通过（含原有 8 项） | `tests/test_ieee_tc_preflight.py` |
| 首次独立监控 witness | 未通过：输出末尾 pretty JSON 与 JSONL 解析不匹配；已清理 | `20260925_external_watchdog_attempt1.json`，保留失败 |
| 正常退出 witness | 通过：3 次采样后显式测试触发；服务及子进程释放；未用强杀 | `20260925_external_watchdog_attempt2.json` |
| 忽略 SIGTERM witness | 通过：10 秒宽限后 cgroup 整树强制退出；无残留 PID | `20260925_external_watchdog_stubborn_attempt1.json` |

后二者服务峰值分别为 10,960,896 / 13,705,216 bytes。这个量只是微型测试
本身的内存，不是 Prime/Serverless 内存峰值。测试触发明确标记
`test_only_not_resource_failure`，不是制造一次真实低内存/OOM。

## 所有权与异常归因

- 仅接受本项目 UUID 命名的资源域，核验 InvocationID、目录 inode、UID。
- 不接管全局 Ray 或未知服务；不用全局 `pkill` / `ray stop`。
- 优雅退出先枚举资源域内进程，核对 PID birth identity，使用 pidfd 发信号；
  超时使用已打开、固定身份的 `cgroup.kill`，不按进程名称清理。
- 独立监控不得处在服务的子孙或祖先资源域中；必须有内存硬限制和辅助 CPU。
- 1 秒采集主机可用内存、swap、PSI、服务 current/peak/high/max/events；
  磁盘/inode 每 30 秒检查。事件值可由首个采样重算增量。
- 主机阈值触发先标 `safety_abort_unattributed`；无法仅凭它断言被测系统 OOM。
- 监控异常是 `protocol_or_launcher_error`，不能作为系统低成本成功。
- 身份不一致时拒绝清理另一 invocation；保留错误，不扩大到全局清理。

实现依据 [Linux cgroup v2 官方文档](https://docs.kernel.org/admin-guide/cgroup-v2.html)
的层级限制、进程继承、`cgroup.events` 与 `cgroup.kill` 语义。实际执行版本与
控制器已在本机测试；没有仅凭新版本在线文档宣布功能存在。

## 仍未通过的正式启动门槛

1. 现有启动入口接入服务/回放两个资源域；模型启动前等待监控 readiness，并
   在监控死亡时停止继续启动。不能把一次离线 witness 当作持续保护。
2. 实际 Ray/容器/vLLM 模型进程、扩容产生的子进程和 GPU UUID 全部核验。
3. 推理机合计 Ray object store、spill/retry 的原生记录；GPU 占用/释放积分。
4. 逐文件系统 quota 和峰值；远端磁盘门槛仍有待用户确认，未擅自降低。
5. 数据采集开销检查、正常完成及中止分母/终态回执。

当前所有安全结果明确保留 `production_launch_authorized=false`。
没有正式性能运行，不能将这些检查描述为完成 M1/M2 或优于任何 baseline。
