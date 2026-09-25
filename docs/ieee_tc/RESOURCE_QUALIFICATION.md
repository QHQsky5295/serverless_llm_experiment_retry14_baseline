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

1. 独立回放和启动期间服务入口接收均已接入并通过微测，见
   `EXTERNAL_REPLAY_QUALIFICATION.md`。监控 readiness 已接入；实际模型初始化
   是否阻塞服务入口、跨进程时间和资源资格仍需真实 worker 验证。
2. 实际 Ray/容器/vLLM 模型进程、扩容产生的子进程和 GPU UUID 全部核验。
3. 推理机合计 Ray object store、spill/retry 的原生记录；GPU 占用/释放积分。
4. 逐文件系统 quota 和峰值；远端磁盘门槛仍有待用户确认，未擅自降低。
5. 数据采集开销检查、正常完成及中止分母/终态回执。

当前所有安全结果明确保留 `production_launch_authorized=false`。
没有正式性能运行，不能将这些检查描述为完成 M1/M2 或优于任何 baseline。

## 现有入口的监控启动握手（2026-09-26）

历史 `run_all_experiments_user_scope.sh` 没有硬限制，systemd 不可用时还会
无隔离继续运行。它不能承担 TC 资格。新增 opt-in
`FAASLORA_TC_QUALIFICATION=1`，沿用同一入口和 preflight：

1. supervisor/watchdog 处在同一 4 GiB、swap=0 辅助域，CPU 2/3/26/27；
   服务位于独立兄弟域，72/80 GiB、swap=2 GiB、CPU 4–23/28–47。
2. 服务首先执行无模型 gate，读回真实限制，再通过短路径私有 Unix socket
   发送 PID/计划/资源身份。supervisor 核对内核 peer credentials，不信任文本 PID。
3. watchdog 验证服务范围、外置身份并取得首个合法资源样本后，才允许 exec
   模型环境；用 PID birth identity 拒绝过期 readiness 文件。
4. 原 engine initialize 和扩容入口再次检查当前服务身份；这条路径不调用
   全局 stale-worker 清理，失败不暗中禁用 cache/prefill 或换内存配置重试。
5. watchdog 异常/死亡时，supervisor 清理原身份下的服务；正常结束按真实
   descendant 存活情况收尾。使用已有 scoped cleanup，不全局停止 Ray。

生产尺寸的资格启动还检查已有 build/service，拒绝与当前安装并行。以下微型
witness 只允许固定 `_worker --mode inspect`，不能借 tiny 开关启动模型。

| 检查 | 结果 | 证据 |
|---|---|---|
| 第一次完整握手 | 未通过，保留：进程已退出，空 transient scope 仍存在；旧判断误等 60 秒 | `launch_handshake_attempt1.json`，SHA `246b59dda3dce88ed7b76098d850971176f5d484076e7066c61f12f8e14a310b` |
| 修正后完整握手 | 通过：外置 watcher 首采样后才 exec，真实子进程继承限制；空 scope 安全停止；全部清理 | `launch_handshake_attempt2.json`，SHA `6bd84776ebcf99d13f81b11617a4404c4f4e9303d6bf02af8381e0fd0a3527bb` |
| 资源/身份规则 | 24 项 system-Python 测试通过 | 缺限制、伪 tiny、已有安装、watcher PID 复用、缺 receipt 等 |
| 原 engine 接线 | 5 项新增测试；完整功能回归 377 项通过 | 禁止全局清理、缺资格拒绝、单配置尝试、原错误保留 |

空 scope 的判断采用 `cgroup.events: populated=0`，不是看到启动进程退出就
默认所有后代已释放。GPU UUID/驱动资源释放仍需独立证明，不能拿空 scope
直接替代生命周期积分。[Linux 原始规范](https://docs.kernel.org/admin-guide/cgroup-v2.html)
明确区分资源域存在与是否仍有活进程。本机 systemd 249 的实际 witness 为
上述行为提供本机证据；未依赖在线文档的新版可选开关。

原始 `.launch/` 留在本机，不覆盖；curated 回执保存其 SHA。当前标准启动：

```bash
FAASLORA_TC_QUALIFICATION=1 \
FAASLORA_TC_LAUNCH_OUTPUT=/absolute/new/campaign/launch.json \
FAASLORA_TC_PREDICTED_GROWTH_GIB=<audited-incremental-peak> \
FAASLORA_PYTHON=/absolute/qualified/environment/bin/python \
bash scripts/run_all_experiments_user_scope.sh <existing-runner-arguments>
```

这不是正式实验放行命令：模型环境仍在安装，native worker census、GPU/Ray
计量和真实模型初始化/清理资格仍待完成。没有启动任何模型，
没有把 128 MiB witness 的峰值当作服务内存消耗。资格结果采用本节状态表，
不绘制缺乏性能含义的曲线。
