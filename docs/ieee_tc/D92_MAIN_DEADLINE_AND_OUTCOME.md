# D92 — 完整回放的超时、失败证据和缓存收尾

## 最新实际回放：3B W0 attempt1 未通过，已收尾

2026-09-28 03:18:53 启动，执行源码 `08cbf45`。以下覆盖后文“下一步”的
启动状态；保留后文作为启动前检查记录。

| 项目 | 实际观察 |
|---|---|
| 计划/已提交/成功/随整轮中断取消 | 4000 / 27 / 23 / 4 |
| 未提交 | 3973；未伪作 timeout |
| 初始及后续激活状态 | 1 个 ready，3 个取消；4 个真实持卡记录全部释放 |
| 驻留规划 epoch | 12 completed / 5 superseded / 1 failed |
| file preparation plan | 14 completed / 5 superseded / 2 cancelled |
| 真实远端传输 | 13 对 UUID 完全匹配，30,210,124 wire bytes |
| 已核验解包字节 / 请求打包 | 542,464,828 / 0 |
| 资源监控 | 164 样本，峰值19,352,363,008 B；最小主机可用95,952,441,344 B |
| high / max / OOM / 实验 swap | 全部 0 |
| 收尾 | 四张卡实际释放；HOST/NVMe 本轮工作区删除；两远端服务/监控停止 |

原始控制异常为 `automatic planning requires one complete confirmed file-owner view`，
发生在 residency epoch 的自动候选输入检查。请求级、原始 traceback、远端
UUID 和独立机制事件均被 D92 新边界完整保留；没有二次 snapshot/cleanup
错误。4 个取消不是1800秒 deadline，也不是4个独立性能失败样本。

外置 launcher 随后记录 publisher 失败并将自身类别标为 protocol/launcher
error，最终 service 返回 -15、watchdog 0；该二次标签不替代上面的原始
控制异常。所有请求/持卡记录都在清理前保存。观察GPU占用329.281494 GPU-s
属于不完整运行的 U_obs，不能把它除以4000作为完整工作负载低成本结果。

暂不判定为网络、OOM或具体竞态。该拒绝条件合并了 owner/epoch/时钟/集合
完整性，当前失败日志没有逐字段输入快照，尚不能确定是哪一个条件不满足。
下一项为 CPU 上隔离 file-owner 完整快照的生成与自校验，不放松拒绝条件，
没有通过反例验证前不再重跑 GPU。7B及 baseline 仍未启动。

本轮原始目录 `d92_20260928/3b_full_w0_attempt1`，curated 文件
`20260928_d92_3b_full_w0_attempt1.json` 含16个来源 SHA；旧147保护项不变。

## 问题与处理

D91 已贯通主入口的配置装配，但尚未运行 canonical Full 4000 请求。
本次只补齐计划 §4.6 的有限保护和主入口失败计量，不修改 IEEE 九式、
模型配置、D89 实测初始化、500 工件或原请求序列；不重做远端缓存发布。

请求剩余保护时间为 `planned_arrival + request_timeout_s - now`。
开发/资格按已批准的 1800 秒，必须显式提供且有限正值；正式逐请求保护
仍须依指标 V1 的共同标定冻结，本次不伪称已经完成。已过期请求不再提交，
排队和初始化导致的迟到不获得新的保护窗口。非 IEEE 旧路径未设置时不改变。

Python 的 timeout 在内部借助取消并在上下文退出时转换为 TimeoutError；
因此终态计量必须放在转换外侧。外部整轮取消继续传播 CancelledError。
vLLM 0.30.0 的 generate 取消路径会调用 abort 后重新抛出，现有本项目
native 引用、传输线程和 reservation 清理继续负责真实资源闭合；超时不
意味着 GPU 已释放，清理错误也不改名为超时。
[Python 官方文档](https://docs.python.org/3.12/library/asyncio-task.html#timeouts)、
[vLLM 固定版本源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)。

## 收尾证据

- physical deployment 下独占写入 `main_outcome.json`，保留错误类别、
  已观察到的中断请求、完整计划分母、HTTP UUID、层级与准备事件、清理结果。
- 请求正常结束后才出现控制/清理错误时，也保留已完成窗口的请求记录。
- 任一收尾阶段失败仍尝试后续阶段；原始执行异常不被二次清理异常替换。
  后者单独记录。原本正常执行若清理或必需快照失败，仍失败退出。
- 两个缓存根必须本轮独占新建，记录 device/inode；既有或重叠目录不清空。
  收尾检查实际 file owner、根身份和未闭合引用/准备，复用既有删除保护。
  失败保留工作区，不以强删掩盖仍在使用的源。没有改动历史结果目录。
- physical allocation/release journal 仍由外层 finalize；终态不能代替释放。
  成功请求主数据继续由原正常结果入口输出；新 sidecar 不产生第二套性能排名。

## 即时检查表（CPU，不是性能实验）

| 项目 | 结果 | 证据边界 |
|---|---|---|
| 已过期请求、执行中超时、外部取消、清理错误 | 通过 | 不伪造 native completion 或 release |
| 旧目录/重叠根、活跃引用、被替换根 | 拒绝误删 | 只清本轮拥有的缓存 |
| 原始异常与二次异常、失败请求/UUID | 保留 | 不替换原始失败、不补造未来请求 |
| 完整相关回归 | 835 项通过，50.078 秒 | 含 native ownership、HTTP、回放与 smoke |
| 3B 实际主入口 | 4000 请求/500 adapters，主动启动前中断 | 1800 秒合同，错误与清理回执通过 |
| 7B 实际主入口 | 4000 请求/500 adapters，主动启动前中断 | 同上；没有 GPU/HTTP 性能测量 |

保留 red 日志和第一次 outcome 测试失败：测试 fixture 误用只接受关键字的
acquire API，修正测试调用；没有改变生产保护条件。实际装配复用 D91 驱动的
最小扩展，原驱动及证据不改写。新配置中只有此次批准的开发 timeout 与新
输出/缓存路径；模型 child 仍严格等于原 D88 已测合同。

历史 ledger 全文原样归档为 `EXECUTION_HISTORY_D81_D91.md`，SHA256
`4cf98a011577c5996c13fa20621cd413b637f40f93d63ef3ad4cce6f990cb7dd`；
活动状态文件保留权限、资产、已完成证据、未完成矩阵和唯一下一步，避免
历史 LIVE 标题导致重启已经结束的实验。源计划和冻结指标 V1 未改。

## 下一步

备份后先执行原 3B4000、W0 的 Full 开发回放，使用实际外置到达、共同
60 秒准备窗口、现有 native 环境、已发布真实远端和物理持卡计量。
清理→校验→表/图→解释之后才执行 7B。baseline 继续暂停；不再重复
D78/D80/D81/D88/D89 或 D90 短前缀。Warm SLO/Resident 参考、正式比较、
消融和敏感性仍待完成。原零权重工件的数值区分性限制不因这些检查而消失。
