# 当前交接 D1082：完整W1已结束，资源改善检查点待结果备份

2026-11-06 12:43 +08:00：D1082唯一完整W1已完成，勿再启动同键。
4000/4000成功、原生token/输入合同匹配、全部GPU真实释放。联合SLO
3865/4000=96.625%，物理5393.347900GPU-s，P95TTFT1044.666805ms。
相对D1081资源下降11.984164%、SLO仅+1条，P95TTFT基本未改善，平均
TPOT/E2E略升。接受为G1资源效率开发检查点，不宣告G2或完整A1完成。
详见D1082_FULL_W1_OUTCOME_20261106.md。

收集已通过，completion/manifest、父版D1081输入匹配及阶段分解均完成。
远端132次/306360162bytes双边校验一致；2263监测样本无foreign/escaped、
无OOM/high/max/swap，7allocation全释放，缓存空、144旧投稿/两dirty SHA不变。
174的3B unit在核对本次PID/InvocationID后已停；7B/k3s不动。两个有效
统计域及一次UUID命名错误的空分析域已停止，无推理重跑。尚未新画图。

下一：按github-sync备份这次完整结果；不再做无新假设的W1微优化。
随后同版本同配置W0→W2验证，再按共享路径依赖更新受影响A1，优先两个
已知简化消融反例；已有外部baseline不重跑。完整剩余范围见结果文档。
目前没有GPU实验在跑。下文保留启动过程，不表示仍在等待。目标ACTIVE。

## 启动及监测历史

2026-11-06 11:59 +08:00更新：完整计划1567行及V1/单轮V2已重新读取。
代码检查点462f9987944d68370be7e411abb9252980a3e2b1已推送并核验
origin/retry14_continuous_queue_v2与faaslora_origin/
backup/d1082-host-budget-projection-20261106。实际FaaSLoRA V2分歧未强推。
backup.json在本轮raw根目录；144个保护文件及两用户dirty SHA保持不变。
备份辅助域c6cc3cc528a548baed99993eca3d08ae确认TasksCurrent=0后已停止。

唯一D1082完整W1已通过allow_exec=true并启动，禁止再次启动同键：

- tmux：tc-d1082-full；launcher PID1289278；service PID1289288；
  外置replay PID1289286；watchdog PID1289309。
- service：primelora-tc-svc-cfba88136b40477c9518b13994fb851e.scope，
  InvocationID=2fda96997d9f42c78361450d373505ff，inode40440246。
- aux：primelora-tc-aux-271c7e7d134540a6afa78bea6d00d5f7.scope，
  InvocationID=da344ea6ec184cf0a5bda3e92ede4326。
- RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/
  d1082_20261106/d1082_3b_host_budget_projection_w1；exec_receipt在launch.launch下。
- receipt确认4000请求、W1、原trace/view SHA、60.0秒共同通知，
  notice monotonic3444802.290648852、replay t0=3444862.290648852。
- D1047 YAML SHA263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2；
  fresh_d747_v1，Full，无prefix/详细profiling，开发身份不改写。
- 只启动远端已有3B unit，PID3779618，
  InvocationID=cde7bd3cdf5648828784c762d5ed4f08；health确认prepublished_gzip_v1、
  artifact_timing_v2及既有3B池。远端空闲130512850944B。未改服务/工件/k3s。
  第一次health只读检查误用缺numpy的系统Python，改用既有LLM环境后成功；
  未安装包或重启服务，不记为性能失败。未新增远端CPU持续监控，不声称排除全部外部干扰。
- 第19次watchdog采样host available109308502016B，disk163918114816B，
  service peak3550851072B；high/max/OOM/swap0、无foreign compute/escaped，
  服务72/80GiB及辅助3/4GiB资源域有效。启动阶段数据不是完整运行验收。

下一步仅监控此运行；测量期间不维护/构建/压缩/推送，也不并行开始新候选。
完整4000请求和真实GPU释放后复用D703/D299/D1064分析，保留任何退化或失败，
不从组件CPU收益推定TTFT/SLO/GPU-s改善。下文保留启动前登记。

启动后已见27条到达、25条成功、0失败，原生GPU进程1290834持卡；
不是只创建tmux。第108次监控：host available100827799552B，
disk163417092096B，service peak13045088256B，high/max/OOM/swap0，
foreign/escaped为空。live界面的5000ms SLO及CE不是冻结联合SLO，不用于判优。
本状态更新尚未在GPU测量期间提交；测量后与完整结果一起检查备份。

## 启动前登记（历史）

先读目标、完整PrimeLoRA-PLAN、V1/单轮V2及本页。D1081已完整结束且备份，
不要重启；旧EXECUTION_STATUS.md为用户dirty，不编辑/暂存。

当前主线3B A1 Full净收益。已有外部baseline与3B十一变体不重跑；HydraServe
独立Podman，k3s不属本项目，保持不动。D1081联合SLO96.60%、U6127.701722、
P95 TTFT1043.464586ms，但未超过简化消融，未宣告目标完成。

D1082仅省去预算消费者不读取的逐adapter footprint表，所有真实张量/alias/
pinning/allocator与预算检查保留。默认完整图仍供路由/planner。详见
D1082_HOST_BUDGET_PROJECTION_20261106.md。52项接入检查通过；CPU组件有
收益但不是服务结果。没有调参、改九式/指标/工件/负载、图表或新baseline。

下一：备份检查点后，用已有D1000入口+D1047配置运行唯一完整W1。
RUN_ID=d1082_20261106；RUN_TAG=d1082_3b_host_budget_projection_w1。
必须60秒、fresh_d747_v1、Full、4000请求，无prefix/详细profiling；当前
尚未记录启动PID，不据此称运行中。所有历史结果保护、150/100GiB磁盘、
102GiB空服务内存准入、72/80GiB服务与3/4GiB辅助域保持。
真实远端只复用D80发布缓存与已安装3B unit，无请求打包或额外压力。

已停止三个空测试/组件域。无GPU作业；用户两dirty SHA不变的复核在备份中
执行。实际FaaSLoRA V2分歧不强推，只备份专用分支并明确非最终集成。

仍待Full净收益、同版本跨场景、M2、7B A1/累积、受控A2–A5、S1–S13、
CE/Pareto/价格、审稿及B1/B2/C1/C2/C3文档、正式图表和实际V2集成。
n_correct未知/开发身份保持，不重开资产支线；目标ACTIVE。
