# D1062：LastKnown 实现、定向检查与执行交接

2026-11-04。落实既有 A1/A3；不是 Full 优化，不新增实验矩阵。
本页保存时已完成实现和 CPU 检查，尚未启动性能回放。
前项 D1061 已完成、分析并备份；3B A1 性能节点仍为9/11。

## 已完成及当前问题

- D1061 普通 locality 相对 ElasticOnly 的 P95 TTFT 低5.2943%，GPU时间高
  0.9134%，是单轮取舍。Full仍有较差尾延迟与较低联合SLO；该反例完整保留。
- 先完成已登记的信息/HOST单因素边界，再综合控制开销及机制证据提出一个
  可证伪的底层优化。不能靠修改SLO、挑请求、取消核心机制来宣告Full胜出。
- 最新共享集群答复已落实：k3s属于其他项目，不访问管理配置或修改它；
  HydraServe继续复用独立rootless Podman适配和D807/D808/D809 3B结果。
  pp1结果不能冒称pipeline/consolidation收益，不重启已完成的基础适配。

## 本次改动与证据边界

设计依据为 `D1062_LASTKNOWN_INFORMATION_BOUNDARY_20261104.md`。历史核查包括
D1049消融入口、D1055 load-only、D1057 demand-only和D1061反例；不是重试
已拒绝的Full优化。在线复核官方[Dynamo KV offloading说明](https://docs.nvidia.com/dynamo/latest/knowledge-base/modular-components/backends/v-llm/native-kv-offloading)：
内部驻留与router可见事件确实可分离，但这只是设计参照，不是LoRA性能证据。

`LastKnownRoutingSourceView`仅保持tier/representation/footprint-bin及来源时间。
owner或成员变化清理旧记录，首次出现bootstrap，到期在下一次请求访问刷新；
0.5秒是已登记替代设计，不声称原生层级广播周期或最优同步间隔。实际源观测
本身可能更旧，分别保存保持年龄和底层观测年龄。

四个router入口（到达预取、可选准入前等待、预取重协调、实际dispatch）共享
信息视图。Q、feasibility、pending、utilization、EWMA、planner、admission
和selected-copy校验保持不变。用当前cost model重估旧来源类，不冻结EWMA。
纯shadow不重复增加正式选择计数；后续执行返回选中副本的**原始来源行**，
不是旧排名行。旧GPU提示不能授权引用或将实际HOST准备时间写成零。

请求回执为 `ieee_predecision_last_known_source_view_v1`，仍保存原始候选和
来源，另附 `ranking_information_view`；Full原回执身份不变。所有信息事件
保存于既有coordination/main-outcome journal。旧Full-only路由诊断会拒绝新
身份，不能直接对原始候选重算并把合理的排名分歧误报成路由错误。

## 检查结果及可复用来源

26项CPU检查通过（8项信息视图、8项既有公式选择、10项实际请求路径检查）。
覆盖A/A、时钟、更新边界、owner更换、live容量、当前成本读取、shadow无副作用、
四入口接线、实际GPU失效后的HOST重选及取消释放。没有新GPU smoke或整池重测。
已有25项共同计量/运行入口检查按源码SHA复用，不宣称已测试新的离线排名分析器。

测试资源域内存峰值680415232B，high/max/OOM/kill/swap均0；已退出。
首次启动CPU检查时scope标识长度不符合既有UUID规则，测试开始前被拒绝；
修正标识后26项通过，未放松规则，无GPU/服务启动。失败事实保存在资格回执中。

配置 `configs/ieee_tc/20261104_d1062_3b_last_known_w1.yaml`：
SHA `9b9273a8d254ac394af34ea09f4820e95832bdbf0d45c945db347a967f1792b7`。
与D1047仅两项信息视图字段及输出/HOST/NVMe路径不同；原始scenario仍为
`tc_ieee_full`，manifest的变体明确为LastKnown，不混写成Full结果。

资格及逐项日志：
`paper_results/ieee_tc/ablation/d1062_3b_last_known_w1/prelaunch/`。
原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1062_20261104/`。
147个旧投稿文件及两项用户dirty SHA未变；不stage用户manifest或EXECUTION_STATUS。
本页作为最新编号交接，不用旧EXECUTION_STATUS的D998头部覆盖实际进展。

指标仍为V1+单轮V2、D648共同阈值、D657参考；source42已知开发回放，
不声称独立留出、CI或统计显著。`n_correct=null`、3B零更新工件的独立非零
LoRA数值证明未知保持不变，不从固定输出和HTTP成功补造正确数。

## 下一步及完整未完成范围

先备份检查点，再用现有入口执行唯一D1062键；不能覆盖已有目录或重跑Full：

```bash
tmux new-session -d -s tc-d1062-last-known-3b-w1 \
  env FAASLORA_D1000_RUN_ID=d1062_20261104 \
      FAASLORA_D1000_RUN_TAG=d1062_3b_last_known_w1 \
      FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1062_3b_last_known_w1.yaml \
      FAASLORA_D1000_WORKLOAD=W1 \
      FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
      bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

实际启动仍核对新鲜磁盘/内存/GPU和远端健康，资源限制不变。不重启健康的D80
只读服务，不打包、不复制造数据。existing_unmonitored的远端连续监控限制保留。

结束后先释放并核验资源，复用D1047/D1061的共同指标、输入配对、机制和
first-service提取；新排名视图须独立重建选择并与实际dispatch来源区分。
view disagreement不等于物理false-ready，未执行候选无权威真值则保持未知，
不生成shadow反事实TTFT。无触发或负结果同样保留，不据胜负调整0.5秒。

之后NoHOST；7B的11配置及两个累积节点、受控A2–A5、M2、S1–S13、离线
CE/价格/Pareto、B1/B2/C1/C2/C3与两轮审稿文档、IEEE图表、最终同版本
来源统一和FaaSLoRA V2集成全部仍待推进。Full未接近目标前按最新指令不画图。

当前远端V2存在分叉，检查点沿已用origin V2和FaaSLoRA专用backup分支备份；
无force push，不把备份分支冒称已整合到实际FaaSLoRA V2。
