# D1014：共享文件来源优先的辅助预取查询

2026-11-02。本文保留 PrimeLoRA 3B W2 开发候选的事前登记。
完整回放现已结束：3652/4000联合通过（91.30%），拒绝替换D1000；详见
`D1014_3B_FILES_FIRST_PREFETCH_OUTCOME_20261102.md`。下文启动前表述保留其历史时态。
继续采用完整 PLAN（SHA `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`）
及冻结 V1／单轮 V2。HydraServe 两模型的 W0/W1/W2 已有记录，不重开基础适配，
不访问共享 k3s。本候选不删除剩余主比较、消融、敏感性与最终交付范围。

## 历史排除和实测依据

- D1000：联合 SLO 3690/4000（92.25%），物理 GPU-s/request 2.002078。
- D1004、D995/D996、D1008–D1010、D1012：等待位置改变、强制在准入前完成
  文件准备及局部压力判据均未解决联合 SLO，不能再次包装成新候选。
- D1005–D1007、D1013：加卡、保留容量或仅加入历史启动时间带来质量/资源
  取舍，未通过目标。本候选不改变这些控制参数，也不启用 D1013 的 prior。
- D1011：四路 movement 在请求前因配置/profile 身份不一致被拒绝，没有性能
  证据。本候选继续沿用实际三路 movement，不绕过这一绑定。
- D1000 的非 GPU-ready 计数不是另一个副本存在 GPU 副本的证据；遵循
  `D1000_METRIC_LABEL_CORRECTION_20261101.md`，不以它证明路由错配。

本轮重新流式读取 D1000 的全部 4000 请求，原始 JSON SHA 为
`f1d0d342814e77e2d41c0cc8d7641aa3e40fb70112b5202e2e8d6b0f5b92dfb0`：

| 到达预取结果 | 请求数 |
|---|---:|
| 完成远端准备 | 508 |
| 完整查询后选中非 Remote 来源 | 3426 |
| 无可行副本 | 25 |
| 副本集合发生变化 | 41 |

3426 个非 Remote 结果分别为 GPU 1694、HOST 1210、NVMe 522。
这些是旧运行的结果，不保证新候选恰好跳过3426次：native HOST/GPU 不等于
共享文件仍存在，时间交错也可能改变。该组请求记录的控制路径均值为63.307ms，
但这不是辅助预取的独立耗时，不可据此推算能节省63.307ms或反事实 SLO。

## 第一性原理与唯一改动

辅助预取只想确定是否需要 Remote→NVMe。当前同一 runner 的文件 owner 为所有
专属副本提供共同 HOST/NVMe 文件来源；`confirmed_source_class` 本来就把相同
`files` 视图用于每个副本。若已有验证通过的同一工件文件，辅助预取不需要再向
全部 native owner 查询，才能判断无需另起远端移动。

令 F(a) 为文件 owner 当前返回、通过原有内容/表示/身份验证的共享来源集合：

```
if F(a) is nonempty:
    skip only the optional arrival-prefetch route probe
else:
    perform the original confirmed snapshot and route; start owned movement if Remote
```

仅启用 `route_confirmed_arrival_prefetch_files_first_v1`，父配置为 D1000。
把已有 file-source 验证提取为同一个纯函数供两处复用，不增加另一份弱验证器。
新记录包含 owner/epoch、观测层、采集时刻、查询耗时以及是否调用完整辅助路由。
代码未添加睡眠、阈值、未来信息、缓存容量或额外 transfer lane。

最重要的边界：这只抑制不必要的辅助工作；它不是执行许可。后续正式请求必须
重新取完整路由快照、同快照排序、原子预留并保护选中来源。两次之间发生驱逐时，
正常请求承担真实再获取及等待；不拿旧文件视图写成 GPU-ready，不省略 admission。
来源查询错误保留辅助任务 failed 回执，普通权威路径保持原行为；取消向上传播。

## 原始文献与实现核对

本轮重新联网查看了：

- [vLLM 0.30.0 的原生 LoRA manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  `LRUCacheWorkerLoRAManager.add_adapter` 区分已加载命中与需要加载的路径，
  但均在原生 owner 中激活。这支持把“避免重复准备”与“执行权限”区分；本候选
  没有改动该 manager 或用文件命中代替它的原生状态。
- [dLoRA 原文](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)：请求与
  adapter 的联合编排同时受执行负载和位置影响。因而本候选不将 locality-only
  排序换入 IEEE 路由，也不把 dLoRA 的跨副本迁移收益当作本机证据。

外部资料说明设计边界，不证明本候选必然改善性能。待检验假设是：缩减非必要的
native 状态查询可减少控制/生成争用，且新增的文件查询不会抵消这一收益。

## 已完成的必要验证

325 项 routing/lifecycle CPU 测试通过，保留19条既有警告。
新测试覆盖真实已发布文件、无文件、异步 owner、主请求仍独立查询、内容/时钟/
采集时刻错误以及取消。旧 confirmed routing 的完整文件验证和原生优先级不变。
测试在辅助 CPU 2,3,26,27、MemoryHigh/Max=3/4GiB、swap0、CUDA隐藏的独立 scope
中运行并退出0；不将这些设置声称为新的 GPU 数值或性能资格。
日志 `results/ieee_tc/profile_output/d1014_20261102/tests1.txt`，SHA
`a858379c1a21f42b21be38ff013c6d404f79c300721913bd5ac1bdb57aeb0288`。

不重复不受影响的后端/生成 smoke。下一仅一次完整4000请求 W2，新执行键：
`d1014_full_20261102/d1014_3b_files_first_prefetch_w2_full`。
复用原 D1000 launcher，用显式覆盖传入新配置及输出路径：

```bash
FAASLORA_D1000_RUN_ID=d1014_full_20261102 \
FAASLORA_D1000_RUN_TAG=d1014_3b_files_first_prefetch_w2_full \
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261102_d1014_3b_files_first_prefetch_w2.yaml \
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

launcher 的历史文件名含 w1，但真实输入视图、`FAASLORA_TC_REPLAY_PROFILE` 和本
候选都是 W2，不能按文件名改写负载身份。原 source seed42、500-ID集合、原生
fixed-length greedy、D725 profiles、共同 D648 SLO、三路移动及资源包络均保留。

## 裁决和停止条件

完成后以原生终态、4000请求时间线、物理持卡并集、取消/来源回执核验完整性；
记录跳过辅助查询数、实际远端次数、失败数及新文件查询耗时。再按冻结协议报告
联合SLO、TTFT/TPOT尾部和物理GPU-s/request。不用dispatch-Remote下降声称
传输消失，不由325项CPU测试推定全量数值正确性。

达到至少3800/4000联合通过只是时延资格，资源与全部正确执行条件分别核验；
未达标仍可记录真实改善或取舍，不通过重复同键、改SLO或筛请求获得通过。
候选没有改善时归档，除非有具体新证据，不再搜索文件查询顺序的变体。
未接近目标前不作展示图。全部已完成 baseline 记录复用，后续完整范围不删减。

## 变更清单与备份边界

| 文件 | 改动 |
|---|---|
| faaslora/experiment/instance_pool.py | 原文件来源验证提取为共用纯函数，无排序/公式变化 |
| scripts/run_all_experiments.py | 可选 files-first 辅助查询与触发记录 |
| tests/test_ieee_tc_request_lifecycle.py | 五项新边界测试 |
| configs/ieee_tc/20261102_d1014_3b_files_first_prefetch_w2.yaml | 从D1000只切换候选及隔离输出路径 |
| 本文 | 来源、假设、验证、执行身份与判据 |

以上触发计数是候选机制证据，不插入内循环 profiler。原始日志与用户修改不覆盖。
本轮核验时 FaaSLoRA 远端 V2 有9个本地没有的历史提交，当前本地另有424个提交；
不强推、不宣称已集成。新检查点先推正常 origin/V2 和 FaaSLoRA 独立 backup 分支，
最终 V2 分歧整合单独处理。`EXECUTION_STATUS.md` 的既有 dirty 内容保持原样，
本文件作为本轮可提交的执行记录。

## 启动回执（2026-11-02 15:44 +08:00）

- 候选检查点 `63c2610414dc9547345e0c6d213a35f278a45e43` 已通过远端读回确认：
  `origin/retry14_continuous_queue_v2` 及
  `faaslora_origin/backup/d1014-files-first-prefetch-candidate-20261102`。
  未强推存在分歧的 FaaSLoRA V2。
- 旧投稿保护 seal 校验147项、零变化；本轮未暂存两份已有用户修改。
- 远端既有3B用户服务已启动，认证health确认正确3B池及
  `prepublished_gzip_v1`；复用既有交付缓存，未重新发布或打包。
  运行期间不更改远端服务。其他项目的k3s未改动。
- 唯一完整运行已经开始，不另开短回放：tmux
  `tc-d1014-3b-files-first-w2`；服务scope
  `primelora-tc-svc-ae77354af7c94398860fea59de79b25a.scope`，invocation
  `b143d093ac1b4c7ca5672df7c501b7f3`。
- 原始目录：
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1014_full_20261102/d1014_3b_files_first_prefetch_w2_full`。
  `launch.launch/exec_receipt.json`确认服务72/80GiB、swap2GiB及辅助3/4GiB、swap0。
  已出现原生EngineCore进程及请求提交事件。第76次资源样本主机可用96.31GiB、
  服务占用7.36GiB，无high/max/OOM事件、无告警/中止原因。这是运行中状态，
  不是完整回放或性能资格。
- 历史D998辅助scope仍被systemd列出，但TasksCurrent=0；不误判为并行实验，
  不在测量时做无关清理。性能回放期间不做构建、批量哈希、压缩、测试或推送。
- 下一步仅监测本次执行键；终态后复用D703审计入口、补新机制计数，报告完整
  4000请求的联合SLO和物理GPU占用。没有最终结果前不宣称改善，也不启动另一候选。

### 同一运行的继续监测

下一目标轮确认为verified-wait：实际服务scope仍active/running、237个任务，
不是仅依赖tmux名称或旧状态文件。第354次外置样本无high/max/OOM及保护事件；
已提交至index176，终态文件当时176行。继续观察同一执行键，不因观察间隔重启。
仅复核既有D810、D715及冻结V1/V2小型文档，未扫描原始大结果、运行分析作业或
改动在线配置。D810基线观测继续保留；其旧PrimeLoRA行不能填充D1014的W0/W1。
95%不是基线入表条件，低于95%的完整观测仍报告；D1014完整性能与机制判定待终态。

### 终态覆盖上述等待状态

4000/4000正常终态，服务/replay/watchdog退出0，服务域及GPU上下文已释放。
联合SLO为91.30%，比D1000低0.95个百分点；物理GPU-s为8007.929976。
辅助完整查询确实跳过3489次，但没有改善最终目标；不重复本执行键，不启用该
候选替换参考。后置147项投稿保护零变化，两份用户dirty文件字节不变。
旧D1013和本轮闭合main_outcome日志已双重解压SHA验证后无损压缩；可恢复，
原始请求结果JSON仍保留。下一主线与全部未完成范围见OUTCOME文档。
