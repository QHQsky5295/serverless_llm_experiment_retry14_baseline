# D1028：保持驻留容量的可达 active-LoRA 执行图候选

2026-11-03。当前状态：接入及 30 项 CPU 检查通过，画像输入已准备；尚无新 GPU 结果。
承接已封存 D1027；不覆盖用户修改的 EXECUTION_STATUS.md。完整目标 active。

## 主线与假设

本轮完整核对目标、PLAN 和最新完成记录。PLAN SHA
d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
D1027 完整 3B W2 联合 SLO 3784/4000（94.60%），8316.260704 GPU-s，
P95 TTFT 1970.868099 ms。仍差 16 条达标；独立数值正确性未知，不能排名。
D1027 不重跑。D810 基线及其适配局限保留，不重开 HydraServe 或共享 k3s。

已核对 D1023、D1021、D724 及修改历史。D724 是 token 轴，D1021 新引入
active-LoRA 专门化后，实际图数增至 88 PIECEWISE/32 FULL。D1027 全部六个
worker 均重复捕获，后续编译已热仍需约 34–37 秒初始化。D1022 初始编译冷、
D1027 热的差异已明确保留，不把 D1027 初期改善全部归给文件校验并行化。
文件校验方向闭合，不扫描线程数，不重开已否决的等待/重路由/文件优先策略。

唯一可证伪假设：保留 32 槽和真实批的原生图映射，只删除 scheduler 不可能
生成的中间活跃计数分支，能否减少初始及自然扩容捕获代价，并在完整 W2 中
改善联合 SLO？更少图可能改变图内存/KV 池，需重新画像，不预先宣称加速。
初始化不是全部瓶颈，后 3500 请求仍全部保留；不能离线扣启动时间造结果。

## 推导与实现边界

每请求至多一个 LoRA，scheduled 请求受 max_num_seqs 限制。因此真实活跃
数至多 min(max_num_seqs,max_loras)。按原生向上取整映射保留到该上界所需的
全部 case，同时保留 0 和 max_loras+1 的 dummy/profile 哨兵。非二次幂上界
也保留其上取整 case。当前 [0,1,2,4,8,16,32,33] 变为 [0,1,2,4,8,33]。
66/24 是静态预期描述数，不是实测收益，也不是底层 CUDA 节点数。

通过 vLLM 公共 worker_cls 接口，在原生 init_device 完成且模型加载/图捕获
尚未开始时设置该 worker 的 runner capture cases。不改全局安装、dispatch、
kernel、token 轴、32 个槽、请求并发、临时内存估计图释放流程或 IEEE 九公式。
保留现有 worker extension 与 loader；不提供失配后静默 eager/default 重试。
新策略保留在严格 model/profile 身份内，初始化实测回执进入现有 worker 观察。

2026-11-03 核查官方 v0.30.0：
[V2 LoRA 计数与 dummy 分支](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/lora_utils.py)、
[图候选与原生向上取整](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/cudagraph_utils.py)。
实际安装文件身份和 V2 路径沿 D1023，测前再次记录 SHA；不以旧 dispatcher
代替实际 V2 路径，不把源码推导等同于 GPU 数值正确性。

## 执行与停止边界

1. CPU 测试验证所有合法活跃计数与原生映射一致、dummy 保留、非法配置拒绝，
   worker 接入不覆盖 loader；验证旧 timing profile 不接受新身份。
2. 复用 prepare_d701/run_d298 的 24 个既有 adapter、76 波、423 请求进行一次
   新画像，旧固定输出长度只作已完成统计初始化，所有时延重新测量。
3. 画像已覆盖改变的原生路径时不追加短 Full。验证实际图数、pool/KV、原生
   token/binding、真实远端、完整资源释放；随后一个新键完整 4000 请求 W2。
4. 比较保持 D648 SLO 与 W2 12083.063870 GPU-s 预算，单轮 V2，不算 CI，
   不删阶段、不筛赢家、不重复同键。反例归档后回主线，不做局部形状扫描。

使用 vLLM 技能定位真实初始化路径；通用吞吐/时延建议不替代冻结协议。
沿 run-experiment 复用现有隔离 runner，测前按 github-sync 检查并备份。
推理机 150/100 GiB、内存/CPU 包络保持；无请求时打包，无共享服务改动。
未完成两模型同版本 W0/W1/W2、G1/G2、独立数值正确性、A1–A5、S1–S13、
成本/Pareto、审稿文档、后期图表与最终 V2 集成不删除。达标前不画图。

## 测前完成记录

- 新增 20 项 CPU 测试及 10 项相关既有/新增长度绑定测试通过，pytest 30 passed、
  115 deselected、19 既有警告，11.42 秒。覆盖原生 CudaGraphManager 的真实
  映射函数、非二次幂界限、dummy、无裁减情况、重复安装、公共 worker_cls
  与原 loader extension 共存及严格画像身份。测试不生成 GPU 性能结论。
- prepare 首次缺 PYTHONPATH，在导入项目之前失败，无输出或 GPU；按既有根
  路径重新执行后通过，没有修改资格规则。24 adapter、76 波、423 请求，
  时延复用 false，新 backend ID 为
  native-model/0f0fea04e5f8ae97cdd6692817758f23239ec5f4c189c4abcbffac7ee8ea0e64。
- 配置 SHA b605361cfb0f6af3eb55b187b030f0148e67fab0764087478cd16c6948fc9010；
  source_spec SHA 50d4f913df644b601b51b5f4c7119c8b9f82734a45a074f8831b2e91b23aa2bc；
  development_contract SHA 00843e27af632d82400ddb474edc43c2d73a2065379f34342b616bd00002fae0。
- 实际 lora_utils/cudagraph_utils/model_runner SHA 与 D1023 一致；gpu_worker
  SHA 6994436e4547c0996ab555b3e814e8ccea26b650dfc1b378b97d640745309347。
  scheduler 限制及断言 running≤max_num_seqs，V2 分派从当前 scheduled IDs
  计算活跃 LoRA；本候选不覆盖多 GPU、speculative 或 V1 runner。
- 147 旧结果保护项通过，两个用户 dirty 文件 SHA 未变。静态 diff/shell 检查通过。
- 独立辅助 CPU 2/3/26/27、3/4 GiB、swap0 中完成 CPU 检查；未保存退出前
  全部资源计数，不能据 scope inactive 宣称完整零事件。
- 测量前无损压缩已关闭 D1020 原始 result，双次解压及原件 SHA 核验后回收
  237019136 B。回执 storage/d1028_closed_d1020_result_20261103.json；保留可
  恢复原始证据，D1027/D1022 未动。可用 161382600704 B，后续仍即时复核。
- 远端仍为既有 PID1984520（10月31日启动），两端链路读回1000/full，无服务
  重启或配置修改；远端可用132670201856 B。无连续远端CPU监控，限制保留。

下一步只启动一次 `bash scripts/ieee_tc/run_d298_capacity_source.sh d1028`，
使用既有原生画像/资源隔离入口；必要备份后启动，不额外插入短 Full。
