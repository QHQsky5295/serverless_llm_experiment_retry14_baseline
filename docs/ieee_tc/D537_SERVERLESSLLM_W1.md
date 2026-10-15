# D537：ServerlessLLM 固定原生工作点的 W1 突发负载

2026-10-15，启动前登记。只推进既定主比较，不新增优化候选。
本文件创建时未启动性能运行；状态以 EXECUTION_STATUS 和实际回执为准。

## 问题与固定合同

检验 D414 已验证的 ServerlessLLM 工作点在 W1 突发到达下的共同联合 SLO、
P95 TTFT 和真实生命周期 GPU 时间。保持 D414 调度、扩缩容、加载与资源
配置，只改变既定到达视图；不是重复 W0，也不为 baseline 搜索更好参数。

- 原 7B 4,000 请求与 500-ID 工件池不变；实际出现 132 IDs，目标输出
  458,224 tokens。源 trace SHA
  `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。
- W1 为同序列 8×500、段内到达间隔减半、段间30秒；不等实际 drain。
  view SHA `103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3`。
  `fixed_length_greedy_v1`、`role_lines_v1`、source42/development41；
  `formal_run=false`，不改写为独立留出证据。
- 原生 vLLM 0.10.2、FP16 native store、RR、已公开轮询及 request-identity
  queue 修正；min1/max4/target4/keep10、maxseqs4、maxLoRAs4、CPULoRAs4、
  tokenbudget1024、GPUfraction0.72、非 eager。原生 backbone checkpoint 复用。
- 共同提前60秒部署通知，原生计时、1,800秒资格保护与 D394 共同 SLO 不变。
  实际 store 持卡按物理 GPU 计量，不能按逻辑实例空闲冒称释放。
- 使用本轮空 LoRA 下载缓存，复用远端 D78/D80 已发布只读 gzip；请求中无打包。
  真实读取、传输、竞争、本地物化计入时延，不做事后扣减。
- 保留原生磁盘编译缓存，已记录 D414 实际 prefix 的当前内容；不声称整个底层
  kernel 缓存初态相同。不预热模型/GPU/HOST/KV，所有实际启动/capture 均计入。
- 服务72/80GiB、swap2GiB；辅助3/4GiB、swap0、既定 CPU 分离；
  推理机150/100GiB、102GiB内存启动门槛与运行保护不变。预估新增32GiB。
- 单键一次完整回放，保留成功、失败和不达标，不计算运行间CI，不筛选赢家。
  独立数值正确性未知继续保留；HTTP成功、原生token/绑定不冒称独立数值证明。

## 复用与必要核查

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d537_20261015`。
沿用 D414 `run7b_repaired.sh`，仅替换本轮路径、独立作用域和显式 W1 输入。
source view 的原始及测量后 app/backend/router/scheduler SHA 均与 D414 相同。
当前 baseline checkpoint `bde215460acb9e850efa479a89bda6fc39a2bd7d`。

共享测量工具此后扩展了其他 backend：原生 ServerlessLLM 的配置、服务生命周期、
源变换、resolver、ingress、NativeRequestObservation 经 AST 核对保持不变。
snapshot collector 新增仅在存在时复制 v030 identity；v0102 hook 只是提取公用
安装函数，仍使用原 snapshot_output 和 v1 marker。新 HTTP consumer 分支与
W1 公共到达变换分别审计，不将工具总 SHA 变化假定为 baseline 策略变化。

第一次只读检查因把共享 collector 要求为完全同 AST 而停止，未创建 view/
安装 overlay/启动 GPU。原检查源码、失败和实际资源释放证据全部保留。
随后114项 CPU 检查中112项在原生环境通过，2项分析专用方法因缺 ijson 产生
6个子案例错误；只将这2项转到已有分析环境核验通过，没有安装新依赖、修改
服务环境或重复性能运行。日志和对应三次辅助域关闭回执均保存。

Loader-only overlay 继续使用既有 SHA-gated 官方版本移植安装器：
`da6a14276f92186457c44c215d45d1ca9d18f7bbea9cdec6665737743f9c5978`。
启动时使用本轮独立 receipt/backups；实际 GPU 释放后仅据本轮回执恢复。
不覆盖旧环境其他文件、不转换模型、不复制工件池、不改无关 k3s。

## 收尾与解释边界

完整终态→物理释放→恢复本轮 loader→按实际 InvocationID 停止本轮远端服务→
保留 worker/调度/传输/资源日志→现有严格 analyzer `full4000_w1`→输入逐请求
join→冻结V1/单轮V2解释→备份。不能因初期排队删除请求或重定义时间起点。
所有实际异常都保留；异常结果先排除适配缺陷，不直接泛化为论文机制缺陷。

接下来仍是其余7B W1/W2、共同预算参考、3B、A1–A5、S1–S13、成本/CE与
审稿文档。D533和D536已封存且均未达95%联合门槛；不能提前宣布 G1/G2 或
PrimeLoRA 最优。接近目标前按最新要求不画图，保存原始证据和状态表。
