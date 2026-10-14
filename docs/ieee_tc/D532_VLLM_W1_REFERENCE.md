# D532：vLLM 7B W1 单轮共同资源参考候选

2026-10-15，运行前登记。复用 D423 的冻结 vLLM0.30、DP4/TP1、原序号 RR、
原生 LoRA cache、FP16、max-seqs2/max-loras4、GPU fraction0.7、chunked
prefill、非eager图执行及无prefix cache。不调优基线，不改变模型/权重/生成。

## 问题与输入

测量八阶段突发到达下，常驻共同后端的完整生命周期 GPU-s 和 P95 TTFT，
为该场景的 U_ref/Q_ref 提供实际候选。不能从 W0 或短回放线性推算；只有
完整合同、真实资源释放和计量核验后才冻结参考。数值正确性边界继续保留。

D531 的103项检查与实际输入核验复用，不重复短GPU资格。完整4000请求、
132实际LoRA ID、458224目标token，W1窗口2186.9921293258667秒。
源 SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`；
W1 view SHA `103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3`。
请求/工件仍是既有 seed42；保留 development41、qualification_only=true、
formal_run=false。不是新独立trace或留出评估。

原始目录：`results/ieee_tc/vllm_qualification/d532_20261015/`。
新按需缓存 `/tmp/tc-d532-7bv`，复用既有编译磁盘缓存并记录来源，不预载
引擎或按未来请求预取。W1只改变到达索引；阶段间不等drain、不清合法缓存。

## 共同合同及安全

部署通知在业务前60秒，准备持卡计费，1800秒既定开发保护；fixed-output
greedy、相同 canonical prompt 和原生 token 计量不变。真实远端仍是 D78/D80
统一发布的只读gzip；无请求时打包，必要读取、传输、竞争与本地物化保留。
远端服务/监控复用已验证配置，按身份管理，不碰共享k3s。

服务high72/max80GiB、swap2GiB、CPU4–23/28–47；回放监控3/4GiB、swap0、
CPU2/3/26/27。预计新增峰值8GiB；推理机启动150GiB、运行100GiB及内存
保护不变。实际worker启动前继承限制，单重型任务，性能期间不维护远端、
安装/构建、压缩、扫描或备份。远端独立容量门槛不套用推理机150GiB。

## 执行、分析和边界

复用 D423 run7b.sh/native-http-service 和 D529 远端、身份、释放管理流程；
代码检查点已由D531推送，本轮只有新执行键及小型管理脚本，无服务代码修改。
`start.py`冻结实际两个仓库commit、输入和编译缓存，完成远端健康后在tmux
启动一次；观察超时只检查原进程，绝不因此重复启动。

完成后：真实GPU释放→exact-owned本地辅助域关闭→同Invocation远端停止→
双端传输证据保存→既有分析器 `resident_rr_v1/full4000_w1` →封存/备份。
共同D394 SLO和V1/单轮V2不变；失败、输出差异、未达标和未知数值证明均保留。
不按胜负追加运行，不输出CI，不提前出图或宣布PrimeLoRA领先。

下一步是已经准备的 D533 PrimeLoRA FullW1（D476策略/原profile不变），再
继续其余已发表系统、W2/G2、3B与完整机制/敏感性。一次参考不代表7B闭口。
本文件创建时仅准备，实际启动/完成以本轮receipt和EXECUTION_STATUS为准。
