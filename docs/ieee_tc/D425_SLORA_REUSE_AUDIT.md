# D425：S-LoRA 7B 历史证据复用决定

2026-10-11，离线完成；没有 GPU/远端操作或运行代码修改。

按 `rg --files results -g '*slora*replay.json'` 且路径包含 `7b`，核对 25 份
回放、230 份支持文件。17 份完整 4,000 请求记录没有原生计数固定长度合同；
五份固定输出检查分别为 100/85、32/32、100/100、100/100、100/100（请求数/
长度匹配数）。另三份 500 请求开发记录 success 为 395、396、500，均保留。
实际 adapter ID 最大 132，未找到 500-ID 请求覆盖。全部 replay server_metrics
为空；所查日志和源码不足以恢复原生首末 token 时间及完整物理 GPU leases。

既有 `slora_official_cu118`、原生 SSE 整数计数、FIFO 和 500-ID 工件池继续复用。
当前嵌套 HEAD `c1ddf488781ea7f551cd0bb07bfd097124c93411`，10 个既有兼容文件
与三份 7 月 22 日 sidecar 全部字节一致。patch 解压等于当前 binary diff：
`7a0f48f60c225789014ad960ec3ab1cf6388611b7650a4f656f1208c981c2d60`；
full-index 格式对应历史 `c62bb51e7ec316e19b60f68807b7eba0fdfe17cdd8dfe67ab7147807a3d64b89`。
原生 `forward` AST 未变；API 文件与官方固定版本逐字核对通过。保留全部源码
修改，不宣称已完成独立数值验证。已有生成/FIFO CPU 回归 9 项通过（0.073s），
实际日志 retained，不把它作为新 GPU 资格。

下一步复用现有共同计量/启动组件，显式新增 S-LoRA 原生事件和 physical-owner
集成。事件在实际采样 token 可用后、文本/HTTP 之前记录；不能用 SSE 到达或
完成尾部代替。保留原生全池 HOST 准备、调度、paging、batching、load/offload；
准备从共同部署通知开始，业务不随 ready 平移。旧 wrapper 顺序启动/ready 后
回放以及按端口清理、旧目录删除入口不能直接用于新合同。

当前为 R2 资产和诊断复用；计数/FIFO/源版本的有效局部证据保留，不使旧完整
运行自动成为 G1/G2 比较点。受影响 100 请求路径、缺失的 500-ID/1,000 兼容
资格及一次完整 W0 仍待实施。PrimeLoRA、ServerlessLLM、vLLM 封存点不重跑。

完整报告在主仓库 `docs/ieee_tc/D425_SLORA_REUSE_AUDIT.md`。
主仓库 curated inventory SHA
`95f471cb65cf5646b5a545221dceae5afdcb3f1fddf341582b5b014e70f71161`；
source evidence SHA
`bb673cafeae78ba5f3de5c901a4477ff9147c2521132079c3722f7f9b80420c4`。
共 276 来源引用核对、147 历史保护项零变化。用户 dirty replay/relayserve 未改。
原始 CPU recipe/testlog 在 main experiment results 的
`ieee_tc/slora_qualification/d425_20261011/`。正式性能与数值资格尚未完成。
