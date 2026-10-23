# D768：S-LoRA 3B输入兼容修复与资格

## 问题、证据及边界

D767中req_00028的共同输入89 tokens，原生旧tokenizer切为91。
同100条CPU复现只有这一条不同，首差异索引34。旧失败与真实资源消耗已
封存于D767，不当作性能主结果。该具体输入错误允许必要适配，不授权baseline
调参；保持共同负载、目标长度、D648 SLO、原生调度和LoRA计算。

新显式`canonical_token_ids_v1`把公共准备器的原生ID送入S-LoRA现有HTTP→
router边界。默认路径仍为旧`native_text_v1`；没有自动fallback。输入文本/
ID的SHA、整数/范围、原生长度限制及实际forward输入/adapter/输出核验保留。
只在新view增加HTTP manager输入适配，原生checkout、旧view和旧结果不变。
详见baseline仓库`docs/ieee_tc/D768_SLORA_CANONICAL_INPUT_BOUNDARY.md`。

CPU57测试通过；其中新增测试覆盖实际API/manager/外层服务接线及AST范围。
两个新fixture的首轮错误已保存并修正，非GPU实验失败。首次启动器scope命名
两次未被既有身份规则接受，均在测试/模型启动前退出、确认空域后关闭；没有
放松身份规则。实际资格使用既有UUID资源域：3/4GiB、swap0、CPU2/3/26/27。
成功测试peak349380608B、high/max/OOM/swap0，空域关闭。

## 执行身份

RAW：`results/ieee_tc/slora_qualification/d768_20261023`（主项目旧结果根）。
仍为3B GQA/BMM、native HOST、四TP1 worker、RR、顺序启动、真实远端完整
500池合法准备、既有100请求视图。源trace SHA
`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`；
view SHA `7fc7c1a556fbf8d161ef607684416b51d7d917eff53a7c5e24ab4088d563f598`。
实际29 IDs、17369 targets；不生成新数据/权重、不移出真实准备成本。

资格脚本/配置/来源及失败测试归档在
`paper_results/ieee_tc/baseline_audit/d768_slora_canonical_input/`。
先备份再启动GPU；CPU合格不等于GPU、完整性能或独立数值正确性合格。

## 返回主线

本检查点尚无D768 GPU结果。下一新键验证同100条受影响路径，再继续缺失
资格和S-LoRA 3B W0/W1/W2；ServerlessLLM和vLLM已完成点复用。
既有HydraServe隔离环境保留，不操作其他项目k3s。3B其余baseline/dLoRA、
PrimeLoRA跨场景与必要优化、最终同版本7B主表/Full、A1–A5、S1–S13、
CE/Pareto、两轮审稿文档与最终图表均未删除；目标active，当前不画图。
