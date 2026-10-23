# D768：S-LoRA 3B输入兼容修复与资格

## 20:14 CST结果：100请求输入/原生计量资格通过

| 项目 | 本轮实测 |
|---|---:|
| 计划/到达/提交/终态/响应 | 均100 |
| 请求失败 | 0 |
| 原生重新核验请求 | 100 |
| 实际输出tokens | 17,369，逐请求等于target |
| 实际请求adapter / 已准备静态池 | 29 / 500 |
| req_00028实际输入 / 输出 | 89 / 245 tokens |
| 联合SLO | 0/100 |
| 平均/P95 TTFT | 215,423.530882 / 379,164.422547 ms |
| 平均/P95 TPOT | 177.025816 / 229.308849 ms |
| 物理GPU时间 | 1,151.107030 GPU-s，仅资格消耗 |
| 服务内存peak | 61,243,949,056 B |
| 服务high/max/OOM/swap | 均0 |

全部原生输入、adapter绑定、输出数量、四worker来源及资源生命周期核验通过；
E2E/TPOT/native重算误差均0。实际通知后的完整池准备传输1,160,474,851线上
bytes，已物化19,482,573,296 bytes，请求打包次数0。没有未来请求预取或免费
准备，也没有因短测试SLO低而更改配置。原生调度/计算/部署策略保持。

本轮0% SLO完整保留。短前缀主要覆盖准备/启动积压，平均dispatch wait
200,943.502582ms；service TTFT和TPOT也有明显开销，不能把全部慢都归于远端。
只有100条启动期数据，不能推断完整负载的总体表现或宣布baseline算法缺陷。
后续既定覆盖/1000/4000结果决定其适用范围；不因低SLO额外搜baseline参数。

这证明D767输入兼容问题已消除，并不证明独立数值正确性：`n_correct=null`，
不进入正式资源排名。四GPU已释放、自有远端三服务已停止，原D767失败保留。
主审计SHA `35b40d242d813f4eaff3d9101e472f12caf0f2d3d0ca299c8e0240a3de11a05d`。
新小型封存位于上述目录的`gpu_prefix100/`；完整原始记录保留在RAW。

执行版本MAIN `3fd146d11af898d57e6001dfc57f4e413b2f5547`，BASE
`84bb7892b4aade6bed0b898fb99d1f6074073a10`。冻结V1/V2、D648共同SLO不变。
以下“尚未运行”文字是启动前的历史记录，不追改。

测量结束后仅回收本轮重复下载缓存：4000文件逐字节匹配保留原件，释放
19,497,332,736 B，剩余163,787,735,040 B。维护域peak2,916,139,008 B，
high/max/OOM/swap均0，空域已关闭；不与服务峰值相加。原始测量、失败及
唯一工件保留。下一D769复用本轮合格配置与57项CPU测试，做既定500-ID
功能覆盖，不重复输入修复测试，不作baseline性能搜索。

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
