# D681：3B 同 owner HOST→GPU 引用交接真实资格

## 问题与唯一变化

D680 已在提交 da30cb2f2e002bdd53ec9b163cb4e71af8fb8c07 完成候选及备份。
D678 的1438条HOST路径中，snapshot→dispatch均值141.999529ms，
其中HOST释放RPC等待均值68.906652ms，两者重叠，不可相加或直接扣减。
此次验证在真实推理中能否安全省去这次独立确认，不预先声称性能改善。
历史、原始文献/官方实现依据、失败约束与语义见 D680_HOST_LEASE_TRANSFER.md。

复用D676的100请求原始前缀、3B配置、D645 profile、D648共同SLO和所有工件。
只显式开启 resource_coordination.ieee_host_lease_transfer=same_owner_host_to_gpu_v1；
另改独占结果/NVMe/HOST路径。九公式、原生生成合同、控制阈值、容量和准备窗口不变。
当前底层生产版本为D680，不把D676前缀视作同代码性能重复。

## 资格与停止条件

- 复用D680完整CPU527项和实际vLLM环境72项原生方法测试，不重跑；
  二者重叠，不相加声称599个独立测试。原失败保留。
- 原有真实100请求检查完整保留：请求/adapter/原生token、可靠时间线、
  真实远端交付与无fallback、dispatch tier、原生GPU引用和真实释放。
- 新检查：HOST交接实际触发>0；每次回执核验相同owner/adapter/source incarnation、
  HOST lease及GPU lease，release epoch晚于acquire，GPU持有期间HOST源不提前失效。
  没触发或缺确认不能认定新机制通过；失败保留并按实际缺陷分类。
- 无新增控制策略、未来输入、阈值搜索、独立数值正确性推定或正式排名。
  这是进入完整W0前的资格，不以100条联合SLO证明4000条达标。
- 使用现有D293主入口、D676管理/关闭和D660备份逻辑。
  实际worker资源约束、GPU释放、所有owned远端服务停止后才离线分析。

## 状态与后续

准备中；没有启动GPU或远端，没有新性能结果。
启动前核验现有D680资格/源SHA、冻结输入/profile和旧147保护文件，
受限组检查配置组装、提交推送明确路径，然后唯一tmux启动。
成功后保存完整资格证据，下一为同候选一次完整3B W0；
失败不无限重跑同键。D677的90.700%仍是最新完整观测，G1/G2未达标。
完整其他场景、基线、消融、机制/敏感性及最终交付均保持待完成。

## 启动前发现的配置归属错误

首个装配检查在实际profile loader处失败，尚无GPU/remote操作；scope已关闭，
峰值497545216B、high/max/OOM/swap全0。原配置/脚本/错误与关闭回执保留。
D680开关仅由控制器读取，worker由具体RPC的release_host_lease_id决定动作，
并没有全局后端配置消费者。因此将该策略归入resource_coordination；
主组装拒绝把它放进model或instance_model_overrides，未知策略仍失败。
严格model_identity和两份D645原始测量文件完全不修改，子进程配置保持逐字段相同。
这个选择不宣称旧profile已测过新的端到端等待；它只保留共同初始化估计，
候选的实际反馈与全量服务质量仍需新运行测量。

2026-10-18查阅[vLLM官方LoRA文档](https://docs.vllm.ai/en/latest/features/lora/)：
LoRARequest按请求指定，max_loras/max_lora_rank/max_cpu_loras属于全局配置。
这一边界提供设计参考，不是官方实现包含PrimeLoRA引用协议的证明。
实际归属由本项目唯一读点和worker命令路径核查；不绕过profile检查。
针对改动补测SelectedSourceAdmissionIntegration和既有配置/原生子进程绑定；
既有底层D680测试不重复。修正后再次装配，再进行唯一GPU资格。

71项受影响测试通过（实际vLLM0.30环境、15.577s），资源事件/swap全0；
修正后真实装配与两份profile加载通过，完整子进程配置等于既有测量身份。
未生成新输入/profile，旧147文件和三用户修改SHA均不变。尚未GPU运行。

## 实际完成（后续记录，2026-10-18）

源版本69e6da68f9a0d0337a1dbbd70f0a3edee4772121已在启动前推送。
唯一tc-d681-3b-smoke100完成；100原生合同、17369输出tokens、0失败。
新同owner交接23次，均为HOST路径，lease/owner/adapter/源incarnation/
epoch/clock和release-before-dispatch全部核验；主动HOST准备148次。
独立数值正确性仍未知，n_correct=null，不能升级正式排名资格。

| 指标 | 单轮100请求观测 |
|---|---:|
| 联合SLO / TTFT / TPOT达成数 | 94 / 94 / 100 |
| 平均TTFT / P95 TTFT (ms) | 677.526472 / 1552.455366 |
| 平均TPOT / P95 TPOT (ms) | 14.240084 / 18.502474 |
| 平均E2E (ms) | 3191.197322 |
| 物理GPU-s | 785.667071 |
| 远端fetch / 线上字节 | 29 / 67350156 |
| 请求中打包 | 0 |

分层remote29/GPU46/HOST23/NVMe2；共同时间分解通过1ms门槛。
服务261资源样本，峰值20470063104B，high/max/OOM/swap全0。
真实GPU释放、服务/外置域关闭，三个owned远端服务inactive且端口关闭。
远端实际1000Mb/s/full，复用D78/D80发布缓存，不重建工件或trace。

第一次收尾collector已完成，但分析误选缺ijson的既有环境而中止；保留
analysis1.log及analysis2回执/关闭。只换到已有vLLM环境离线读取同一结果，
不安装依赖、不重复下载/推理；analysis3通过并关闭，峰值138104832B、事件swap0。
原始结果SHA5144b0b8a94a56316de79584d335c012c9025b9415bdd63cae5c3859049faea6，
分析SHA1f42a9bab7ccfafc7a67b951af12788115f9acef673b402ded2f7565a671cfd4。

这证明新路径在真实短前缀运行可用，不证明完整W0联合SLO达标或性能领先。
94%也不触发改阈值或同键重跑。下一同候选唯一完整3B W0，再判断收益/退化。
