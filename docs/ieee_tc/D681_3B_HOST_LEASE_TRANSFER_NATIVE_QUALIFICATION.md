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
