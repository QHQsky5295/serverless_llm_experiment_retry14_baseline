# D533：保留的 PrimeLoRA Full 工作点进入7B W1

2026-10-15，启动前登记。不是新的优化候选，不代表已取得结果。

## 问题和固定条件

检验D476的资源效率能否延续到突发负载。沿用D476的全部非路径配置、
九个公式、初始化profile和控制规律，未采纳D529退化的metadata候选。
唯一场景变换是已有W1：原4000请求分8×500、段内间隔减半、段间30秒；
原prompt、adapter、target、到达序列来源均不重建。输入资格D531的103项
检查及D534的16项入口检查按SHA链复用，不重复跑资格实验。

与D536共用D534编译器合同和D535通用sampling产物。模型相关实际JIT仍在
测量中；不新增热身、不删首段、不扣除启动。原共同60秒部署准备、真实远端
只读交付、fixed_length_greedy_v1、72/80GiB服务限制、2GiB swap、四卡峰值、
CPU集合和生命周期积分全部不变。

## 有界初态核查与解释限制

D536当前15份直接backbone缓存元数据中，仅6d1961d6a9使用本轮相同环境及
compiler hash；相对旧vLLM前缀，code/config hash相同，环境差异仅CUDA_HOME，
compiler hash亦不同。这个事后检查不能证明运行前所有文件均不存在。

PrimeLoRA的max_loras=14、max_lora_rank=16，vLLM对照为4、64。实际安装源码
将这些字段纳入LoRA hash，再纳入总配置及AOT hash，不能将D536完整模型编译
产物直接视为D533的同配置缓存。原始官方对应实现：
[LoRA配置hash](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/lora.py)、
[AOT因素](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/compilation/caching.py)。
本机源SHA另存initial_state_review.json；不据网页版本号假定本地字节一致。

图级hash不同不能排除底层kernel缓存共享。D533开始前记录实际模型缓存
元数据，运行后核查实际原生入口、编译命中/新建和时间线；不静默重置共享
缓存，也不声称双方完全冷编译。无法排除的初态差异明确保留在比较限制中。
这不阻止一次已登记的完整W1观测，但其本身不是G2最优或因果优势证据。

D536已经封存：joint72.675%、U9008.761400188087、Q95 9777.026765514165ms。
其八段joint依次60.2/73.0/58.4/63.6/78.4/75.4/75.6/96.8%，不能把全部违约
归于初始编译。候选U_ref/Q_ref在D533之前记录，不按D533胜负选择；
参考正式冻结及跨系统初态/原生合同join仍待完成，不将未知数值证明填true。

## 执行顺序

复用原runner的d533分支以及既有启动、远端健康、资源保护和精确关闭入口。
原始目录：results/ieee_tc/p2_backend_qualification/d533_20261015/7b_full_w1。
启动以start_receipt与实际进程为准；只运行一次，观察等待超时不是重启授权。

执行前显式路径备份、source/archive/SHA/秘密/保护147项检查及新鲜内存磁盘
GPU gate。运行中无其他重型任务、测试、压缩、哈希扫描、备份或远端维护。
完成后真实释放→精确关闭本项目远端→保存双边证据→既有完整analyzer与
W1原生合同核验→保存单轮指标及全部失败→返回W1/W2/G2主线。
不触及无关k3s。整个计划、3B、消融、机制、敏感性和文档图表范围不减少。
