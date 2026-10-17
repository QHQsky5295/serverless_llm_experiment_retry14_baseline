# D664：3B原生HOST事务优化资格验证

继续D663候选，不引入第二项优化。按完整PLAN、冻结V1/单轮V2及最新执行
补充，复用D659原W0前100请求及全部3B配置，只更换新代码版本与唯一结果
路径；保持真实远端只读预发布交付。已有655次CPU检查复用，不重复整套测试。

本轮用原生加载回执检查D663 marker至少实际触发一次，且全部触发均来自
受保护registered HOST、匹配source incarnation/adapter/owner，实际调用
native loader并保持fence/引用释放。所有事务的前后HOST容量检查必须成功。
GPU-hit、冷加载等标记分别核验；不把marker数量当作性能改善或独立数值证明。

共同原生生成、绑定、计划到达计时、分解恒等式、物理GPU账目、远端传输与
资源安全均复用既有analyzer及管理脚本。source/config/脚本与保护清单SHA
在启动前记录。现有k3s和已完成HydraServe适配不变。

当前状态：准备启动资格验证，尚无GPU结果。短回放不是4000请求G1/G2资格。
通过后运行同非目标配置完整3B W0，与保留D660作描述性比较；不换SLO、重复
同键挑胜者或用短样本决定完整性能。真实路径错误则保留失败并限定诊断。

原始：`results/ieee_tc/p2_backend_qualification/d664_20261018/`。
curated：`paper_results/ieee_tc/p2_backend/d664_3b_full_smoke100/`。
3B后续场景/基线/dLoRA、同版本7B与消融、A1–A5/S1–S13、成本/审稿/图表均保留。
