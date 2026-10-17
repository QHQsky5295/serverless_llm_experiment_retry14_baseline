# D658：当前 3B Full 入口的实测配置绑定

2026-10-18；这是 CPU 配置资格，不是新的性能结果。六项检查通过，尚未运行
D659 100 请求，也未宣称 3B 完整负载、数值正确性或 G1/G2 达标。

## 问题与改动范围

D640/D643/D645 已证明旧 D138 只能复用结构与历史诊断；D641 的单副本配置
不能直接当作弹性 Full。复用 prepare_d298_capacity.py 和现有主启动入口，
将 D643 的实际 child model、D645 实测服务/准备 profile 绑定到当前 Full。
不改 IEEE 九个公式，也不修改运行时算法源码。本点是迁移候选，不是单因素消融。

3B 继续使用 8 路并发、max LoRA rank 16、min=1/max=4、既有扩缩容窗口和
阈值；未复制 7B 的 14 路/min=2 配置。路由、handoff、hierarchy、admission
均采用现行 Full 路径；采用当前已验证的 observed-arrivals 初始控制、CPU
布局与状态快照实现。在线 5000 ms 控制阈值不等于 D648 最终共同 SLO。

历史依据为 D138、D640、D643、D645 与当前主入口的修改记录；本轮重新核查
[vLLM 0.30 优化说明](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/)
和 [LoRA 配置说明](https://docs.vllm.ai/en/v0.30.0/features/lora/)。这些文档
支持按实测 token/内存配置验证，不证明更大 batch 必然更优。rank16 对应既有
3B rank8/16 工件上限，chunked-prefill 与 8 路是已测绑定，不新增参数搜索。

## 已完成验证

- 六项 CPU 检查：实际 child 身份、输入不变性、100/4000 仅人口差异、异模型
  拒绝、未测容量/副本下界拒绝、两类真实 profile loader 及错误身份拒绝。
- 历史 shell 分支移除 D659 新增段后与前一提交逐字节相同；现有生产路径 SHA
  记录在 qualification.json，未重跑 368 个源用例或 89 项后端资格。
- D658 资源域 peak=716099584 B，high/max/OOM/kill/swap=0，递归空后关闭。
- 147 个旧投稿文件未变，三处用户修改保持原 SHA。

输入仍为既有 3B 4000 请求源的前100条，W0，不新生成 trace/权重。D659 使用
新结果/NVMe/HOST 目录、真实远端 18080、已发布 gzip 交付以及共同资源包络。
D648 阈值与 D657 三场景资源参考不变。开发身份与 n_correct 未知保留，不把
原生绑定/数量检查冒称独立数值证明。

## 下一项和停止边界

先备份此检查点，再执行 D659 一次 100 请求 Full 资格；核对完整请求身份、
原生输出数量/计时、dispatch tier、真实 GPU 分配/释放及远端内容。资格通过
后继续完整 W0/W1/W2。性能不足时依据新阶段证据优化，不调评价阈值、不重复
同键筛选赢家，不对短前缀外推完整预算。其余基线、消融和敏感性仍在主线中。

证据：paper_results/ieee_tc/p2_backend/d658_3b_full_entry/preparation.json；
execution_evidence.tar.gz 保留资格与下一运行管理入口，result.sha256 核验。
