# D663：受保护的原生 HOST 激活内复用存储事实

## 问题与范围

当前主线仍是 3B PrimeLoRA W0 优化。D660 完整回放的联合 SLO 为
3575/4000（89.375%），U=15902.267275708029 GPU-s；尚不满足冻结 G1/G2。
D662 已复用全部保留回执：1441 条 HOST 路径中，1426 条最后实际加载均值
15.680ms，acquired→post-snapshot 均值83.012ms。后一段还包含其他工作，
不能把83ms全部算作可删除的遍历时间，更不能从旧 TTFT 中直接扣除。

候选假设：同步、受保护、已注册 HOST 激活期间没有 CPU 权重存储变更，
第二次完整 HOST 存储遍历可由本事务第一次的事实替代；仍重新读取 allocator
占用和 workspace。减少核心线程上的重复检查可能减少请求等待及 decode
干扰。该假设须由真实原生路径和完整回放验证，CPU 测试不证明性能收益。

## 历史与官方实现依据

- D329 证明相同 adapter/model ID 或 owner epoch 不足以证明存储、alias、
  pin 状态不变；禁止跨请求缓存，本轮保留并重跑这六个反例。
- D352 已限定复用到 confirmed GPU-hit 无加载事务；本轮只扩展到有独立
  路径证明的 HOST 激活，不能将原先明确排除 HOST 的证明直接套用。
- D333 的 pending ACK→计数更新→HOST 保护顺序不动；不合并这些 RPC。
- D595/D628 的组件微测收益曾未转化为完整回放收益；不重做已撤销方案，
  也不因本轮少一次遍历就接受候选。
- [vLLM v0.30 worker_manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  的已注册、non-inplace 分支只获取已有 adapter，然后激活，不再加载/注册 CPU 权重。
- [vLLM v0.30 model_manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
  的 LRU 激活可以回收 GPU slot，但不需要移除 CPU 注册对象；逐层将已有权重复制到 slot。
- 本机实际安装源码也已核查；本候选复用现有 `_ieee_host_copy_contract`，
  限制为已审计 dense setter/resetter、TP=1、既有 packed 布局及相同 dtype，
  不扩大到 MoE、转换、packed expansion 或任意外部 loader。

## 实现与不变量

1. 每个新事务首先完整测量所有 registered/staged HOST 存储、alias、pin 和
   allocator 占用，并执行原有预算检查。没有 epoch/TTL 缓存。
2. 仅当前确为 registered HOST、不是 GPU-hit/staged、调用方提供有效 source
   incarnation，且相同 source 有活跃 demand-preparation HOST lease 时适用。
3. 新 callback 验证原生 loader/manager 类型和方法绑定、固定复制路径，记录
   registered/staged/module 成员身份。在同一同步 owner 调用中加载、fence、
   acquire 后再次核对。身份一致本身不是物理不变证明，已审计的只读路径才是。
4. 后检查复用本次首次存储事实，但 allocator 及 workspace 重新读取。增长
   仍可拒绝；后检查异常仍按既有语义使 owner 失效，不能发布成功、重试掩盖
   或虚构 rollback。
5. 不符合适用域的路径保留原来两次完整检查；资格 callback 本身报错则显式
   失败，不捕获并静默降级。每个新请求仍重新验证，不缓存前一请求的证明。
6. 回执标记 `same_transaction_registered_host_activation_v1`，与现有 GPU-hit
   标记和 `fresh_inventory` 分开；后续原生验收必须检查该标记的触发域。

九公式、queue/pending 次序、routing、handoff、planner/admission、completion
fence、资源计量、SLO 和所有3B非目标配置均不改变。无人工新增性能阈值。
一次源代码检查在测试前发现插入位置错误，已移至 `_load_and_acquire` 正确
前后检查间；没有以错误版本启动模型或实验。

## 本轮验证与安全

- 定向21项通过，覆盖真实方法绑定、事务身份、每请求新鲜性、前后失败、
  transport replay、CPU/GPU共享引用和动态allocator增长。
- 既有 native reference、worker observation、transfer pressure、request
  lifecycle 四套共619项通过；冷加载、staging、unconfirmed路径均保留原语义。
- D329六项反例通过；该模块还导入九项既有 footprint 测试，unittest一并执行。
  因而总执行次数21+619+15=655，其中九项重复，不称655个独立新测试。
- 三次辅助检查在3/4GiB、swap=0与固定辅助CPU集合中运行；峰值分别
  637083648、659873792、632664064字节，high/max/OOM/kill/swap事件均为0，
  已递归确认空并关闭。147个旧投稿保护文件及三项用户修改SHA不变。
- 无GPU运行、无远端服务变更、无全局k3s改动、无图表、新权重或新负载。
  用户确认现有集群非本项目；继续复用已完成的HydraServe隔离适配和7B证据，
  不重启基础适配工作。

原始证据：`results/ieee_tc/p2_backend_qualification/d663_20261018/`。
curated：`paper_results/ieee_tc/p2_backend/d663_registered_host_transaction/`。
github-sync按显式路径备份生产候选、测试和小型证据；不stage用户manifest。

## 下一步与停止条件

复用D659/D660管理入口，先运行一次新版本3B原W0前100请求原生资格；仅改变
版本及输出路径，原工件/trace/生成/资源/3B配置和D648阈值不动。要求原生
绑定/数量/计时完整、资源实际释放、至少一次真实HOST事务触发且全部触发
均满足上述域。任一路径错误先分析，不用兜底或增加超时掩盖。

通过后以同非目标配置运行一次完整3B W0，再与保留D660作描述性比较；不得
以100请求或测试通过判定完整达标。若未改善，保留并分析/撤销候选，不盲调
SLO、不重复同键筛选赢家。独立数值正确性未知仍保持未知。

3B W1/W2、各基线/dLoRA、同版本7B主表与消融、A1–A5、S1–S13、成本/CE、
两轮审稿文档与最终图表全部保留；当前不因完成CPU验证跳过3B W0目标。
