# D494–D495：原生 7B 传输、GPU 加载与服务初始化

2026-10-13。这是必要适配后的功能资格证据，不是性能比较或完整复现结论。
现有其他项目 k3s 未访问、未修改。沿用 D493 独立 rootless Podman 接口，
单个真实物理节点，不把四张 GPU 虚构为四个计算节点。

## 保留的原生路径与必要适配

保留官方修改版 vLLM 0.4.2、CPU 模型 packing、分段、双线程 TCP 传输、
private shared-memory local_server、C++ fastload 及 app 启动重叠。
原 ResourceManager 仍选择实际 GPU、创建、等待真实 `/init` 和确认删除。

- 现有 7B 有两个 safetensors 分片。只读迭代原文件，逐 tensor 核对 index；
  不生成合并权重，不下载/重新划分 checkpoint，不改唯一原文件。
- 只生成小型 config/partition/duplicate 映射和原 tokenizer/权重链接。
  分段来自源代码固定的原生 get_layer_distribution。当前限定已核验 7B；
  不假定 3B 的 tied-embedding 语义已经通过。
- provider 显式注册当前 7B，避免原默认模型列表加载未获授权的 13B。
  唯一用到的 AutoConfig 直接来自已安装 Transformers，移除未用 ModelScope imports。
- app 显式启用原生 LoRA，现有工件无附加词表，因此 extra_vocab=0。
  D491 原生共享 LoRA manager 绑定继续复用；不改变 LRU/rank/batch-sequence defaults。
- stdlib-only SHA 绑定 import overlay，不在原 app 的启动加载前额外导入 torch。
  容器放行前实际查询可见 GPU UUID，不能只凭环境变量判断 GPU 隔离。

## D494 失败与 D495 最小修正

D494 完整 CPU provider、13 GiB 私有共享内存、C++ GPU 权重加载均已实际运行，
但 `/init` 返回 500：官方 SchedulerConfig 默认 `16*4096=65536`，而同一
源码的 LoRAConfig 要求不超过 `65528`。不是数据、OOM 或 SLO 不达标。

D495 仅把启用 LoRA 的三个 EngineArgs 入口限制在该原生内核上限；值直接从
固定源码 AST 提取，不作性能参数搜索，不关闭校验。CPU provider 无 LoRA，
保持原默认值。官方 [config.py](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/vllm/config.py)
已重新联网获取，SHA `5a17e5ca3741314175daa825024172be0decb56dd411d6e40ff146cfe15d8332`
与本地输入一致。D494 的原 overlay、执行源码、错误与退出证据保留，不覆盖。

D495 实际通过：CPU provider → 原生双线程传输 → 私有 SHM → C++ GPU fastload
→ 原生 Llama/ModelRunner/LoRA manager → KV 初始化 → `/init` 200、`/health` 200
与 `/v1/models`。完整序列化流长 13,476,855,675 B；既有两分片总文件大小
13,476,876,272 B，差异来自原生 packing/header，不能以文件大小差异断言丢权重。
初始化通过也不替代后续 token/LoRA 数值正确性。

## 资源与限制

两次均使用共同 high72/max80 GiB、swap2 GiB、CPU4–23/28–47 和独立外部监控。
provider、storage、worker 合计受父限额约束，不把每容器80 GiB相加成额外配额。
13 GiB池按既有权重字节向上取整，local_server 原生完整触页；不是免费 HOST 缓存。

| 运行 | 结果 | 服务内存峰值 B | high/max/OOM/swap |
|---|---|---:|---|
| D494 | native LoRA 配置边界冲突 | 42,040,569,856 | 全零 |
| D495 | 真实 app 初始化通过 | 42,318,667,776 | 全零 |

两次 worker/provider/storage/pause 均退出，GPU 释放，服务域实际消失，aux 域
核对 InvocationID/inode/递归空组后关闭。运行没有访问远端174工件服务；这一步
只验证按计划允许的本地 backbone 路径，不冒称真实远程 LoRA 实验。

仍需处理的原生接口事实：app 的启动 asyncio.run 关闭旧 loop，随后
end_load_model_shm 打印 `Event loop is closed`；本次 storage 在 worker 删除前后
才记录 free_buffer。不可把本轮退出后的释放当成稳态及时归还证明。继续核查
连接所有权和反复扩容/并发加载是否受阻，必要时做协议不变的正确性适配。
注册名没有启用 chat template；健康与模型登记不证明 ChatEngine 请求成功。
共同 canonical prompt 应通过必要输入适配保持不变，不能用新 chat 包装改变工作量。

## 下一主线

先完成原生请求层 LoRA 身份、固定输出与原生 token/时间合同；保留 scheduler、
pipeline、consolidation 和资源生命周期。当前源码 Request/ChatEngine 只有
prompt/model，并固定 temperature=1、max_tokens=1024，尚不满足共同生成合同。
原生 vLLM completion 已有 LoRARequest/token_ids，可接通但不能绕开 HydraServe
控制器冒名运行。随后100→500-ID→1000→4000，再比较全部可用 baseline，
才恢复 PrimeLoRA 优化。分段/consolidation/KV migration 的完整请求正确性仍未闭口。

原始证据在原实验仓库 results/ieee_tc/hydraserve_qualification/d494_20261013 与
d495_20261013；本仓库 paper_results/ieee_tc/hydraserve/d495 收录来源清单、
精选失败/成功记录及 SHA 核验包。其余 W1/W2/G2、3B、B6、A1–A5、S1–S13、
离线成本、文档与最终图表范围不变。本轮没有新增性能排名或领先结论。
