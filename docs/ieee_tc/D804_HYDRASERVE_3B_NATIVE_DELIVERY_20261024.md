# D804：HydraServe 3B 原生交付路径的受影响检查

2026-10-24。不是性能排名或参数搜索；复用既有隔离平台，不修改其他项目k3s。

## 问题与必要范围

D803在本地checkpoint路径的prefill/两步paged decode共6个检查点通过，
但没有覆盖原生CPU provider→私有共享内存→worker。因此不能直接把数值
通过转授给真实加载路径。D804仅沿D576已有入口，将模型注册、metadata和
已验证模型overlay切换到实际3B；复用原trace前100请求，不生成新负载。
这是受影响模型路径检查，不重开隔离平台、安全协议或baseline优化。

## 固定条件

| 项目 | 实际设置 |
|---|---|
| 模型 | 既有 Llama-3.2-3B-Instruct，原BF16 checkpoint不变，原生provider/worker一致FP16 |
| 输入 | 既有seed42 trace前100请求，固定输出合同，原prompt/adapter/target |
| 工件 | 既有500-ID池、D80只读gzip发布，真实HTTP 18080，无本地fallback |
| 调度 | 原生Scheduler/algorithms、batch8、原30秒idle、无参数搜索 |
| 原生模型注册 | `[6,9000,200,1000000,1]`，只派生缺失的3B大小 |
| 资源 | 共同72/80GiB服务域、2GiB swap上限及既有外置watchdog |
| 平台 | 已验证rootless Podman与原生专用后端；不访问其他项目k3s |

ModelInfo的字段语义由[官方源码](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/kubernetes/vllm/src/ModelInfo.py)核对。
3B大小为现有索引unique tensor字节除2**30向上取整；其余字段取原test配置，
不复制7B的13GiB大小，也不依性能结果挑数值。这里的在线目标不是冻结共同
TTFT/TPOT评价阈值。原[provider的共享权重分派](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/kubernetes/vllm/storage_server/remote_server.py)
继续使用；D803元数据直接链接复用，没有第二套tensor池。

## 预先约定的核验与局限

1. 100个原生请求绑定、实际token数量、prompt/adapter和真实远端内容一致。
2. 实际provider/private-SHM/native worker完成加载，真实资源和释放可追踪。
3. req_00001的前三原生token与D803独立参考检查通过的相同输入前缀对应；
   其余输出保留，不把前缀一致扩大成全部数值证明。
4. pipeline/consolidation记录实际触发；未触发不声称合格，不另行搜索部署。

停止条件：受影响路径通过后进入完整W0/W1/W2，不再补重复100/1000回放。
若存在实际加载或计量错误，保留失败并只处理具体错误，不用超时/阈值兜底。
当前只完成准备并启动，结果待实际终态；本文不提前报告成功。

原始位置：`results/ieee_tc/hydraserve_qualification/d804_20261024`（主实验
共享结果根下），preregistration保存输入/模板/代码身份。D803已经分别备份
MAIN be60ae1和BASE235b1e8；本运行期间不push、不维护远端、不做磁盘扫描。

## 实际结果：配置初始化失败，不是性能观测

| 事实 | 实测值 |
|---|---:|
| 已登记输入合同 | 100 |
| 实际到达／响应 | 0／0 |
| GPU worker启动 | 0 |
| 服务内存峰值 | 7,208,349,696 B |
| high/max/OOM/swap | 全0 |
| 本机服务与辅助资源释放 | 完成 |
| 174两制品服务及监测停止 | 完成 |

provider的SchedulerConfig在模型加载前拒绝了4194304的上下文长度。
源模型已声明131072；旧后端再次乘RoPE factor32造成错误。即使只修正
这个乘法，131072仍大于原provider的65536批量token上限，因此不能只消除
第一条报错而忽略原有上下文约束的生效位置。

[vLLM官方实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3.post1/vllm/config.py)
明确排除对llama3的这次长度乘法；
[HydraServe原引擎](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/vllm/engine/llm_engine.py)
本来会按MAX_MODEL_LEN（默认4096）约束实际模型，只是在配置校验之后
才赋值。必要修正从原源码AST提取这一已有表达式，提前传给provider和
三个app初始化入口；不新造context数值，不增大token预算，不改原位置编码
频率或输入。仅显式3B模型opt-in启用；旧默认overlay不变。

34项CPU回归0.998秒通过，覆盖长度推导、旧缩放分支、越界拒绝、环境值
与原默认值，以及四处EngineArgs契约。它不代替下一次实际provider验证。
D803原生数学的6个通过点保持原证据身份，不将新源码追记为其执行版本。

远端原stop已核对InvocationID并成功停止三服务，但原collector假定存在7B
传输日志，因此在0到达的情况下返回1。后置只读核验确认本次两个clock_id
均无transfer记录、服务inactive/PID0、端口消失；监测文件复制SHA一致。
保存该collector失败，不伪造空“成功传输”日志，不改原运行记录。

失败审计在failure_audit.json；原source_manifest的快照全部校验，当前修正
源码单独提交。n_correct未知、performance_result=false，不能据此给HydraServe
填入延迟或资源效率排名。后续只复核受影响路径，然后恢复完整3B矩阵。
