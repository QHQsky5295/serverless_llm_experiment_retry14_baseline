# HydraServe：历史资格结论的只读复核

日期：2026-10-11。不修改原 gate，不执行 GPU、容器启动、安装或集群管理。
主线仍是正在运行的 D435 S-LoRA 完整 W0；Loquetier、HydraServe 的实际资格
必须在其收尾、共同对比和安全检查之后依序进行。本记录不是资格通过或性能结果。

## 1. 本次修正的判断

1. 保留官方修改后的 vLLM 0.4.2；旧版本本身不是排除理由，也不替换成
   PrimeLoRA 的 vLLM 后端来冒称复现 HydraServe。
2. 历史“请求路径不传 adapter，所以一定要更改调度语义”的推论过强。
   请求身份可以作为不参与调度决策的字段传播；是否需改变核心算法，须逐条
   检查实际 worker、分层权重和合并后的执行，不能仅由缺少字段推断。
3. 但也不能把“vLLM 有 LoRA 参数”当完整支持证据。本轮发现需进一步验证
   pipeline 分层编号及 consolidation 目标模型的 LoRA 管理绑定；仅补一个
   请求字段或开启 enable_lora 不足以证明所有阶段正确。
4. 历史平台障碍部分仍有证据，部分必须更新：当前 k3s、Docker、containerd
   服务为 active，podman/k3s 可执行且 qhq 有 subuid/subgid。不能继续写成
   本机没有容器/Kubernetes；也不能把服务存在等同于用户有可用 GPU 部署权限。
5. 尚无完整相同负载结果可复用；保留旧控制模块导入、源码审查和失败证据。
   既不提前判定完整不可复现，也不标成已合格主表行。

## 2. 复用身份

- 官方仓库：`https://github.com/LLMServe/hydraserve`。
- 本地：`vendor_new_baselines/HydraServe_main_20260520`。
- 本轮 git HEAD：`8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`；git status无修改。
- 既有控制环境：`/home/qhq/anaconda3/envs/hydraserve_py_20260520`。
- 历史记录：`HydraServe_project/{README.md,docs/GATE_2026-05-20.md,evidence/gate_2026-05-20.json}`。
- 未重新导入运行模块、下载镜像、编译后端、生成模型分片或扫描工件池。

## 3. 直接源码证据与待验证问题

路径相对上述 checkout；本次读取源文本，不代表执行过以下分支。

| 路径/位置 | 本次确认的事实 | 资格含义 |
|---|---|---|
| `scripts/kubernetes/vllm/src/main.py:RequestModel` | 只有 id/model/prompt/stream | adapter、原生输入与固定输出配置未端到端传递 |
| `src/scheduler.py:Request/Instance.chat` | 以 model_id 排队，调用 ChatEngine；chat 有5秒等待 | 需区分调度键与 adapter 元数据；共同超时适配须显式记录，不改实例选择规则 |
| `src/engine.py:ChatEngine.chat` | 固定 model=self.model_id、max_tokens=1024、temperature=1；非流式只返回文本 | 不满足 fixed_length_greedy_v1 或原生计时合同 |
| `scripts/image/app.py:init_model/start_load_dest` | AsyncEngineArgs/EngineArgs 未接 enable_lora；OpenAI serving 未接静态 LoRA 注册 | 请求 adapter 名称必须与实际 worker 注册对应，不能静默基座 fallback |
| `vllm/executor/gpu_executor.py:init_dest_model_worker` | 修改目标层数并调用现有 runner.load_dest_model | 要检查旧/新模型共存与实际 GPU 占用，不能只按控制器副本数计费 |
| `vllm/worker/model_runner.py:load_model/load_dest_model/execute_model` | 初始 load_model 建 LoRA manager；load_dest_model 没有相同注册；execute_model 对一个 manager 设 active 后按use_dest选模型 | 目标新增层、已有共享层与 LoRA slot 的绑定尚未证实，不能只测合并前 |
| `vllm/model_executor/models/llama.py:LlamaModel/check_duplicate` | pipeline 用环境中的 rank/size、本地 ModuleList；consolidation 依映射复用部分 decoder layers | 要核对 LoRA 全局层名与各阶段本地层号，不能假定完整PEFT文件直接加载即可 |
| `vllm/lora/models.py:from_local_checkpoint` | 存在原生 LoRA tensor/parser 路径 | 支持进一步忠实适配审查，不等于已验证上述自定义 pipeline |
| `scripts/kubernetes/vllm/src/utils.py:create_deployment` | GPU-memory资源请求；宿主 /dev/shm；按GPU内存派生Pod memory limit | 必须让所有实际Pod/共享内存服从共同总资源与CPU/swap协议，CLI受限不够 |

其中 `src/...` 在表中均指 `scripts/kubernetes/vllm/src/...`，不是仓库根src。
对目标 LoRA manager 的判断是“注册路径缺口需要验证”，不是已经观察到错误
输出；共享部分层可能继承引用，必须分已复用层和新增层验证后再作结论。

## 4. 当前平台元数据（23:27–23:28，只读）

- `/var/run/docker.sock`：root:docker，0660；当前 qhq 不在 docker 组，
  本次 test -r/-w均未通过。未连接daemon、未停止任何既有服务。
- `/usr/local/bin/kubectl`、`helm`：root:root，0700；当前用户不可执行。
- `/usr/bin/podman` 和 `/usr/local/bin/k3s`：root:root，0755。
- `systemctl is-active k3s/docker/containerd` 三项均 active。
- `KUBECONFIG/DOCKER_HOST/CONTAINER_HOST` 未设置；默认 `~/.kube/config`
  未发现。没有读取或打印集群凭据、令牌、私钥。
- `/etc/subuid`、`/etc/subgid` 均有 qhq 的65536范围。

这只说明有进一步检查 rootless/已部署集群可用性的依据，不证明 GPU runtime、
资源限制、调度插件、用户认证或集群归属已合格。不得改动未知现有集群或全局
网络/驱动来试运行。若实际步骤需要新增管理员协调，明确报告那一步的边界。

## 5. 进入实际资格时的约束

优先检查现有官方路径能否在共同安全包络内运行，复用原生后台与已有本地基座，
不重新下载模型/生成负载。接口适配和模型注册可以做；pipeline、consolidation、
启动重叠及原调度决策不替换。分别证明每请求 LoRA 在分层与合并前后正确执行，
再推进100/500-ID/1000/4000资格。失败、未触发、无法计量如实保留；不搜索
baseline性能配置、不重复同键求胜。共同真实远端仅放LoRA，本地基座场景可能
减少其原论文针对基座冷加载的收益，必须明确实验范围，不推广为原论文无效。

若最后确认官方路径不能忠实运行，按原计划继续核查 ServerlessLoRA、BlitzScale，
保留 HydraServe 资格失败原因，不用裸vLLM或自制调度框架冒名替代。

官方定位与机制核查：[仓库](https://github.com/LLMServe/hydraserve)、
[NSDI 2026论文入口](https://www.usenix.org/conference/nsdi26/presentation/lou)
（2026-10-11查阅）。论文强调分层模型获取、准备重叠和pipeline consolidation；
本轮仅作源码与复现条件审查，没有测得任何性能或正确性结论。
