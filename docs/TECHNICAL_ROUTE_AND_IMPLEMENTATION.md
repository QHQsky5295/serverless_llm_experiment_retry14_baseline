# FaaSLoRA 技术路线与实现说明（对齐研究内容）

本文档按技术路线串起三项贡献的实现，并与 S-LoRA / ServerlessLLM 研究内容对应；同时说明热度计算、阈值含义与依据，便于写论文或汇报。

---

## 一、项目定位与问题

本项目的目标是构建一个与 S-LoRA（SOSP'23）和 ServerlessLLM（NSDI'24）同类型的、真实的多 LoRA 无服务器推理系统 FaaSLoRA。S-LoRA 在 LightLLM 上实现 CPU 常驻内存与 LRU GPU 缓存，ServerlessLLM 以 NVMe 为 checkpoint 层、默认采用 Transformers 进行加载与调度；二者都强调真实存储层次与真实加载。FaaSLoRA 在同样追求“真实系统”的前提下，针对多 LoRA 并发时的冷启动、存储层次与显存竞争问题，通过三项技术贡献实现：**热点感知的三层级联预加载（远端→硬盘→内存→GPU）**、**多级驻留管理（谁在 GPU、谁在内存、谁在硬盘）**、以及**负载与显存协调**，在统一推理引擎与工作负载下与上述工作形成可对比的基线。

---

## 二、技术路线与系统层次

系统在技术路线上采用“**存储层 — 预加载与驻留 — 协调**”三层设计。存储层将 LoRA 划分为**远端、NVMe（硬盘）、主机内存（HOST）、GPU** 四级；请求到达时按 **GPU → HOST → NVMe → 远端** 的顺序解析所在层，再决定是直接推理、从内存/硬盘加载到 GPU，还是先从远端下载到硬盘再加载，**禁止从远端直接加载到 GPU**。预加载与驻留层负责在启动时做三层级联预加载（远端→NVMe→HOST→GPU），并按容量与热度阈值决定谁进哪一层；协调层在显存紧张时对“加载到 GPU”的请求排队或驱逐，在负载结束后按访问得分保留 warm pool。三项贡献分别对应这三层中的关键机制，下面按实现顺序说明，并给出**热度计算、阈值及其依据**。

---

## 三、贡献一：热点感知的三层级联预加载（远端→硬盘→内存→GPU）

第一项贡献解决的是“在请求到达前，把**最热的放到 GPU、次热的放到内存、再次的放到硬盘**，避免远程直连 GPU”的问题，与 ServerlessLLM 的“有 NVMe 但无预加载”形成对比。

### 3.1 触发时机

- **启动时**：一次性的三阶段预加载（Stage 1 远端→NVMe，Stage 2 NVMe→HOST，Stage 3 HOST/NVMe→GPU warmup）。
- **运行中**：当扩缩决策判定为 **scale-up** 时，再触发一次预加载（当前实现为按固定容量 200MB 向 GPU 层补货，由 PreloadingManager 生成计划并执行）。scale-up 的判定见贡献三。

### 3.2 热度如何计算

- **静态热度（预加载与 GPU warmup 的初始依据）**  
  每个 LoRA 在配置（如 `experiments.yaml` 的 `lora_adapters.adapters[].hotness`）中有一个 **[0,1] 的标量**，表示先验的“预期访问频率”。在实验栈注册时（`_ensure_registered`）写入 ArtifactRegistry 的 `hotness_score`，并据此计算 `value_per_byte = hotness / max(size_mb, 1)`，用于预加载阶段的候选筛选与排序。

- **动态热度（运行中更新，供后续 scale-up 预加载与驻留使用）**  
  每次请求命中某 LoRA 时，会调用 **HotnessTracker.record_access(adapter_id)**。热度更新逻辑为：
  - **滑动时间窗**：保留最近 `window_seconds`（默认 300 秒）内的 (timestamp, adapter_id) 记录，最多 `window_max_entries`（默认 5000）条。
  - **窗内占比**：对当前 adapter，`count_in_window = 窗内该 adapter 的访问次数`，`total = 窗内总访问数`，则  
    `hotness = count_in_window / max(total, 1)`（若 total 为 0 则用 EWMA 值）。
  - **放大与截断**：`hotness = min(1.0, hotness * 2.0)`，使高占比的 adapter 更快接近 1。
  - **EWMA 备用**：`new_ewma = alpha * 1 + (1 - alpha) * prev_ewma`，默认 `alpha = 0.2`，用于窗内无数据时的平滑。
  - 上述热度与 `value_per_byte = hotness / max(size_mb, 0.1)` 写回 ArtifactRegistry，供 PreloadingPlanner 与 ResidencyManager 使用。

因此：**预加载阶段**主要依赖**静态热度**（配置）；**运行中 scale-up 预加载与驻留/驱逐**会逐步依赖**动态热度**（请求驱动的窗内占比 + EWMA）。

### 3.3 预加载用到的阈值及其依据

- **min_hotness_threshold（PreloadingPlanner）**  
  - **含义**：只有 `metadata.hotness_score >= min_hotness_threshold` 的 artifact 才进入“预加载候选”。  
  - **当前取值**：配置中默认 `0.1`（ExperimentStack 的 preloading 配置）。  
  - **依据**：过滤掉几乎无人问津的 LoRA，减少无效拷贝；0.1 表示“在窗内或先验上至少有约 10% 级别的相对热度”才考虑预加载。

- **value_threshold（PreloadingPlanner）**  
  - **含义**：`value_per_byte >= value_threshold` 才作为候选。  
  - **当前取值**：默认 `0.01`（实验栈）；production/default 中有的用 0.1。  
  - **依据**：value_per_byte = hotness/size_mb，过滤掉“体积大且热度低”的 LoRA，优先预加载“单位字节价值”高的。

- **min_hotness（场景预加载配置，YAML）**  
  - **含义**：在**未使用 full stack** 的旧路径里，用于筛“热”的 adapter 做简单预加载；**使用 full stack 时**，三阶段预加载由 PreloadingPlanner 的 `min_hotness_threshold` 与容量约束决定，不直接读该值。  
  - **当前取值**：faaslora_nvme 为 0.4，faaslora_no_coord / faaslora_full 为 0.3。  
  - **依据**：更保守场景（仅 C1）用更高阈值 0.4，减少误预加载；C1+C2/C1+C2+C3 用 0.3，稍多预加载以发挥多层驻留。

- **gpu_warmup_hotness（Stage 3：谁进 GPU）**  
  - **含义**：在三阶段预加载的 Stage 3，只有 `adapter_info[aid].hotness >= gpu_warmup_hotness` 的 LoRA 才会被真实加载到 GPU（warmup）。  
  - **当前取值**：三个 FaaSLoRA 场景均为 `0.6`。  
  - **依据**：保证“进 GPU”的是真正最热的一批（高于 0.5），避免把有限显存浪费在温/冷 LoRA 上；0.6 与 0.3/0.4 形成梯度（进硬盘 < 进内存 < 进 GPU）。

### 3.4 选择与执行

- **选择**：PreloadingPlanner 从“当前所在层低于目标层”的 artifact 中筛出满足上述热度与 value 阈值的候选，再按 **priority_score** 排序；priority 的公式为：  
  `0.4 * value_per_byte + 0.3 * hotness_score + 0.2 * (1/(predicted_load_time_ms+1)) + 0.1 * tier_factor`，其中 tier_factor 为“目标层权重 − 当前层权重”（GPU>HOST>NVMe>REMOTE）。在容量约束下用 hybrid/greedy 或背包策略选出本次预加载集合。
- **执行**：Stage 1 做远端→NVMe 的真实文件拷贝并 admit NVME；Stage 2 做 NVMe→HOST 的真实拷贝并 admit HOST；Stage 3 对热度≥gpu_warmup_hotness 的 LoRA 从 HOST 或 NVMe 路径做真实 GPU 加载（引擎 load 或短推理）并 admit GPU。这样，**最热在 GPU、次热在内存、再次在硬盘**，与论文中“真实迁移、真实存储层次”一致。

---

## 四、贡献二：多级驻留与“谁在 GPU、谁在内存、谁在硬盘”

第二项贡献解决的是“在有限 GPU 显存下，哪些 LoRA 常驻 GPU、哪些只在 HOST 或 NVMe”的问题，对应 S-LoRA 的“CPU/GPU 分层 + LRU”和 ServerlessLLM 的“NVMe checkpoint”的层次思想，但用统一的四层（远端、NVMe、HOST、GPU）与准入/驱逐策略实现。

### 4.1 分层与阈值

- **四层**：REMOTE → NVMe → HOST → GPU；请求解析顺序为 **GPU → HOST → NVMe → 远端**，冷启动时**只做远端→NVMe（或本地盘）**，再从本地加载到 GPU，从不从远端直连 GPU。
- **谁进 GPU**：启动时由 **gpu_warmup_hotness**（如 0.6）决定；运行时由 ResidencyManager 的容量与驱逐策略决定（见下）。
- **谁进 HOST**：由 Stage 2 的预加载计划决定，计划在容量约束（host_capacity_mb）下按 priority 从已在 NVMe 的 LoRA 中选出。
- **容量**：NVMe 用 `nvme_capacity_mb`（如 20480），HOST 用 `host_capacity_mb`（如 4096），GPU 由硬件与模型配置（gpu_budget_mb、model_weights_mb、kv、预留比例）推导。

### 4.2 准入与驱逐

- 当需要把某 LoRA 迁入某层时，ResidencyManager 检查该层容量；若超过**准入阈值**（如 utilization > admission_threshold），则先按配置策略（LRU / value_based / hybrid）驱逐若干已在层内的 LoRA 到下一层，再接纳新 LoRA。
- 驱逐时下一层由 tier 层级决定（如 GPU→HOST→NVMe→REMOTE）。

### 4.3 请求路径

- 先查 GPU（命中则 cache_tier="gpu"，无加载延迟）；再查 HOST（命中则从 host 路径加载到 GPU，cache_tier="host"）；再查 NVMe（从 nvme 路径加载，cache_tier="nvme"）；最后走远端（ensure_local 下载到 nvme 后再加载，cache_tier 记为 "nvme" 或 "remote"）。从 HOST/NVMe 到 GPU 的加载使用引擎的真实接口，并可选择用实测时间写回 registry（D1 块），保证“真实加载、真实层次”。

---

## 五、贡献三：资源协调

第三项贡献解决的是“在突发负载与缩容时，如何避免盲目驱逐、又如何回收显存”的问题。

### 5.1 Scale-up 时的加载协调

- 当某请求需要将 LoRA 加载到 GPU 但当前显存不足时：
  - **协调开启（faaslora_full）**：不立即驱逐，而是将本次加载**排队**，等待显存释放（如批次结束、KV 释放或其它加载完成）后再执行，等待时间计入该请求的 defer_ms。
  - **协调关闭（faaslora_no_coord）**：立即按策略驱逐若干 LoRA 以腾出空间再加载。
- **Scale-up 的触发**：每处理完一批请求（batch），用该批的 **batch_rps = len(batch)/Δt** 与 AutoScaler 的规则做决策；其中一条规则为 **scale_up_threshold_rps**（默认 3.0）：当 batch_rps > 3.0 且满足 cooldown、实例数上限等条件时，判定为 SCALE_UP，并触发一次预加载（trigger_scaling_preload）。

### 5.2 Scale-down 与 warm pool

- **触发时机**：当前实现是在**本阶段所有请求处理完后**统一调用 `trigger_scale_down()`（与“RPS 低于某阈值再触发”的实时判断可区分开，见下节阈值与公式讨论）。
- **保留谁**：在最近 `access_window_s`（默认 60 秒）内，对每个 GPU 驻留 LoRA 计算  
  `score = freq / log(1 + recency)`，其中 freq 为窗内访问次数，recency 为距最近一次访问的秒数。按 score **升序**排序（最冷在前），保留 **warm_pool_size**（如 4）个最热的，其余从 GPU 逐出到下一层。逐出与保留都作用在 ResidencyManager/引擎的真实驻留集合上。

---

## 六、与 S-LoRA / ServerlessLLM 的对应关系

- **S-LoRA**：CPU 常驻 + LRU GPU；本系统用 slora_style 基线模拟“CPU→GPU + LRU 逐出”的延迟与行为；FaaSLoRA 的贡献一/二用“HOST（内存）+ NVMe（硬盘）+ GPU”三层实现“最热→GPU、次热→内存、再次→硬盘”，与 S-LoRA 的层次思想对齐但扩展为四层。
- **ServerlessLLM**：NVMe checkpoint、无预加载、Transformers 默认；本系统用 serverlessllm 基线模拟“有 NVMe、无预加载、按需加载”；FaaSLoRA 的贡献一明确“有预加载且三层级联”，形成对比。
- **FaaSLoRA**：hit-aware 预加载（含热度与阈值）、多层驻留、资源协调；热度由静态配置 + 动态窗内占比与 EWMA 计算；阈值（min_hotness_threshold、value_threshold、gpu_warmup_hotness、scale_up_threshold_rps、warm_pool_size）均在配置中可调，依据见上文。

---

## 七、阈值设置是否合适与公式化触发建议（不改代码，仅说明）

### 7.1 当前阈值是否合适

- **min_hotness_threshold = 0.1**：偏宽松，能覆盖“有一定热度”的 LoRA，适合实验与多数 trace；若希望更保守，可提高到 0.2～0.3。
- **gpu_warmup_hotness = 0.6**：与 0.3/0.4 形成梯度，保证进 GPU 的是最热一批，合理。
- **scale_up_threshold_rps = 3.0**：固定常数，在负载波动大时可能过于敏感或迟钝。
- **trigger_scale_down**：当前是“阶段结束即执行”，未与实时 RPS/队列长度绑定，无法做到“仅当负载低于某值时才缩容”。

整体上，**预加载与 warmup 的阈值**（0.1、0.3、0.4、0.6）在经验上可用；**扩缩触发的阈值**目前是固定常数，灵活度有限。

### 7.2 公式化、实时计算触发的思路（类似“最低调度值”）

可以引入**实时计算的门槛**，只有当前指标**超过（或低于）该动态值**时才触发，使策略更灵活、更贴近负载：

- **Scale-up 预加载**  
  - 思路：不固定 `scale_up_threshold_rps = 3.0`，而设一个**动态下限**，例如  
    `T_scale_up = max(T_min, baseline_rps * (1 + margin))`  
    其中 `baseline_rps` 为近期（如过去 1～5 分钟）的平均 RPS 或 P50 RPS，`margin` 为可调比例（如 0.2～0.5），`T_min` 为最小触发线（如 1.0）。当 **当前 batch_rps > T_scale_up** 时触发 scale-up 预加载。  
  - 效果：负载基线高时触发线自动抬高，避免轻微波动就扩容；基线低时仍可用 T_min 防止完全无触发。

- **Scale-down（含 warm pool 保留）**  
  - 思路：不“阶段结束必缩”，而设**动态上限**，例如  
    `T_scale_down = min(T_max, baseline_rps * (1 - margin))`  
    当**近期平均 RPS < T_scale_down** 且持续一段时间（如 30～60 秒）时，再调用 `trigger_scale_down()`。  
  - 效果：只有负载明显低于“正常水平”才缩容，减少抖动；类似 AWS target-tracking 的“scale-in 需低于 target 一定比例才触发”。

- **预加载候选热度**  
  - 思路：`min_hotness_threshold` 改为动态，例如  
    `T_hot = percentile(registry 中所有 LoRA 的 hotness_score, p)`  
    取当前热度分布的 p 分位（如 p=25）作为门槛，只预加载热度 > T_hot 的 LoRA。  
  - 效果：随流量分布变化自动调整“多热才算热”，无需手调固定 0.1。

上述公式均可先在设计文档或实验计划中给出，待你确认后再在代码中实现（你已要求本步仅检查与撰写，不改代码）。
