# D1172：7B 需求加权部署期 NVMe 子集预物化候选

状态：已登记，尚未启动 GPU 性能回放。

## 1. 研究问题

D1100 W2 的 503 个 remote-cold 请求中，496 个是本轮首次出现的 adapter。D947
把全部 500 个 adapter 在 replay notice 之前放入 NVMe 后，P95 TTFT 下降，但
生命周期 GPU-s/request 增加约 13.5%，不满足冻结 G1 的资源优先级。因此 D1172
只测试一个有界问题：**在同一 16 GiB NVMe 预算内，按 manifest 需求先验和实际
字节大小选择部署期子集，能否减少 W2 首次 remote-cold 尾部而不超过 D1118 的
物理 GPU-s 预算。**

## 2. 唯一机制差异

相对 D1118，D1172 仅增加：

```yaml
preloading:
  ieee_initial_nvme_preload: true
  ieee_initial_nvme_capacity_mb: 16384
  ieee_initial_nvme_selection_policy: demand_weighted_capacity_v1
  ieee_initial_nvme_staging_multiplier: 1.5
```

选择器对每个 manifest adapter 计算

\[
  V_i = \frac{h_i\,\max(1,\widehat{t_i})}{b_i},
\]

其中 (h_i) 是已有 manifest/static hotness，\(\widehat{t_i}\) 是已有预测加载
时间（缺失时取单位值），(b_i) 是实际 artifact 字节数。按 (V_i) 降序贪心
选择，满足

\[
  \sum_{i\in S} b_i \le
  \frac{B_{\mathrm{NVMe}}}{1.5}.
\]

1.5 不是性能调参，而是 D945 已测得的保留副本加临时 staging 的部署容量倍数。
预物化在 replay notice 之前完成；其 wall time、字节和传输次数单独记录，不计入
请求 TTFT/E2E，但由它改变的生命周期 GPU-s 完整计入物理资源结果。

未入选 adapter 保持原始 confirmed routing、handoff、hierarchy、admission 和
remote-cold 路径；不删除请求、不降低 SLO、不读取未来 trace 行。选择日志记录
manifest 总数、入选数、字节、有效容量和排除数。

## 3. 冻结输入与执行入口

- 模型：Llama-2-7B，vLLM 0.30.0，TP=1。
- workload：D1118 W2，4,000 请求、500-adapter、seed42、同一 trace、同一 W2
  adapter map、fixed-length greedy v1。
- 资源：max_instances=3、16 GiB NVMe、16 GiB HOST、GPU/CPU/远端交付合同均
  与 D1118 相同。
- 配置：`configs/ieee_tc/20261109_d1172_7b_demand_weighted_preload_w2.yaml`。
- 运行脚本：复用 `scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh` 的
  已审计入口，仅传入新的 PROFILE=W2、输出目录、NVMe/HOST 目录、配置文件和
  scenario；不修改旧运行键。

预期命令（启动前再次检查 GPU 空闲、远端 invocation 归属和磁盘余量）：

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh \\
  W2 \\
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1172_20261109/7b_demand_weighted_preload_w2 \\
  d1172_7b_demand_weighted_preload_w2 \\
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1172_20261109/7b_nvme_demand_weighted_preload_w2 \\
  /dev/shm/tc-d1172-7b-demand-weighted-preload-w2 \\
  configs/ieee_tc/20261109_d1172_7b_demand_weighted_preload_w2.yaml \\
  tc_ieee_full
```

## 4. 证伪与闭口

结果只有在 4,000/4,000 请求、native timing/generation contract、physical
allocation union 和远端交付证据完整时才进入比较。候选相对 D1118 必须同时满足：

1. 联合 TTFT/TPOT 达成率不低于 D1118 的 97.500%；
2. 物理 GPU-s 不超过 D1118 的 9,873.292173 GPU-s；
3. P95 TTFT 低于 D1118 的 3,053.004479 ms；
4. remote-cold 请求数及预物化集合由原始逐请求和部署日志直接支持。

若任一项失败，保留完整负结果并退出该候选；不再增加第二种预物化规则，也不
重复全量 D947。若全部满足，D1172 才能作为 W2 同预算尾延迟候选，随后再进入
既定 7B 消融主线。

## 5. 保护边界

不覆盖 D1100、D1117、D1118 或历史 D947/D946；不修改 `lora_manifest_1000.json`、
旧图、论文源文件、共享 k3s 或远端持久工件。当前 shell 缺少 vLLM 的离线单元
测试环境限制仍只作为环境记录，不将其误报为候选运行结果。
