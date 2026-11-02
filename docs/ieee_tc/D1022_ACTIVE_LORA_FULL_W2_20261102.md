# D1022：活跃 LoRA 专门化的完整 3B W2 验证

2026-11-02。状态：已登记、静态画像绑定通过，尚未启动Full。
当前交接以本文及D1021完成节为准；用户dirty的EXECUTION_STATUS.md不覆盖。
上一目标回合属于progress：新候选接入、必要画像实际启动；本回合闭合该唯一
画像并推进完整回放，不缩小完整目标、不重复已完成基线。

## 问题与保持不变的部分

D1020已在4000请求上观测到3746联合达标（93.65%）、8135.629131GPU-s、
P95 TTFT 1940.053078ms，仍少54条达到3800门槛。D1021测试原生vLLM
specialize_active_lora=true：32槽的驻留容量与当前活跃LoRA计算规模分离。
历史、官方源码与可证伪假设详见D1021，不重复已否决的控制延后、文件优先、
扩缩容参数、HOST图、token预算及槽数搜索。

本次只相对D1020改变原生specialization选项及其必要新画像/身份；请求并发8、
GPU槽32/CPU槽32、其他backend配置、D747 fresh descriptor、stdlib gzip、
路由/handoff/residency/admission、扩缩容和IEEE九公式保持。D1021画像启动
增加16.228秒、KV减少349块，完整运行必须包括这些代价。官方捕获集合不是
实测kernel证据；不在本次顺便裁减捕获集合或修改策略。

完整原4000请求、500-adapter universe、W2 rotation100映射、arrival/prompt/
token、fixed_length_greedy_v1、业务前60秒通知和真实已发布远端合同不变。
源seed42是已知开发回放，不冒称独立留出；同键单轮、无CI、不筛赢家。

## 冻结输入及静态检查

- PLAN SHA：d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
- 指标V1 SHA：5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22。
- 单轮V2 SHA：193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
- 配置：configs/ieee_tc/20261102_d1022_3b_active_lora_w2.yaml，SHA
  f8d4ac96c9121dba134e2f3ec5e6a9e354b09dcf44447bb5cc65ecefc8fda171。
- D1021 manifest SHA：1fa53f43ffec59d472fbe6fdc070b02ca63d60f9e0564405708cefa2df7281a2。
- 服务画像SHA：e846698d6c8cb3aa0d94c31b92fcbf65e852a5962ad9e6fb86c7f8cd955ef643。
- 准备画像SHA：b35a6d32b36b445fcadbff301794159165fe2263d24f7f70ec1f059feaa5fc12。
- 独占raw：/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1022_20261102/d1022_3b_active_lora_w2_full。

CPU-only实际ScenarioRunner组装和两个画像加载器通过：160服务类、96准备类，
模型/画像身份一致；关闭specialization时两个加载器均拒绝。父配置差异检查
确认只有目标字段、画像及专属输出路径变化。该检查未运行GPU；3/4GiB、swap0、
辅助CPU2/3/26/27，完成退出，未保存终态资源峰值。D1021已覆盖变更的原生
混合批路径，不再追加无问题目标的100请求Full。

## 唯一完整回放入口

```bash
env -u FAASLORA_TC_CORE_TIMING -u FAASLORA_TC_CORE_STEP_TIMING -u FAASLORA_TC_HOST_INVENTORY_TIMING \
  FAASLORA_IEEE_HOST_INVENTORY_MODE=descriptor \
  FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so \
  FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044 \
  FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=stdlib \
  FAASLORA_D1000_RUN_ID=d1022_20261102 \
  FAASLORA_D1000_RUN_TAG=d1022_3b_active_lora_w2_full \
  FAASLORA_D1000_CONFIG=configs/ieee_tc/20261102_d1022_3b_active_lora_w2.yaml \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

入口旧文件名含W1，实际固定W2；无prefix变量。测量前提交备份、即时安全检查；
服务72/80GiB、swap2，辅助3/4GiB、swap0、固定CPU和独立watchdog。空机内存
≥102GiB，磁盘≥max(150,100+1.5×32)GiB；运行期16/100GiB保护不变。
不与压缩、构建、推送、其他GPU测量或远端配置操作并行。复用既有健康远端，
不接管共享k3s。无连续远端CPU监控，如实保留证据范围限制。

## 完成判定、解释与后续

先核验实际资源释放，再以现有audit_d703显式指定本配置/D1021 manifest，
恢复共同D648 SLO、4000请求阶段分解、原生绑定/输出计数、真实远端传输与
物理GPU积分；不使用runtime旧5000ms显示替代冻结评价。G1全部正确及至少
3800联合达标后比较资源；G2还需U≤12083.063870GPU-s，再比较W2 P95与
观测SLO差。独立数值正确性未知仍记n_correct=null，不偷换正式排名资格。

与D1020和D810全部baseline观测一起看，不仅与自己比；vLLM W2为100%联合/
16110.752GPU-s/P95 441.635ms，ServerlessLLM为98.125%/16084.201/480.724ms。
它们超过当前75%预算，不把这些点直接称同预算可行对照，也不隐藏其更好延迟。
其他基线低分/失败及适配局限全部保留，不为追求95%重开其优化。

结果若退化则保存并否决，不重跑同键；结果改善也需满足完整目标，不能只凭
内核开关或画像通过宣称最优。达到合适工作点后推进同版本W0/W1及比较，
不陷入局部无限优化。全部两模型最终G1/G2、A1–A5、S1–S13、数值正确性
范围、B1/B2/C1/C2/C3及两轮审稿文档、后期图表和最终V2集成仍未删除。
当前不绘图、不改论文、不新增工件/trace，不覆盖历史结果。
