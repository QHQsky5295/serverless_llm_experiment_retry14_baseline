# D452：ElasticLocality 原始1000请求兼容检查

2026-10-12。D451完成全500-ID功能覆盖并由520e23a推送封存；当前任务是计划
§6.7的下一项，不重复500-ID、不调优对照、不以短回放替代完整4000W0。

## 冻结输入和机制

- 使用已有W0源trace的前1000行，不覆盖或生成新trace。与D434 S-LoRA的旧
  1000请求视图逐条相同：60个实际出现的adapter、122,510目标tokens。
- 两个历史trace路径的源内容SHA均为
  efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c；
  1000视图SHA为0a5dfcf8048abdc8fc1e3a8ccaf92bfb925a9db326c8d27507663bede96a0981。
  路径名字含local-sim是历史输入身份，本次执行仍走真实HTTP已发布工件。
- 配置与D450/D451仅output/NVMe/HOST路径不同。slack1、普通GPU locality与
  负载回退、普通文件LRU，无handoff/hierarchy/主动admission；保持原生按需
  加载、物理安全、同autoscaler、vLLM0.30.0、FP16/TP1、slot/concurrency14。
- 共同60秒通知、原始open-loop到达、fixed_length_greedy_v1、共同D394 SLO、
  1800秒资格保护不变。不平移到达，不因等待或结果差而修改配置。

官方[vLLM benchmark文档](https://docs.vllm.ai/en/latest/cli/bench/serve/)区分
到达速率与并发限制，后者可能限制实际发起速率；本次继续使用已验证的独立
回放与服务分离路径，保留原始计划到达，不换成同均值的随机合成请求。

## 预运行检查与安全

既有20项输入/回放测试通过（0.633秒）；配置非路径字段相等，算法源码、
底层runtime、指标和安全入口与520e23a逐字无变化。唯一已有脚本改动是新增
固定d452运行分支，PREFIX_COUNT=1000。shell语法和辅助Python编译通过。
147个历史保护项和冻结PLAN/V1/V2、用户修改保持；D451原始结果未改动。
两个CPU验证资源域准确身份/递归空进程树检查后关闭，high/max/OOM/swap均0。

沿用受保护的启动与收尾脚本，不新建运行框架；实际启动前重查磁盘150GiB、
预计新增32GiB、内存增长余量、GPU空闲、已推送HEAD及输入SHA。一次重型
任务；服务72/80GiB和swap2GiB，辅助3/4GiB和零swap、CPU分离、外置watchdog。
D78/D80只读gzip缓存复用，不重新发布；记录真实链路，运行期间不做远端维护。

raw：原项目results/ieee_tc/elastic_locality_qualification/d452_20261012。
input：paper_results/ieee_tc/elastic_locality/d452_prefix_input_qualification.json。
config：configs/ieee_tc/20261012_d452_7b_elastic_locality_1000.yaml。

## 验收与后续

1000/1000原生数量、request/adapter/source/lease绑定，实际prompt和目标与
D434保留合同一致；可靠原生时间与GPU生命周期、真实两端transfer记录一致。
原始未知数值正确性仍保留未知，不把名称/长度当kernel数值证明。
按冻结共同SLO如实报告，不因不达标调优此对照，不外推为G1/G2或完整W0排名。
资源实际释放后才关闭归属明确的远端服务；校验、备份后进行一次完整4000W0。
7B W1/W2/G2与必要PrimeLoRA优化、3B、全部消融/机制/敏感性仍未取消。
