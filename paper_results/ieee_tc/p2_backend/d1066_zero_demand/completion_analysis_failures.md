# D1066 离线分析失败与恢复

未重跑 GPU，原始数据与已成功的 `full4000.json` 均不覆盖。

1. 首个收集器正常完成 4000 请求原生指标重算，随后 profile audit 拒绝旧
   YAML 的输出目录。D1000 入口实际使用 `FAASLORA_OUTPUT_DIR` 覆盖。恢复版
   检查实际结果 metadata 的覆盖值与 launcher 原 YAML 路径，只为既有分析器
   生成输出目录绑定投影；原 YAML 和策略不改。投影及来源另存，恢复分析通过。
2. 第一次机制比较拒绝两个历史 manifest 内同路径分析器的不同历史 SHA。
   这是代码版本身份不同，不是测量源矛盾。保留 `comparison_sources.json`，
   第二版只将当前函数实际读取的 metrics、requests、outcome、launch 绑定进
   比较源集，同时保留原完整 manifests 的 SHA。未绕过请求输入或数据 SHA 校验。

完整回放末尾轻量监测曾对 terminal event 读取仅 resource_sample 有的字段，
发生 null 除法格式化错误；launcher 的成功和实际资源释放证据独立核实，
未据此重启、丢弃结果或补造资源值。
