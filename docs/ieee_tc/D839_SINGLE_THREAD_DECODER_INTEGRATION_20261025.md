# D839：单线程兼容解码的最小请求路径接入

2026-10-25，基点362b17c9；上一轮D837/D838为progress，有实际组件证据。
当前3B W2完整联合3683/4000（92.075%），差117条。D711独立数值身份未知
继续保留，不能由本组件推出N_correct=4000或最终G1/G2合格。

## 唯一假设及历史

D838同归档、同owner最终SHA的解包从stdlib148.048–148.689ms降至
isal56.099–61.508ms，六个调用及全部失败校验已封存。仅将这个兼容解码器
接入现有HTTP物化；不修改routing/admission/预算/状态传播/生成或IEEE九公式。
D837直接r|gz更慢已拒绝；D834无联合收益已撤回。D628移锁、D595/D598并行
ACK、D822共享压力不重开。D825目的文件完整SHA和D831文件ready/settlement
重叠继续保留，不从时间线上事后扣除准备或收尾开销。

## 实现合同

- HttpArtifactStoreClient显式gzip_decoder取stdlib或isal。标准为兼容默认；
  无auto、异常重试换解码器或缺包回退。选择只在构造时确定。
- endpoint_from_env读取FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER；新候选显式
  选择isal，记录版本。依赖固定1.8.0，复用D838独立overlay，不修改冻结环境。
- 单线程igzip.open＋原tar解析/提取；解码器与tar对象在异常/取消时均关闭。
  malformed gzip头转换成原tarfile.ReadError类别，不吞掉提取异常。
- 归档SHA、字节/大小/成员集合、预分配、取消、owner内容校验、原子发布和
  共享压力收尾全部不变。远端归档/服务/带宽、权重和负载不变，无请求打包。
- 每次传输的gzip_decoder字段记录实际选择及库版本；它不是新的观测开销
  计时器。其他系统默认路径不静默升级；若将来作为共同交付优化，要另列
  新合同并完整披露，不能把纯解码加速冒称分层机制收益。

## 官方依据与适用边界

本轮重新核查[固定isal源码](https://raw.githubusercontent.com/pycompression/python-isal/v1.8.0/src/isal/igzip.py)
的文件对象接口和单线程C读器；以及[CPython 3.12.12 tarfile](https://raw.githubusercontent.com/python/cpython/v3.12.12/Lib/tarfile.py)
的gzopen关闭/异常边界。D837/D838已有ServerlessLLM加载路径和vLLM原生加载
依据继续复用；本项针对实测CPU热点，不引入它们的工件布局或更改后端算法。

## 验证与主线

复用现有HTTP原子发布与ConfirmedFilePublication夹具，在明确选择isal的
子类运行原契约测试；独立构造client的帮助函数仍为标准路径，不能冒称全部
助手都切到isal。另覆盖配置冻结、缺包报错、未知选择、实际runner记录、
解包异常关闭。再跑标准HTTP/owner路径，保证既有系统未被默认改变。
测试仅CPU，3/4GiB、swap0、四辅助CPU，原150GiB磁盘和147保护继续。

通过后只做一个新执行键完整3B W2，不加短Full/组件寻优。按共同SLO、物理
GPU-s及完整阶段证据决定保留或撤回，不能由组件加速推算联合达标。
同版本7B/3B W0/W1/W2、M1/M2、A1–A5、S1–S13、CE/Pareto、审稿证据、
图表及最终V2仍未全部完成，范围不减。效果接近目标前不画图。

## 实际验证结果

候选选择/ISA-L夹具75项、原标准HTTP/owner回归69项全部通过，无跳过。
这是144次测试执行，包含两种解码器对同一合同的重复覆盖，不当作144种
独立机制或性能样本。两域内存峰值及关闭、保护校验见closure/evidence。
high/max/OOM/swap均为0，147个旧投稿文件不变。没有GPU回放或正式排名。
下一只冻结D840完整3B W2一次；缺失独立数值身份仍为未知。
