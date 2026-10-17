# D631：撤回未获完整性能支持的私有hash移锁候选

2026-10-17。D629完整W2及D630离线诊断已备份，后者提交
8d7e2b6ec3a272aff84b6f909106b094dc2390fa与远端相等。D630 publication
资源域已关闭，峰值38285312B、events/swap0；不重复该发布。

本次精确撤回e12e73f在residency_manager和request_lifecycle测试中的差异。
先核对两个工作区文件与该提交相等，再用apply_patch应用反向差异，未用
整提交revert、reset或checkout，没有删除旧文档、组件记录、失败或性能数据。
恢复文件与e12e73f父版本字节一致：

- residency_manager.py：fd783131934664f1dee49f78b21366c2aff7264d60a15c2a9bcb975e207bec4f
- test_ieee_tc_request_lifecycle.py：d0495b859a863cfd223e8f471fbde3b07e5531061b9864b4d42348e7ced42232

理由是D629未证明整体收益而出现SLO、尾延迟及GPU-s退化，不是已经证明
解锁策略在所有条件下都错误。D630已显示退化主要位于许可等待/启动重叠，
没有精确grant/release边界，不能把全部等待强归于锁。保留完整验证字节和
事务语义，不添加异常兜底、放宽容量或缓存陈旧身份。D606/D610库存改进、
D618合法device域修复和其他已测试改动不变。

遵循历史+D630事实与已核查官方资料：Python Future恢复与完成不同步；
Llumnix评价实例数与尾部的取舍；Knative明确持续下限与initial规模不同。
链接和适用范围见D630，不能据文献推断本机min2达标。撤回只恢复父语义，
没有新增公式、调度策略、模型配置或性能结论。

CPU资格复用D610/D595有界入口，分别运行请求生命周期、transfer pressure、
静态准备、库存观察和HTTP交付模块，避免旧测试导入污染；3/4GiB、swap0、
辅助CPU，GPU不可见，147保护清单及用户manifest不改。不重复组件微测或
旧性能键。通过后按明确路径备份。此时min2尚未建立/资格/启动；下一任务
为D630限定的单一模型资源下限候选，先核查initial、pending与回收边界。

全目标ACTIVE；独立数值UNKNOWN，不把撤回或CPU通过写成G1/G2性能达标。

## 资格结果

五个模块分别215/178/13/16/36项，共458项全部通过；没有新增GPU运行。
实际qualification资源域InvocationID 1f09de3f3c424e32aadba22729adcb25，
inode26877334，峰值1660514304B，high/max/OOM/kill/swap全0，已按身份
核对递归空并关闭、路径消失。全部源码SHA在测试前后相等，147保护项未变。
这只证明精确撤回后的相关CPU合同，没有证明min2性能或独立数值正确性。
min2仍是下一独立候选；不将本次回归测试或D621旧回放计入它的结果。
