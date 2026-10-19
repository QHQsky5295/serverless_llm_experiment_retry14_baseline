# D701：3B 同规格 native profile 资格准备

日期：2026-10-19
执行键：`D701_20261019`
状态：CPU-only preparation complete；尚未启动 GPU 或远端服务

## 动机

D700 已证明 D698 的 30 个类别缺口可以由真实原生观测补齐，但 D700 使用 6
个 representative adapters 的 gap-only source-spec，不能与 D645 的 24 个
representative source-spec 直接合并。若直接拼接，会违反 profile collector 的
同规格 provenance 约束。D701 因此采用唯一可审计的路线：新建一个完整的、同一
规格的 source collection，保留 D645 的所有代表和原始 waves，只追加 D700 已
静态准备的 gap waves。

## 静态合同

- 不生成 trace、请求负载、LoRA 工件或权重；所有引用来自已有 1,000-request
  prefix 与 500-adapter frozen pool。
- 保留 D645 的 24 个 exact-content representatives、46 个既有 waves 和原
  runtime/input/model contract。
- 仅将候选 service-profile 的 declared-output edges 固定为 `[89,152,179,256]`。
- 追加 D699/D700 已登记的 30 个 gap measurement waves；这些波次的背景请求、
  target request、source tier 和 admission edge 均沿用既有静态引用。
- 静态 resolver 通过：76 waves、75 个 representative measurement waves、
  423 个既有 request references、24 个 representative identities。
- 输出目录为
  `paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/`；其中的
  `source_spec.json` 和 `prepare.json` 为追加写入，不覆盖 D645/D699/D700。

## 运行前后验收

D701 只有在资源门禁、GPU idle、内存/swap/OOM 门禁和真实 prepublished 远端
交付健康检查全部通过后才可运行。动态 collector 必须逐请求验证原生 token、
adapter binding、admission/source class、释放和阶段恒等式；请求期不得发生
远端打包。完成后必须重新运行严格 exporter：service-class domain、exact-content
source coverage、同一 source-spec SHA 和 loader 验证全部通过，才可生成
development-only profile。

在这些条件满足前，D701 的 `launchable=false`、`gpu_runs=0`、
`profile_export_authorized=false` 均保持不变；不能把它当作 3B PrimeLoRA
性能证据，也不能用 D700 的 61 条请求替代完整同规格 collection。

机器可读准备产物：
`paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/prepare.json`。
准备脚本：`scripts/ieee_tc/prepare_d701_3b_same_spec_profile.py`。
