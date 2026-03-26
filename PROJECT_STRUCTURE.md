# FaaSLoRA 项目结构说明（当前 clean-tree）

> 当前仓库主语义、实验进度和论文口径，请以 [README.md](README.md)、[docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md](docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md) 和 [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md) 为准。

本文档只说明当前 clean-tree 的文件布局和各目录职责。

## 1. 当前仓库根目录

当前权威代码树：

- `/home/qhq/serverless_llm_experiment_retry14_baseline`

历史脏树：

- `/home/qhq/serverless_llm_experiment`

仅保留作历史参考，不再作为正式实验主树。

## 2. 一级目录总览

```text
serverless_llm_experiment_retry14_baseline/
├── faaslora/               核心系统模块
├── scripts/                实验入口、数据与适配器脚本
├── configs/                实验配置与 curated manifest
├── docs/                   仓库主文档
├── docs copy/              IDE 常用文档镜像
├── tests/                  smoke / regression tests
├── README.md               项目概览
├── EXPERIMENT_GUIDE.md     实验运行指南
├── PROJECT_STRUCTURE.md    本文件
├── pyproject.toml          Python 包配置
└── requirements.txt        依赖说明
```

## 3. 关键目录说明

### `faaslora/`

核心研究代码所在目录，主要包含：

- `experiment/`：实验完整栈、实例池、在线热度与 routing 组织
- `preloading/`：贡献 1，命中感知预加载
- `memory/`：贡献 2，GPU/HOST/NVMe 驻留与监控
- `scheduling/`：贡献 3，资源协同调度
- `coordination/`：autoscaler 与扩缩容逻辑
- `serving/`：vLLM / transformers 后端封装
- `datasets/`：Azure trace 与 ShareGPT 相关数据入口

### `scripts/`

当前最关键的脚本有：

- `run_all_experiments.py`
- `run_all_experiments_user_scope.sh`
- `generate_lora_adapters.py`
- `download_model.py`
- `download_datasets.py`

### `configs/`

包含：

- `experiments.yaml`：主实验配置
- `generated/lora_manifest_1000.json`：当前刻意纳入 Git 的 curated manifest

### `docs/` 与 `docs copy/`

- `docs/` 是仓库主文档
- 顶层 `docs copy/*.md` 是常用镜像文档
- 同步 GitHub 前，两者应保持一致
- 误生成的嵌套镜像目录不应提交

## 4. 当前实验入口习惯

虽然 `configs/experiments.yaml` 顶层默认 `profile_selection` 仍保留 14B 入口，但当前 rollback 主线更常见的做法是：

- 使用环境变量覆盖 profile 选择
- 固定跑 `Qwen 7B V2 publicmix + 500 adapters + 500 requests`

详见 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)。

## 5. Git 跟踪边界

当前仓库应提交：

- source/config/docs/tests
- curated manifest

不应提交：

- `results/`
- `artifacts/`
- `data/`
- 模型权重
- `/tmp` 运行日志
