# Punica 环境说明

本文档记录 `Punica` 在当前 baseline 工作区中的隔离环境要求。

## 原则

1. 不与 `FaaSLoRA` 正式环境混用
2. 不覆盖已有 `ServerlessLLM` 相关环境
3. 能独立删除、重建与回退

## 当前状态

- 尚未开始正式安装
- 后续将在确认官方依赖和 CUDA / vLLM / PyTorch 版本要求后补全

## 预期内容

后续将补充：

1. 建议 conda 环境名
2. Python 版本
3. 关键依赖版本
4. 与当前机器 GPU / CUDA 条件的匹配说明
