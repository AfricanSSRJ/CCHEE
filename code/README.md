# Experiment Code

# 实验代码

This directory contains the experiment code associated with the CCHEE dataset
and the CCDA data augmentation method.

本目录包含 CCHEE 数据集与 CCDA 数据增强方法相关的实验代码。

## Structure

## 目录结构

```text
code/
├── requirements.txt
├── argument_extraction_baselines/
│   ├── run.py
│   ├── train_span.py
│   ├── train_bio.py
│   ├── generative_baseline.py
│   ├── dataset.py
│   ├── common/
│   ├── nn_models/
│   └── data/
└── ccda_augmentation/
    ├── run.py
    ├── augment.py
    ├── esam.py
    ├── rcsm.py
    ├── quality_control.py
    ├── evaluate.py
    ├── analysis.py
    ├── ext_models/
    ├── llm/
    ├── common/
    ├── nn_models/
    └── data/
```

`argument_extraction_baselines/` provides argument extraction baselines used to
evaluate the released event-argument annotations, including span classification,
BIO tagging, and generative zero-shot/few-shot settings.

`argument_extraction_baselines/` 用于数据集上的论元抽取基线实验，包括 Span 分类、BIO 序列标注以及生成式 zero-shot/few-shot 设置。

`ccda_augmentation/` provides the implementation of the CCDA augmentation
experiments, including ESAM, RCSM, quality control, long-tail analysis, and
wrappers for Pipeline-Span, Joint-BIO, and Generative settings.

`ccda_augmentation/` 用于 CCDA 数据增强实验，包括 ESAM、RCSM、质量控制、长尾分析，以及 Pipeline-Span、Joint-BIO 和 Generative 模型设置。

## Environment

## 环境

Install dependencies from the repository root:

在仓库根目录安装依赖：

```bash
pip install -r code/requirements.txt
```

The experiments use public base models such as GuwenBERT, BERT-base-Chinese,
Qwen2.5-7B-Instruct, and Xunzi-Qwen2-7B. Model weights are not included in this
repository.

实验使用 GuwenBERT、BERT-base-Chinese、Qwen2.5-7B-Instruct 和 Xunzi-Qwen2-7B 等公开基础模型。本仓库不包含模型权重。

## Data Preparation

## 数据准备

Each experiment directory contains sample files under `data/` to illustrate the
required format. For full experiments, place the complete files under the
corresponding `data/` directory:

每个实验目录的 `data/` 下包含样例文件，用于说明代码所需的数据格式。运行完整实验时，请将完整文件放入对应目录：

```text
data/schema.txt
data/all_arguments.jsonl
data/event_type_definitions.json    # optional for CCDA augmentation
```

Argument offsets may use either `start`/`end` or `start_offset`/`end_offset`.
They are normalized internally during loading.

论元位置字段支持 `start`/`end` 和 `start_offset`/`end_offset` 两种写法，代码在读取时会统一转换。

## Run Argument Extraction Baselines

## 运行论元抽取基线

Run from the `code/` directory:

在 `code/` 目录下运行：

```bash
python -m argument_extraction_baselines.run \
  --schema argument_extraction_baselines/data/schema.txt \
  --data argument_extraction_baselines/data/all_arguments.jsonl
```

This command runs span classification, BIO tagging, and generative zero-shot or
few-shot baselines according to the configuration file.

该命令会根据配置运行 Span 分类、BIO 序列标注以及生成式 zero-shot/few-shot 基线。

## Run CCDA Augmentation Experiments

## 运行 CCDA 数据增强实验

Run from the `code/` directory:

在 `code/` 目录下运行：

```bash
export OPENAI_API_KEY=...
python -m ccda_augmentation.run \
  --schema ccda_augmentation/data/schema.txt \
  --data ccda_augmentation/data/all_arguments.jsonl \
  --definitions ccda_augmentation/data/event_type_definitions.json \
  --use_gpt4o
```

When `--use_gpt4o` is not set, the code runs Baseline, Random Oversampling, and
EDA comparison settings without calling GPT-4o.

不设置 `--use_gpt4o` 时，代码只运行 Baseline、Random Oversampling 和 EDA 等不调用 GPT-4o 的对比设置。

Additional options:

补充选项：

- `--longtail_ratio`: cumulative frequency threshold for long-tail grouping, default `0.7`.
- `--fast_qc`: skip model-level batch filtering and keep event-level and argument-level checks only.

启用 `--use_gpt4o` 时，ESAM 与 RCSM 的增强样本只生成一次，并在主实验和消融实验中复用，以保持对比设置之间的增强来源一致。

## Notes

## 说明

The released dataset is the output of the GPT-4o assisted annotation workflow.
The code here focuses on dataset experiments and CCDA augmentation experiments.
All evaluation scripts use strict matching: a prediction is correct only when
both the span boundary and role label match the gold annotation.

本仓库发布的数据集是 GPT-4o 辅助标注流程的输出。本目录代码主要对应数据集实验与 CCDA 数据增强实验。所有评估均采用严格匹配口径，即预测论元需要同时满足边界和角色标签完全一致。
