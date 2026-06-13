# CCHEE

CCHEE is a Classical Chinese Historical Event Extraction dataset constructed for fine-grained event extraction from historical texts.

本仓库面向论文“基于细粒度数据增强的古汉语历史事件抽取方法”整理，包含古汉语历史事件抽取数据集、事件模式、提示词模板、数据校验脚本和实验代码。数据集标注事件类型、触发词、论元文本、论元角色及其字符级 span，可用于古汉语事件抽取、事件论元抽取、低资源信息抽取和数据增强研究。

## Repository Structure

```text
CCHEE/
├── README.md
├── LICENSE
├── CITATION.cff
├── data/
│   └── data.jsonl
├── schema/
│   ├── event_schema.json
│   ├── schema.md
│   └── schema_raw.txt
├── docs/
│   ├── annotation_guideline.md
│   ├── data_format.md
│   ├── examples.md
│   ├── model_and_experiment_settings.md
│   └── reproducibility.md
├── prompts/
│   ├── gpt4o_argument_annotation_prompt.json
│   └── ccda_augmentation_prompts.md
├── code/
│   ├── README.md
│   ├── requirements.txt
│   ├── argument_extraction_baselines/
│   └── ccda_augmentation/
├── scripts/
│   ├── check_format.py
│   └── count_statistics.py
└── statistics/
    ├── dataset_statistics.json
    └── dataset_statistics.md
```

## Dataset Overview

## 数据集概览

| Item | Count |
|---|---:|
| Documents | 24 |
| Sentences | 8122 |
| Event mentions | 14154 |
| Event types in schema | 67 |
| Event types appearing in data | 67 |
| Argument roles appearing in data | 113 |
| Argument mentions | 31025 |

More details are available in [`statistics/dataset_statistics.md`](statistics/dataset_statistics.md).

## Event Schema

## 事件模式

The event schema contains 67 fine-grained event types. Each event type is associated with a predefined argument-role set.

事件类型和论元角色集合见 [`schema/schema.md`](schema/schema.md) 和 [`schema/event_schema.json`](schema/event_schema.json)。

## Data Format

## 数据格式

The dataset is stored in JSON Lines format. Each line corresponds to one sentence and contains sentence text and event annotations.

数据格式说明见 [`docs/data_format.md`](docs/data_format.md)。

## Example

## 样例

```json
{
  "sen_id": 1,
  "doc_id": 32025,
  "text": "庄襄王为秦质子于赵，见吕不韦姬，悦而取之，生始皇。",
  "events": [
    {
      "id": 1,
      "trigger": "见",
      "label": "交流-个人交流-见面",
      "start_offset": 10,
      "end_offset": 11,
      "arguments": [
        {
          "role": "见面人",
          "text": "庄襄王",
          "start": 0,
          "end": 3
        },
        {
          "role": "见面人",
          "text": "吕不韦姬",
          "start": 11,
          "end": 15
        }
      ]
    },
    {
      "id": 2,
      "trigger": "取",
      "label": "人生-结婚",
      "start_offset": 18,
      "end_offset": 19,
      "arguments": [
        {
          "role": "新郎",
          "text": "庄襄王",
          "start": 0,
          "end": 3
        },
        {
          "role": "新娘",
          "text": "吕不韦姬",
          "start": 11,
          "end": 15
        }
      ]
    },
    {
      "id": 3,
      "trigger": "生",
      "label": "人生-出生",
      "start_offset": 21,
      "end_offset": 22,
      "arguments": [
        {
          "role": "出生人物",
          "text": "始皇",
          "start": 22,
          "end": 24
        },
        {
          "role": "母亲",
          "text": "吕不韦姬",
          "start": 11,
          "end": 15
        },
        {
          "role": "父亲",
          "text": "庄襄王",
          "start": 0,
          "end": 3
        }
      ]
    }
  ]
}
```

More examples are available in [`docs/examples.md`](docs/examples.md).

## Prompts, Models, and Reproducibility

## 提示词、模型与复现说明

The prompt template used for GPT-4o assisted argument annotation is provided in
[`prompts/gpt4o_argument_annotation_prompt.json`](prompts/gpt4o_argument_annotation_prompt.json).

Prompt templates for the CCDA augmentation modules are provided in
[`prompts/ccda_augmentation_prompts.md`](prompts/ccda_augmentation_prompts.md).

Model and experimental settings are summarized in
[`docs/model_and_experiment_settings.md`](docs/model_and_experiment_settings.md).

The experiment code is provided in [`code/`](code/). It includes argument
extraction baselines and CCDA data augmentation scripts.

实验代码见 [`code/`](code/)，包括论元抽取基线模型、CCDA 数据增强流程、质量控制、消融实验与长尾分析相关脚本。

For a concise reproducibility checklist, see
[`docs/reproducibility.md`](docs/reproducibility.md).

## Code

## 代码

Install the code dependencies with:

```bash
pip install -r code/requirements.txt
```

The code directory contains two parts:

- `code/argument_extraction_baselines/`: argument extraction baselines, including
  span classification, BIO tagging, and generative zero-shot/few-shot settings.
- `code/ccda_augmentation/`: CCDA augmentation and comparison
  experiments, including ESAM, RCSM, quality control, long-tail analysis, and
  model wrappers for Pipeline-Span, Joint-BIO, and Generative settings.

代码目录包括两部分：

- `code/argument_extraction_baselines/`：论元抽取基线实验，包括 Span 分类、BIO 序列标注以及生成式 zero-shot/few-shot 设置。
- `code/ccda_augmentation/`：CCDA 数据增强与对比实验，包括 ESAM、RCSM、质量控制、长尾分析，以及 Pipeline-Span、Joint-BIO 和 Generative 模型设置。

See [`code/README.md`](code/README.md) for detailed commands.

## Validation

## 数据校验

Run the following command to check whether trigger and argument spans match the sentence text:

```bash
python scripts/check_format.py --input data/data.jsonl
```

Generate dataset statistics:

```bash
python scripts/count_statistics.py --input data/data.jsonl --schema schema/event_schema.json
```

## License

## 许可证

This dataset is released for academic research use. See [`LICENSE`](LICENSE) for details.

## Citation

## 引用

If you use this dataset, please cite our paper. A citation template is provided in [`CITATION.cff`](CITATION.cff).
