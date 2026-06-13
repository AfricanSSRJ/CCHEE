# 基于数据增强与约束校验的古汉语历史事件抽取

## 环境

```bash
pip install -r requirements.txt
```

## 数据准备

每个章节文件夹下的 `data/` 内含 `*.sample.*` 样例文件用于演示数据格式（详见各 `data/README.md`）。
请将完整数据按相同文件名放入对应章节的 `data/` 目录：事件模式 `schema.txt`、
论元级标注数据 `all_arguments.jsonl`，以及第五章可选的事件类型定义 `event_type_definitions.json`。

论元位置字段可写作 `start`/`end`（与真实标注产出一致），也可写作 `start_offset`/`end_offset`，
二者均被支持，载入后会统一规范化。

## 运行（在项目根目录执行）

第三章 · 论元抽取基线（Span 分类、BIO 序列标注、生成式零样本/少样本）：

```bash
python -m chapter3_argument_extraction.run \
  --schema chapter3_argument_extraction/data/schema.txt \
  --data chapter3_argument_extraction/data/all_arguments.jsonl
```

第五章 · 数据增强（三类模型对比；`--use_gpt4o` 启用 ESAM/RCSM 生成）：

```bash
export OPENAI_API_KEY=...
python -m chapter5_data_augmentation.run \
  --schema chapter5_data_augmentation/data/schema.txt \
  --data chapter5_data_augmentation/data/all_arguments.jsonl \
  --definitions chapter5_data_augmentation/data/event_type_definitions.json \
  --use_gpt4o
```

不设置 `--use_gpt4o` 时，第五章仅运行 Baseline、Random Oversampling 与 EDA 三种对比设置。

第五章新增/补充选项：
- `--longtail_ratio`：长尾划分的累计贡献阈值（二八原则，默认 `0.7`）。
- `--fast_qc`：跳过第三级（模型层）批次质量控制，仅保留事件层/论元层结构校验以加速；
  默认执行完整三级质量控制。第三级会在候选训练集上反复训练轻量探针模型，开销较大，
  快速联调或算力受限时建议加上 `--fast_qc`。

启用 `--use_gpt4o` 时，ESAM 与 RCSM 的增强样本仅生成一次并被主实验、消融实验全程复用，
保证各对比设置之间增强来源一致，同时显著降低大模型调用成本。

## 说明

第三章基于 GPT-4o 的自动标注代码不包含在本项目中；第三、五章所需的论元级标注数据
`all_arguments.jsonl` 即为该标注流程的产物。所有评估均采用严格匹配口径（边界与标签完全一致）。

