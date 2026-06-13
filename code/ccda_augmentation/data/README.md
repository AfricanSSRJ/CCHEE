# 数据说明

本目录存放古汉语历史事件抽取数据集。仓库内仅提供少量样例文件（`*.sample.*`）用于演示数据格式，
完整数据请按下列格式放入对应文件名后再运行实验。

## schema.txt

事件模式定义，JSON 数组。每个元素描述一种细粒度事件类型及其论元角色集合：

```json
[
  {"event_type_id": 1, "event_type": "人生-出生", "role_list": [{"role": "发生时间"}, {"role": "发生地点"}, {"role": "出生人物"}, {"role": "母亲"}, {"role": "父亲"}]}
]
```

其中 `event_type` 为完整层级标签字符串。代码以其首段作为粗粒度类型（如 `人生`），
末段作为细粒度类型（如 `出生`），并据此构建粗细粒度层级与论元角色空间。

## all.jsonl

事件检测数据（第四章使用），每行一个句子级样本：

```json
{"sen_id": 1, "doc_id": 32025, "text": "庄襄王为秦质子于赵，见吕不韦姬，悦而取之，生始皇。",
 "events": [{"id": 1, "trigger": "见", "label": "交流-个人交流-见面", "start_offset": 10, "end_offset": 11}]}
```

`label` 为完整层级标签字符串，`start_offset`/`end_offset` 为触发词在 `text` 中的字符位置（左闭右开）。

## all_arguments.jsonl

论元级标注数据（第三章、第五章使用）。在 `all.jsonl` 基础上为每个事件补充 `arguments` 字段，
该文件由第三章基于 GPT-4o 的自动标注流程产出：

```json
{"sen_id": 1, "doc_id": 32025, "text": "...",
 "events": [{"id": 1, "trigger": "见", "label": "交流-个人交流-见面", "start_offset": 10, "end_offset": 11,
   "arguments": [{"role": "见面人", "text": "吕不韦姬", "start": 11, "end": 15}]}]}
```

**论元位置字段约定**：每个论元的字符位置可写作 `start`/`end`（与真实自动标注产出一致），
也可写作 `start_offset`/`end_offset`；二者均被支持。代码在数据载入后由
`common/normalize.py` 统一规范为 `start_offset`/`end_offset`，并在提供 `text` 时
按“就近触发词”原则校验与修复偏移，因此使用任一约定都可正常运行。
事件触发词位置统一使用 `start_offset`/`end_offset`。

## translations.jsonl（第四章可选）

句子级现代汉语释义，每行 `{"sen_id": 1, "translation": "..."}`。出于版权考虑不随仓库分发，
缺失时模型自动以空释义运行。

## event_type_definitions.json（第四、五章可选）

事件类型自然语言定义，形如 `{"军事-作战-攻击-征伐": "一方主动出兵攻击另一方的军事行为……"}`。
未提供的类型由代码依据类型名称与角色自动生成模板定义。
