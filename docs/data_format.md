# Data Format

The dataset is stored in JSON Lines format. Each line is a sentence-level annotation instance.

数据集采用 JSON Lines 格式存储，每一行为一个句子级标注实例。

## Sentence-level Fields

| Field | Type | Description |
|---|---|---|
| `sen_id` | integer | Sentence identifier. |
| `doc_id` | integer/string | Document identifier. |
| `text` | string | Classical Chinese sentence. |
| `events` | list | Event mentions annotated in the sentence. |

## Event-level Fields

| Field | Type | Description |
|---|---|---|
| `id` | integer | Event mention identifier. |
| `trigger` | string | Event trigger text. |
| `label` | string | Fine-grained event type. |
| `start_offset` | integer | Trigger start offset. |
| `end_offset` | integer | Trigger end offset. |
| `arguments` | list | Argument mentions of the event. |

## Argument-level Fields

| Field | Type | Description |
|---|---|---|
| `role` | string | Argument role. |
| `text` | string | Argument text span. |
| `start` | integer | Argument start offset. |
| `end` | integer | Argument end offset. |

## Offset Convention

All offsets are character-level offsets in the sentence. The `start` index is inclusive and the `end` index is exclusive: `[start, end)`.

所有位置均为句子内的字符级偏移量，采用左闭右开区间 `[start, end)`。

## Example

```json
{
  "sen_id": 12,
  "doc_id": 32025,
  "text": "三年，蒙骜攻韩，取十三城。",
  "events": [
    {
      "id": 23,
      "trigger": "攻",
      "label": "军事-作战-攻击-征伐",
      "start_offset": 5,
      "end_offset": 6,
      "arguments": [
        {
          "role": "发生时间",
          "text": "三年",
          "start": 0,
          "end": 2
        },
        {
          "role": "攻击方",
          "text": "蒙骜",
          "start": 3,
          "end": 5
        },
        {
          "role": "受击方",
          "text": "韩",
          "start": 6,
          "end": 7
        }
      ]
    },
    {
      "id": 24,
      "trigger": "取",
      "label": "军事-停战-战胜",
      "start_offset": 8,
      "end_offset": 9,
      "arguments": [
        {
          "role": "发生时间",
          "text": "三年",
          "start": 0,
          "end": 2
        },
        {
          "role": "战胜方",
          "text": "蒙骜",
          "start": 3,
          "end": 5
        },
        {
          "role": "战败方",
          "text": "韩",
          "start": 6,
          "end": 7
        },
        {
          "role": "发生地点",
          "text": "十三城",
          "start": 9,
          "end": 12
        }
      ]
    }
  ]
}
```
