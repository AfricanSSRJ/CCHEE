# Examples

The following examples are selected from `data/data.jsonl`. Offsets use character-level half-open intervals `[start, end)`.

## Example: sen_id=1, doc_id=32025

Text:

```text
庄襄王为秦质子于赵，见吕不韦姬，悦而取之，生始皇。
```

Events:

- Event type: `交流-个人交流-见面`
  - Trigger: `见` [10, 11)
  - Arguments:
  - 见面人: 庄襄王 [0, 3)
  - 见面人: 吕不韦姬 [11, 15)

- Event type: `人生-结婚`
  - Trigger: `取` [18, 19)
  - Arguments:
  - 新郎: 庄襄王 [0, 3)
  - 新娘: 吕不韦姬 [11, 15)

- Event type: `人生-出生`
  - Trigger: `生` [21, 22)
  - Arguments:
  - 出生人物: 始皇 [22, 24)
  - 母亲: 吕不韦姬 [11, 15)
  - 父亲: 庄襄王 [0, 3)

## Example: sen_id=12, doc_id=32025

Text:

```text
三年，蒙骜攻韩，取十三城。
```

Events:

- Event type: `军事-作战-攻击-征伐`
  - Trigger: `攻` [5, 6)
  - Arguments:
  - 发生时间: 三年 [0, 2)
  - 攻击方: 蒙骜 [3, 5)
  - 受击方: 韩 [6, 7)

- Event type: `军事-停战-战胜`
  - Trigger: `取` [8, 9)
  - Arguments:
  - 发生时间: 三年 [0, 2)
  - 战胜方: 蒙骜 [3, 5)
  - 战败方: 韩 [6, 7)
  - 发生地点: 十三城 [9, 12)

## Example: sen_id=50, doc_id=32025

Text:

```text
十年，相国吕不韦坐嫪毐免。
```

Events:

- Event type: `法律-犯罪`
  - Trigger: `坐` [8, 9)
  - Arguments:
  - 发生时间: 十年 [0, 2)
  - 犯罪人员: 吕不韦 [5, 8)
  - 犯罪原因: 嫪毐 [9, 11)

- Event type: `职位-官位-免职`
  - Trigger: `免` [11, 12)
  - Arguments:
  - 发生时间: 十年 [0, 2)
  - 被免职人: 吕不韦 [5, 8)

