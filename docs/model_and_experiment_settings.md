# Model and Experiment Settings

This document summarizes the model and experiment settings associated with the
dataset and the CCDA data augmentation experiments.

## Automatic Argument Annotation

The released argument annotations were obtained with a GPT-4o assisted
annotation workflow followed by rule-based validation and manual checking.

- Model: GPT-4o
- Temperature: 0
- Max tokens: 512
- Output format: JSON
- Span alignment: exact string matching against the original sentence
- Offset convention: character-level half-open interval `[start, end)`
- Validation rules:
  - the model output must be valid JSON;
  - every argument span must appear in the original sentence;
  - every role name must belong to the predefined role set of the event type.

For samples with multiple matching positions, the nearest match to the event
trigger is selected as a heuristic span alignment rule.

## Baseline Models

The experiments described in the paper involve the following model settings.

| Setting | Description |
|---|---|
| Pipeline-Span | GuwenBERT encoder for trigger/event prediction, followed by a span classifier for argument extraction conditioned on the predicted trigger. |
| Joint-BIO | GuwenBERT encoder with joint trigger and argument prediction in a unified BIO tagging space. |
| Generative | Xunzi-Qwen2-7B based conditional generation model for complete event extraction. |

Related implementation files are available under `code/chapter3_argument_extraction/`
and `code/chapter5_data_augmentation/`.

## Data Split

The dataset is split into training, validation, and test sets at an 8:1:1 ratio
in the paper experiments.

## CCDA Augmentation Settings

CCDA contains an Event Sample Augmentation Module (ESAM), a Role Context
Supplement Module (RCSM), and a three-level quality control mechanism.

- Generation model: GPT-4o
- Temperature: 0.7
- Max tokens: 512
- ESAM target: rare event types
- RCSM target: common event types with missing roles
- ESAM candidate samples: 30 generated candidates for each rare event type
- Quality control: event-level, argument-level, and model-level filtering

In the experiment setting described in the paper, 49 rare event types entered
the ESAM workflow, producing 1,470 candidate samples and 1,286 retained samples
after filtering. RCSM processed common-type samples with supplementable missing
roles and retained 2,316 supplemented samples after filtering.

## Notes on Model Files

This repository does not include pretrained model weights. The experiments use
publicly available base models and task-specific training settings described in
the paper. Large model checkpoints should be downloaded from their official
release pages or model hubs according to their licenses.
