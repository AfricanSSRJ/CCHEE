# Reproducibility Checklist

This repository provides the dataset, schema, validation scripts, statistics,
prompt templates, and experiment-setting descriptions needed to inspect and use
CCHEE.

## Included

- `data/data.jsonl`: sentence-level event and argument annotations.
- `schema/event_schema.json`: machine-readable event type and role schema.
- `schema/schema.md`: human-readable event schema.
- `docs/data_format.md`: field definitions and offset convention.
- `docs/annotation_guideline.md`: annotation principles.
- `docs/examples.md`: representative annotation examples.
- `prompts/gpt4o_argument_annotation_prompt.json`: GPT-4o assisted argument annotation prompt.
- `prompts/ccda_augmentation_prompts.md`: CCDA augmentation prompt templates.
- `scripts/check_format.py`: span and format validation script.
- `scripts/count_statistics.py`: dataset statistics script.
- `statistics/dataset_statistics.md`: dataset statistics.
- `docs/model_and_experiment_settings.md`: model, split, generation, and filtering settings.

## Basic Validation

Check whether trigger and argument spans match the original sentence text:

```bash
python scripts/check_format.py --input data/data.jsonl
```

Generate dataset statistics:

```bash
python scripts/count_statistics.py --input data/data.jsonl --schema schema/event_schema.json
```

## Not Included

The repository does not include large pretrained model weights, API keys, or
private training infrastructure. Public base models should be obtained from
their official releases according to the corresponding licenses.

The repository currently provides data-processing and validation utilities
rather than a full training framework. The model architectures and experiment
settings are described in the paper and summarized in
`docs/model_and_experiment_settings.md`.
