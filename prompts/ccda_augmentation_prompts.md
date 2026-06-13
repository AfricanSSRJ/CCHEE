# CCDA Augmentation Prompts

This document records the prompt templates used by the CCDA data augmentation
procedure. The templates are intended to describe the generation constraints and
output formats used in the paper. They do not contain API keys or author
information.

## Event Sample Augmentation Module

ESAM generates complete event samples for rare event types.

### System Message

```text
You are an assistant for Classical Chinese historical event extraction.
Given an event type, its trigger semantics, and its argument-role schema,
generate a Classical Chinese sentence that expresses exactly one event of the
specified type. The generated sentence should follow the style of historical
records and should avoid modern Chinese expressions.

The output must be valid JSON only. Do not provide explanations.
```

### User Message Template

```json
{
  "task": "generate_event_sample",
  "event_type": "{EVENT_TYPE}",
  "role_schema": ["{ROLE_1}", "{ROLE_2}", "{ROLE_3}"],
  "constraints": {
    "language_style": "Classical Chinese historical prose",
    "include_trigger": true,
    "include_arguments": true,
    "avoid_modern_expressions": true,
    "output_spans_from_sentence": true
  },
  "output_json_schema": {
    "sentence": "{GENERATED_SENTENCE}",
    "event": {
      "trigger": "{TRIGGER_TEXT}",
      "event_type": "{EVENT_TYPE}",
      "arguments": [
        {
          "role": "{ROLE_NAME}",
          "text": "{ARGUMENT_TEXT}"
        }
      ]
    }
  }
}
```

## Role Context Supplement Module

RCSM supplements missing argument contexts for samples in common event types.

### System Message

```text
You are an assistant for Classical Chinese historical event extraction.
Given a Classical Chinese sentence, an event trigger, an event type, existing
arguments, and missing roles, supplement only the missing argument contexts.
The supplemented content must be concise, compatible with the original sentence,
and written in Classical Chinese style.

The output must be valid JSON only. Do not provide explanations.
```

### User Message Template

```json
{
  "task": "supplement_missing_roles",
  "sentence": "{ORIGINAL_SENTENCE}",
  "event_type": "{EVENT_TYPE}",
  "trigger": {
    "text": "{TRIGGER_TEXT}",
    "start": "{TRIGGER_START}",
    "end": "{TRIGGER_END}"
  },
  "existing_arguments": [
    {
      "role": "{ROLE_NAME}",
      "text": "{ARGUMENT_TEXT}"
    }
  ],
  "missing_roles": ["{MISSING_ROLE_1}", "{MISSING_ROLE_2}"],
  "constraints": {
    "language_style": "Classical Chinese historical prose",
    "max_supplement_length": 20,
    "preserve_original_event_type": true,
    "avoid_modern_expressions": true
  },
  "output_json_schema": {
    "supplemented_sentence": "{SENTENCE_WITH_SUPPLEMENTED_CONTEXT}",
    "supplemented_arguments": [
      {
        "role": "{ROLE_NAME}",
        "text": "{ARGUMENT_TEXT}"
      }
    ]
  }
}
```

## Generation Settings

The paper uses GPT-4o for data augmentation with:

```text
temperature = 0.7
max_tokens = 512
```

Generated samples are filtered at event, argument, and model levels before being
merged with the original training data.
