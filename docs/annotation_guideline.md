# Annotation Guideline

This document summarizes the annotation principles used by the dataset.

## Annotation Targets

For each sentence, annotators identify:

1. Event triggers.
2. Fine-grained event types.
3. Event arguments.
4. Argument roles.
5. Character-level spans for triggers and arguments.

## Event Type

Each event mention is assigned one event type from the predefined schema in `schema/schema.md`.

## Trigger

A trigger is the word or phrase that most directly evokes the event.

## Argument and Role

Arguments are text spans participating in the event. Each argument is assigned a role from the role set of the corresponding event type.

## Span Convention

All spans are character-level half-open intervals `[start, end)`.

## Notes

- One sentence may contain multiple event mentions.
- One event may contain multiple arguments with the same role.
- If an argument is not explicitly recoverable from the sentence, it is not annotated in the released instance.
