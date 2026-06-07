import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Count CCHEE dataset statistics.")
    parser.add_argument("--input", required=True, help="Path to data.jsonl")
    parser.add_argument("--schema", default=None, help="Path to event_schema.json")
    args = parser.parse_args()

    records = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    event_counter = Counter()
    role_counter = Counter()
    trigger_counter = Counter()
    arg_count = 0
    event_mentions = 0
    for item in records:
        for event in item.get("events", []):
            event_mentions += 1
            event_counter[event.get("label", "")] += 1
            trigger_counter[event.get("trigger", "")] += 1
            for arg in event.get("arguments", []):
                arg_count += 1
                role_counter[arg.get("role", "")] += 1

    schema_types = None
    if args.schema:
        with Path(args.schema).open("r", encoding="utf-8") as f:
            schema_types = len(json.load(f))

    stats = {
        "documents": len({r.get("doc_id") for r in records}),
        "sentences": len(records),
        "event_mentions": event_mentions,
        "event_types_in_schema": schema_types,
        "event_types_in_data": len(event_counter),
        "argument_roles_in_data": len(role_counter),
        "arguments": arg_count,
        "top_event_types": event_counter.most_common(20),
        "top_argument_roles": role_counter.most_common(20),
        "top_triggers": trigger_counter.most_common(20),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
