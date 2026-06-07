import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate CCHEE JSONL format and spans.")
    parser.add_argument("--input", required=True, help="Path to data.jsonl")
    args = parser.parse_args()

    path = Path(args.input)
    errors = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_no}: invalid JSON: {exc}")
                continue
            text = item.get("text", "")
            for ev in item.get("events", []):
                trig = ev.get("trigger", "")
                start = ev.get("start_offset")
                end = ev.get("end_offset")
                actual = text[start:end] if isinstance(start, int) and isinstance(end, int) else None
                if actual != trig:
                    errors.append(
                        f"Line {line_no}, event {ev.get('id')}: "
                        f"trigger span mismatch: {trig!r} vs {actual!r}"
                    )
                for arg in ev.get("arguments", []):
                    arg_text = arg.get("text", "")
                    arg_start = arg.get("start")
                    arg_end = arg.get("end")
                    actual_arg = (
                        text[arg_start:arg_end]
                        if isinstance(arg_start, int) and isinstance(arg_end, int)
                        else None
                    )
                    if actual_arg != arg_text:
                        errors.append(
                            f"Line {line_no}, event {ev.get('id')}, role {arg.get('role')}: "
                            f"argument span mismatch: {arg_text!r} vs {actual_arg!r}"
                        )

    print(f"Checked {total} JSONL records.")
    if errors:
        print(f"Found {len(errors)} errors:")
        for error in errors[:100]:
            print("-", error)
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more errors omitted")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
