"""Patch run_ablation.py to support consistent metadata keys across methods.

Run from repository root after installing the retry metadata patch:
    python patches/fix_ablation_csv_fields.py
"""
from pathlib import Path

path = Path("run_ablation.py")
if not path.is_file():
    raise SystemExit("Run this script from the repository root; run_ablation.py was not found.")
text = path.read_text(encoding="utf-8")
start = text.find("        writer = csv.DictWriter(")
end_anchor = "        writer.writerows(rows)"
end = text.find(end_anchor, start)
if start < 0 or end < 0:
    if "fieldnames = list(dict.fromkeys(key for row in rows for key in row))" in text:
        print("CSV writer already appears patched; no change made.")
        raise SystemExit(0)
    raise SystemExit("Could not locate the CSV writer block. No file was changed.")
end += len(end_anchor)
replacement = '''        # Some methods emit attempt-trace metadata that other methods do not.
        # Build a union of keys so DictWriter does not fail on extra fields.
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        serializable_rows = []
        for row in rows:
            serializable = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    serializable[key] = json.dumps(value, ensure_ascii=False)
                else:
                    serializable[key] = value
            serializable_rows.append(serializable)

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(serializable_rows)'''
backup = path.with_suffix(path.suffix + ".before_csv_fix.bak")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
print(f"Patched {path}. Backup: {backup}")
