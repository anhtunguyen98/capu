"""Full integrity audit for sharded CAPU data before expensive training."""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from train_hf import parse_line


def main(args):
    root = Path(args.data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = Counter()
    for source in manifest["sources"]:
        for split in ("train", "dev", "test"):
            expected[split] += int(source.get(f"{split}_samples", 0))
    report = {"expected": dict(expected), "splits": {}}
    for split in ("train", "dev", "test"):
        files = sorted(glob.glob(str(root / split / "*.txt")))
        counts, punctuation = Counter(), Counter()
        for filename in files:
            with open(filename, encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    words, labels = parse_line(line)
                    if len(words) != len(labels):
                        raise ValueError(f"Alignment error: {filename}:{line_number}")
                    if not args.min_words <= len(words) <= args.max_words:
                        raise ValueError(f"Length error: {filename}:{line_number} "
                                         f"has {len(words)} words")
                    counts["samples"] += 1
                    counts["words"] += len(words)
                    for field in line.rstrip().split()[1:]:
                        if "$APPEND_?" in field: punctuation["?"] += 1
                        elif "$APPEND_." in field: punctuation["."] += 1
                        elif "$APPEND_," in field: punctuation[","] += 1
                        elif "$APPEND_:" in field: punctuation[":"] += 1
                        else: punctuation["NONE"] += 1
        if counts["samples"] != expected[split]:
            raise ValueError(f"{split}: {counts['samples']} != {expected[split]}")
        report["splits"][split] = {
            "shards": len(files), **dict(counts),
            "punctuation": dict(punctuation)}
    target = root / "audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/capu_bilingual_15m")
    p.add_argument("--min_words", type=int, default=10)
    p.add_argument("--max_words", type=int, default=50)
    main(p.parse_args())
