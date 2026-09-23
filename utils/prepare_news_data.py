"""Stream corpora into resumable, fixed-word CAPU training shards."""

import argparse, hashlib, html, json, re, shutil, unicodedata
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PUNCTUATION = {".", ",", ":", "?"}
SEP, OP_SEP = "SEPL|||SEPR", "SEPL__SEPR"
TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|[.,:?]", re.UNICODE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
HTML_RE = re.compile(r"<[^>]+>|\b(?:href|onclick|onmouseout)\s*=", re.I)
SPACE_RE = re.compile(r"\s+")

def normalize_text(value):
    text = unicodedata.normalize("NFC", html.unescape(str(value)))
    text = text.replace("…", ".").replace("!", ".").replace(";", ",")
    text = EMAIL_RE.sub(" ", URL_RE.sub(" ", text))
    return SPACE_RE.sub(" ", text).strip()

def document_split(key, seed, dev_ratio, test_ratio):
    digest = hashlib.blake2b(f"{seed}\0{key}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / 2**64
    return "test" if value < test_ratio else (
        "dev" if value < test_ratio + dev_ratio else "train")

def word_records(text):
    records = []
    for token in TOKEN_RE.findall(text):
        if token in PUNCTUATION:
            if records:
                records[-1][1] = token
        else:
            records.append([token, "NONE"])
    return records

def chunks(records, words, overlap, min_words):
    for start in range(0, len(records), words - overlap):
        chunk = records[start:start + words]
        if len(chunk) < min_words:
            break
        yield chunk
        if start + words >= len(records):
            break

def case_operation(word):
    lower = word.lower()
    if word == lower:
        return "$KEEP"
    if word == lower.capitalize():
        return "$TRANSFORM_CASE_CAPITAL"
    if word == lower.upper():
        return "$TRANSFORM_CASE_UPPER"
    return "$REPLACE_" + word

def tagged_line(chunk):
    fields = ["$START" + SEP + "$KEEP"]
    for word, punct in chunk:
        ops = [case_operation(word)]
        if punct != "NONE":
            ops.append("$APPEND_" + punct)
        fields.append(word.lower() + SEP + OP_SEP.join(ops))
    return " ".join(fields)

class ShardWriter:
    def __init__(self, root, split, shard_size):
        self.root, self.split, self.shard_size = Path(root), split, shard_size
        self.count = self.shard_index = 0
        self.handle = None
    def write(self, line):
        if self.count % self.shard_size == 0:
            if self.handle:
                self.handle.close()
            path = self.root / self.split / f"part-{self.shard_index:05d}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("w", encoding="utf-8")
            self.shard_index += 1
        self.handle.write(line + "\n")
        self.count += 1
    def close(self):
        if self.handle:
            self.handle.close()

def load_stream(source):
    kwargs = dict(path=source["dataset"], split=source.get("split", "train"),
                  streaming=True)
    if source.get("config"):
        kwargs["name"] = source["config"]
    dataset = load_dataset(**kwargs)
    buffer_size = int(source.get("shuffle_buffer", 0))
    return dataset.shuffle(seed=source.get("seed", 42),
                           buffer_size=buffer_size) if buffer_size else dataset

def prepare(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(args.sources).read_text())
    writers = {s: ShardWriter(output, s, args.shard_size)
               for s in ("train", "dev", "test")}
    totals, marks = Counter(), {s: Counter() for s in writers}
    minimum_free = args.min_free_gb * 1024**3
    try:
        for source in sources:
            accepted = Counter()
            limits = {"train": source["train_samples"],
                      "dev": source.get("dev_samples", 0),
                      "test": source.get("test_samples", 0)}
            progress = tqdm(total=sum(limits.values()), desc=source["name"])
            for row in load_stream(source):
                if all(accepted[s] >= limits[s] for s in limits):
                    break
                raw = row.get(source.get("text_column", "text"))
                if not raw or HTML_RE.search(str(raw)):
                    continue
                if row.get("int_score", 99) < source.get("min_int_score", 0):
                    continue
                if row.get("language_score", 1.0) < source.get("min_language_score", 0):
                    continue
                text = normalize_text(raw)
                key = row.get("id") or row.get("url") or hashlib.blake2b(
                    text.encode(), digest_size=12).hexdigest()
                split = document_split(key, args.seed, args.dev_ratio, args.test_ratio)
                if accepted[split] >= limits[split]:
                    continue
                for chunk in chunks(word_records(text), args.chunk_words,
                                    args.overlap_words, args.min_words):
                    if accepted[split] >= limits[split]:
                        break
                    writers[split].write(tagged_line(chunk))
                    accepted[split] += 1
                    totals[f"{source['name']}:{split}"] += 1
                    marks[split].update(p for _, p in chunk)
                    progress.update()
                    if sum(totals.values()) % args.disk_check_interval == 0 and \
                            shutil.disk_usage(output).free < minimum_free:
                        raise RuntimeError("Disk guard reached; partial shards kept")
            progress.close()
            if any(accepted[s] < limits[s] for s in limits):
                raise RuntimeError(f"Source exhausted: {source['name']} "
                                   f"{dict(accepted)} / {limits}")
    finally:
        for writer in writers.values():
            writer.close()
    manifest = {"chunk_words": args.chunk_words, "overlap_words": args.overlap_words,
                "seed": args.seed, "sources": sources, "samples": dict(totals),
                "punctuation": {s: dict(v) for s, v in marks.items()}}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sources", required=True)
    p.add_argument("--output_dir", default="data/capu_bilingual_15m")
    p.add_argument("--chunk_words", type=int, default=50)
    p.add_argument("--overlap_words", type=int, default=10)
    p.add_argument("--min_words", type=int, default=10)
    p.add_argument("--shard_size", type=int, default=100_000)
    p.add_argument("--dev_ratio", type=float, default=.002)
    p.add_argument("--test_ratio", type=float, default=.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_free_gb", type=float, default=40)
    p.add_argument("--disk_check_interval", type=int, default=10_000)
    a = p.parse_args()
    if not 0 <= a.overlap_words < a.chunk_words:
        p.error("overlap_words must be in [0, chunk_words)")
    prepare(a)
