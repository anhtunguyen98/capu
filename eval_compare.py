import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


NAMES = ("Upper", ".", ",", ":", "?", "none")
SEP = "SEPL|||SEPR"
OP_SEP = "SEPL__SEPR"
PUNCTS = {".", ",", ":", "?"}
TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|[.,:?]", re.UNICODE)


def events_from_gold(word, operations):
    result = set()
    for operation in operations.split(OP_SEP):
        if operation in ("$TRANSFORM_CASE_CAPITAL", "$TRANSFORM_CASE_UPPER"):
            result.add("Upper")
        elif operation in ("$TRANSFORM_CASE_CAPITAL_1", "$TRANSFORM_CASE_UPPER_-1"):
            result.add("Complex-Upper")
        elif operation.startswith("$REPLACE_"):
            target = operation[len("$REPLACE_"):]
            if target.lower() == word.lower():
                if target in (word.capitalize(), word.upper()):
                    result.add("Upper")
                else:
                    result.add("Complex-Upper")
        elif operation.startswith("$APPEND_"):
            punct = operation[len("$APPEND_"):]
            if punct in PUNCTS:
                result.add(punct)
    if not result:
        result.add("none")
    return result


def load_test(path):
    examples = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            words, gold = [], []
            for field in line.rstrip().split()[1:]:
                word, operations = field.split(SEP, 1)
                words.append(word)
                gold.append(events_from_gold(word, operations))
            examples.append((words, gold))
    return examples


def output_events(source_words, output):
    output_words, punctuation = [], []
    for token in TOKEN_RE.findall(output):
        if token in PUNCTS:
            if punctuation:
                punctuation[-1] = token
        else:
            output_words.append(token)
            punctuation.append(None)
    results = []
    for index, source in enumerate(source_words):
        events = set()
        if index < len(output_words):
            target = output_words[index]
            if target.lower() == source.lower() and target != source:
                if target in (source.capitalize(), source.upper()):
                    events.add("Upper")
                else:
                    events.add("Complex-Upper")
            if punctuation[index] in PUNCTS:
                events.add(punctuation[index])
        if not events:
            events.add("none")
        results.append(events)
    return results


def update_counts(counts, gold_rows, pred_rows):
    for gold, pred in zip(gold_rows, pred_rows):
        for name in NAMES:
            g, p = name in gold, name in pred
            counts[name]["support"] += int(g)
            counts[name]["tp"] += int(g and p)
            counts[name]["fp"] += int(not g and p)
            counts[name]["fn"] += int(g and not p)


def report(counts):
    rows = []
    for name in NAMES:
        c = counts[name]
        precision = c["tp"] / max(1, c["tp"] + c["fp"])
        recall = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * precision * recall / max(1e-15, precision + recall)
        rows.append({"label": name, "precision": precision, "recall": recall,
                     "f1-score": f1, "support": c["support"]})
    return rows


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def eval_ours(args, examples):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint).to("cuda").eval()
    counts = defaultdict(lambda: defaultdict(int))
    for start in range(0, len(examples), args.batch_size):
        chunk = examples[start:start + args.batch_size]
        words_batch = [x[0] for x in chunk]
        encoded = tokenizer(words_batch, is_split_into_words=True, padding=True,
                            truncation=True, max_length=args.max_length,
                            return_tensors="pt")
        word_ids = [encoded.word_ids(i) for i in range(len(chunk))]
        logits = model(**{k: v.to("cuda") for k, v in encoded.items()}).logits.argmax(-1).cpu()
        for row, (_, gold) in enumerate(chunk):
            pred = []
            previous = None
            for token_index, word_id in enumerate(word_ids[row]):
                if word_id is None or word_id == previous:
                    continue
                case, punct = model.config.id2label[int(logits[row, token_index])].split("|")
                events = set()
                if case != "KEEP": events.add("Upper")
                if punct != "NONE": events.add(punct)
                if not events: events.add("none")
                pred.append(events)
                previous = word_id
            # A production restorer leaves words beyond tokenizer truncation
            # unchanged. Count those positions as `none` so every model is
            # evaluated against the exact same gold support.
            pred.extend([{"none"}] * (len(gold) - len(pred)))
            update_counts(counts, gold, pred)
    timing_rows = examples[:args.benchmark_samples]
    for words, _ in timing_rows[:args.warmup_samples]:
        encoded = tokenizer([words], is_split_into_words=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
        model(**{key: value.to("cuda") for key, value in encoded.items()})
    torch.cuda.synchronize()
    started = time.perf_counter()
    for words, _ in timing_rows:
        encoded = tokenizer([words], is_split_into_words=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
        model(**{key: value.to("cuda") for key, value in encoded.items()})
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000 / len(timing_rows)
    return {"classes": report(counts), "parameters": parameter_count(model),
            "latency_ms_per_sample_batch1": latency_ms}


@torch.no_grad()
def eval_dragon(args, examples):
    code_dir = args.dragon_code_dir or args.dragon_dir
    sys.path.insert(0, code_dir)
    from gec_model import GecBERTModel
    model = GecBERTModel(vocab_path=os.path.join(code_dir, "vocabulary"),
                         model_paths=args.dragon_dir, split_chunk=True, device="cuda")
    counts = defaultdict(lambda: defaultdict(int))
    for start in range(0, len(examples), args.dragon_batch_size):
        chunk = examples[start:start + args.dragon_batch_size]
        words_batch = [x[0] for x in chunk]
        outputs = model(words_batch, is_split_into_words=True)
        for (words, gold), output in zip(chunk, outputs):
            pred = output_events(words, output)
            update_counts(counts, gold, pred)
        if start % 1000 == 0:
            print(f"dragon {start}/{len(examples)}", flush=True)
    timing_rows = examples[:args.benchmark_samples]
    for words, _ in timing_rows[:args.warmup_samples]:
        model([words], is_split_into_words=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for words, _ in timing_rows:
        model([words], is_split_into_words=True)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000 / len(timing_rows)
    parameters = sum(parameter_count(item) for item in model.models)
    return {"classes": report(counts), "parameters": parameters,
            "latency_ms_per_sample_batch1": latency_ms}


def print_table(title, rows):
    print("\n" + title)
    print("label\tprecision\trecall\tf1-score\tsupport")
    for row in rows:
        print(f"{row['label']}\t{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f1-score']:.4f}\t{row['support']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", default="data/news_capu_10m/test.txt")
    p.add_argument("--checkpoint", default="outputs/videberta-xsmall-capu/checkpoint-117187")
    p.add_argument("--dragon_dir", default="/workspace/.hf_home/hub/models--dragonSwing--vibert-capu/snapshots/261c60f2c30b02455dfce21a43c3ef14fc26992c")
    p.add_argument("--dragon_code_dir")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--dragon_batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--benchmark_samples", type=int, default=500)
    p.add_argument("--warmup_samples", type=int, default=20)
    p.add_argument("--output", default="outputs/comparison_metrics.json")
    args = p.parse_args()
    examples = load_test(args.test_file)
    ours = eval_ours(args, examples)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output + ".partial", "w", encoding="utf-8") as f:
        json.dump({"test_samples": len(examples), "ours": ours}, f,
                  ensure_ascii=False, indent=2)
    torch.cuda.empty_cache()
    dragon = eval_dragon(args, examples)
    result = {"test_samples": len(examples), "ours": ours, "dragon": dragon}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print_table("ViDeBERTa-xsmall CAPU", ours["classes"])
    print(f"parameters: {ours['parameters']}")
    print(f"latency_ms_per_sample_batch1: {ours['latency_ms_per_sample_batch1']:.4f}")
    print_table("dragonSwing/vibert-capu", dragon["classes"])
    print(f"parameters: {dragon['parameters']}")
    print(f"latency_ms_per_sample_batch1: {dragon['latency_ms_per_sample_batch1']:.4f}")
