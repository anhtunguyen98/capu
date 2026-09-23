"""Evaluate an original and vocabulary-pruned CAPU model side by side."""

import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from eval_compare import NAMES, load_test, report, update_counts


def prediction_events(model, tokenizer, examples, max_length, device):
    words_batch = [row[0] for row in examples]
    encoded = tokenizer(
        words_batch, is_split_into_words=True, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt")
    word_ids = [encoded.word_ids(row) for row in range(len(examples))]
    logits = model(**{key: value.to(device)
                      for key, value in encoded.items()}).logits.argmax(-1).cpu()
    rows = []
    for row, (words, _) in enumerate(examples):
        predictions, previous = [], None
        for token_index, word_id in enumerate(word_ids[row]):
            if word_id is None or word_id == previous:
                continue
            case, punct = model.config.id2label[
                int(logits[row, token_index])].split("|")
            events = set()
            if case != "KEEP":
                events.add("Upper")
            if punct != "NONE":
                events.add(punct)
            predictions.append(events or {"none"})
            previous = word_id
        predictions.extend([{"none"}] * (len(words) - len(predictions)))
        rows.append(predictions)
    return rows


def aggregate(counts):
    edit_names = [name for name in NAMES if name != "none"]
    tp = sum(counts[name]["tp"] for name in edit_names)
    fp = sum(counts[name]["fp"] for name in edit_names)
    fn = sum(counts[name]["fn"] for name in edit_names)
    micro = 2 * tp / max(1, 2 * tp + fp + fn)
    rows = report(counts)
    macro = sum(row["f1-score"] for row in rows
                if row["label"] != "none") / len(edit_names)
    return {"micro_f1_no_none": micro, "macro_f1_no_none": macro,
            "classes": rows}


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda")
    original_tokenizer = AutoTokenizer.from_pretrained(args.original)
    pruned_tokenizer = AutoTokenizer.from_pretrained(args.pruned)
    original = AutoModelForTokenClassification.from_pretrained(
        args.original).to(device).eval()
    pruned = AutoModelForTokenClassification.from_pretrained(
        args.pruned).to(device).eval()
    output = {}
    for name, path in (("vi", args.vi_test), ("en", args.en_test)):
        examples = load_test(path)
        original_counts = defaultdict(lambda: defaultdict(int))
        pruned_counts = defaultdict(lambda: defaultdict(int))
        differing_positions = 0
        total_positions = 0
        differing_samples = 0
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            original_rows = prediction_events(
                original, original_tokenizer, batch, args.max_length, device)
            pruned_rows = prediction_events(
                pruned, pruned_tokenizer, batch, args.max_length, device)
            for (_, gold), old_row, new_row in zip(
                    batch, original_rows, pruned_rows):
                update_counts(original_counts, gold, old_row)
                update_counts(pruned_counts, gold, new_row)
                differences = sum(a != b for a, b in zip(old_row, new_row))
                differing_positions += differences
                differing_samples += int(differences > 0)
                total_positions += len(gold)
        output[name] = {
            "samples": len(examples),
            "original": aggregate(original_counts),
            "pruned": aggregate(pruned_counts),
            "prediction_differences": {
                "samples": differing_samples,
                "word_positions": differing_positions,
                "total_word_positions": total_positions,
                "rate": differing_positions / max(1, total_positions),
            },
        }
        print(name, json.dumps(output[name], ensure_ascii=False), flush=True)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(output, destination, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--pruned", required=True)
    parser.add_argument("--vi_test", required=True)
    parser.add_argument("--en_test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    evaluate(parser.parse_args())
