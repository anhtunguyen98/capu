import argparse
import json
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from eval_compare import load_test, report, update_counts


LABELS = [f"{case}|{punct}" for case in ("KEEP", "CAPITAL", "UPPER")
          for punct in ("NONE", ".", ",", ":", "?")]


def create_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=options,
                                providers=["CPUExecutionProvider"])


def predicted_events(label):
    case, punct = label.split("|")
    events = set()
    if case != "KEEP":
        events.add("Upper")
    if punct != "NONE":
        events.add(punct)
    if not events:
        events.add("none")
    return events


def run_batch(session, tokenizer, words_batch, max_length):
    encoded = tokenizer(words_batch, is_split_into_words=True, padding=True,
                        truncation=True, max_length=max_length,
                        return_tensors="np")
    logits = session.run(["logits"], {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    })[0]
    return encoded, logits.argmax(-1)


def evaluate(args, examples, session, tokenizer):
    counts = defaultdict(lambda: defaultdict(int))
    for start in range(0, len(examples), args.batch_size):
        chunk = examples[start:start + args.batch_size]
        encoded, predictions = run_batch(
            session, tokenizer, [x[0] for x in chunk], args.max_length)
        for row, (_, gold) in enumerate(chunk):
            pred, previous = [], None
            for token_index, word_id in enumerate(encoded.word_ids(row)):
                if word_id is None or word_id == previous:
                    continue
                label = LABELS[int(predictions[row, token_index])]
                pred.append(predicted_events(label))
                previous = word_id
            pred.extend([{"none"}] * (len(gold) - len(pred)))
            update_counts(counts, gold, pred)
    rows = report(counts)
    edit_names = [name for name in ("Upper", ".", ",", ":", "?")]
    tp = sum(counts[name]["tp"] for name in edit_names)
    fp = sum(counts[name]["fp"] for name in edit_names)
    fn = sum(counts[name]["fn"] for name in edit_names)
    micro = 2 * tp / max(1, 2 * tp + fp + fn)
    macro = sum(row["f1-score"] for row in rows
                if row["label"] != "none") / len(edit_names)
    return {"micro_f1_no_none": micro, "macro_f1_no_none": macro,
            "classes": rows}


def benchmark(args, examples, session, tokenizer):
    rows = examples[:args.benchmark_samples + 1]
    run_batch(session, tokenizer, [rows[0][0]], args.max_length)
    times = []
    for words, _ in rows[1:]:
        start = time.perf_counter()
        run_batch(session, tokenizer, [words], args.max_length)
        times.append((time.perf_counter() - start) * 1000)
    return {"samples": len(times), "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "samples_per_second": 1000 / float(np.mean(times))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/videberta-xsmall-capu/onnx-int8/model.int8.onnx")
    p.add_argument("--tokenizer", default="outputs/videberta-xsmall-capu/onnx-int8")
    p.add_argument("--test_file", default="data/news_capu_10m/test.txt")
    p.add_argument("--output", default="outputs/videberta-xsmall-capu/onnx-int8/eval.json")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--benchmark_samples", type=int, default=500)
    args = p.parse_args()
    examples = load_test(args.test_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    session = create_session(args.model, args.threads)
    rows = evaluate(args, examples, session, tokenizer)
    timing = benchmark(args, examples, session, tokenizer)
    result = {"provider": session.get_providers()[0], "threads": args.threads,
              "test_samples": len(examples), "metrics": rows,
              "batch_1_timing": timing}
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
