"""Scan bilingual CAPU data and prune a Unigram tokenizer/model embedding.

The scanner reads only source words from tagged CAPU lines. Counts are saved
after every shard, so a stopped scan can resume without rereading old shards.
"""

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForTokenClassification, AutoTokenizer


SEP = "SEPL|||SEPR"


def resolve_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.txt"))
    else:
        files = glob.glob(path)
    if not files:
        raise FileNotFoundError(f"No data files matched: {path}")
    return sorted(files)


def source_words(line):
    words = []
    for field in line.rstrip().split()[1:]:  # skip $START
        position = field.find(SEP)
        if position > 0:
            words.append(field[:position])
    return words


def save_counts(path, vi, en, completed, samples, tokens, elapsed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, vi=vi, en=en)
    os.replace(temporary, path)
    state = {
        "completed_files": completed,
        "samples": samples,
        "token_occurrences": tokens,
        "elapsed_seconds": elapsed,
    }
    state_path = path.with_suffix(".state.json")
    temporary_state = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary_state.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary_state, state_path)


def scan(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for corpus scanning")
    vocab_size = len(tokenizer)
    state_path = Path(args.output).with_suffix(".state.json")
    if Path(args.output).exists() and state_path.exists():
        arrays = np.load(args.output)
        vi, en = arrays["vi"], arrays["en"]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed = list(state["completed_files"])
        samples = int(state["samples"])
        token_occurrences = int(state["token_occurrences"])
        prior_elapsed = float(state.get("elapsed_seconds", 0))
    else:
        vi = np.zeros(vocab_size, dtype=np.uint64)
        en = np.zeros(vocab_size, dtype=np.uint64)
        completed, samples, token_occurrences, prior_elapsed = [], 0, 0, 0.0

    completed_set = set(completed)
    files = resolve_files(args.data)
    started = time.perf_counter()
    for file_index, filename in enumerate(files):
        if filename in completed_set:
            continue
        shard_id = int(Path(filename).stem.rsplit("-", 1)[-1])
        counts = en if shard_id >= args.english_start_shard else vi
        batch = []
        with open(filename, encoding="utf-8") as source:
            for line in source:
                words = source_words(line)
                if words:
                    batch.append(words)
                if len(batch) >= args.batch_size:
                    encodings = tokenizer._tokenizer.encode_batch(
                        batch, is_pretokenized=True, add_special_tokens=False)
                    ids = np.fromiter(
                        (token_id for item in encodings for token_id in item.ids),
                        dtype=np.int64)
                    counts += np.bincount(ids, minlength=vocab_size).astype(np.uint64)
                    samples += len(batch)
                    token_occurrences += len(ids)
                    batch.clear()
            if batch:
                encodings = tokenizer._tokenizer.encode_batch(
                    batch, is_pretokenized=True, add_special_tokens=False)
                ids = np.fromiter(
                    (token_id for item in encodings for token_id in item.ids),
                    dtype=np.int64)
                counts += np.bincount(ids, minlength=vocab_size).astype(np.uint64)
                samples += len(batch)
                token_occurrences += len(ids)

        completed.append(filename)
        elapsed = prior_elapsed + time.perf_counter() - started
        save_counts(args.output, vi, en, completed, samples,
                    token_occurrences, elapsed)
        unique_vi = int(np.count_nonzero(vi))
        unique_en = int(np.count_nonzero(en))
        rate = samples / max(1e-9, elapsed)
        print(json.dumps({
            "shards": len(completed), "total_shards": len(files),
            "samples": samples, "samples_per_second": round(rate, 1),
            "unique_vi": unique_vi, "unique_en": unique_en,
            "unique_union": int(np.count_nonzero(vi + en)),
        }), flush=True)


def remap_tokenizer(tokenizer_path, output_dir, keep_ids):
    tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
    raw = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    if raw["model"]["type"] != "Unigram":
        raise RuntimeError("Only tokenizer.json Unigram models are supported")
    old_vocab = raw["model"]["vocab"]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
    raw["model"]["vocab"] = [old_vocab[index] for index in keep_ids]
    raw["model"]["unk_id"] = old_to_new[raw["model"]["unk_id"]]
    for token in raw.get("added_tokens", []):
        token["id"] = old_to_new[token["id"]]
    special = raw.get("post_processor", {}).get("special_tokens", {})
    for value in special.values():
        value["ids"] = [old_to_new[index] for index in value["ids"]]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pruned_file = output_dir / "tokenizer.json"
    pruned_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    backend = Tokenizer.from_file(str(pruned_file))
    original = AutoTokenizer.from_pretrained(tokenizer_path)
    original._tokenizer = backend
    original.save_pretrained(output_dir)
    return old_to_new


def prune(args):
    arrays = np.load(args.counts)
    vi, en = arrays["vi"], arrays["en"]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    special_ids = set(tokenizer.all_special_ids)
    keep = (vi >= args.min_vi_count) | (en >= args.min_en_count)
    for token_id in special_ids:
        keep[token_id] = True
    keep_ids = np.flatnonzero(keep).tolist()
    old_to_new = remap_tokenizer(args.checkpoint, args.output_dir, keep_ids)

    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    old_embedding = model.get_input_embeddings()
    new_embedding = torch.nn.Embedding(
        len(keep_ids), old_embedding.embedding_dim,
        padding_idx=old_to_new.get(old_embedding.padding_idx),
        max_norm=old_embedding.max_norm, norm_type=old_embedding.norm_type,
        scale_grad_by_freq=old_embedding.scale_grad_by_freq,
        sparse=old_embedding.sparse,
        device=old_embedding.weight.device, dtype=old_embedding.weight.dtype)
    with torch.no_grad():
        indices = torch.tensor(keep_ids, device=old_embedding.weight.device)
        new_embedding.weight.copy_(old_embedding.weight.index_select(0, indices))
    model.set_input_embeddings(new_embedding)
    model.config.vocab_size = len(keep_ids)
    model.save_pretrained(args.output_dir)

    metadata = {
        "source_checkpoint": args.checkpoint,
        "source_vocab_size": len(tokenizer),
        "pruned_vocab_size": len(keep_ids),
        "min_vi_count": args.min_vi_count,
        "min_en_count": args.min_en_count,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embedding_parameters": new_embedding.weight.numel(),
        "old_token_ids": keep_ids,
    }
    Path(args.output_dir, "pruning.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items()
                      if key != "old_token_ids"}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("--tokenizer", required=True)
    scan_parser.add_argument("--data", required=True)
    scan_parser.add_argument("--output", required=True)
    scan_parser.add_argument("--batch_size", type=int, default=2048)
    scan_parser.add_argument("--english_start_shard", type=int, default=100)
    scan_parser.set_defaults(function=scan)
    prune_parser = commands.add_parser("prune")
    prune_parser.add_argument("--checkpoint", required=True)
    prune_parser.add_argument("--counts", required=True)
    prune_parser.add_argument("--output_dir", required=True)
    prune_parser.add_argument("--min_vi_count", type=int, default=2)
    prune_parser.add_argument("--min_en_count", type=int, default=2)
    prune_parser.set_defaults(function=prune)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
