import argparse
import glob
import json
import math
import os
import random
import shutil
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


CASES = ("KEEP", "CAPITAL", "UPPER")
PUNCTS = ("NONE", ".", ",", ":", "?")
LABELS = [f"{case}|{punct}" for case in CASES for punct in PUNCTS]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
SEP = "SEPL|||SEPR"
OP_SEP = "SEPL__SEPR"


def parse_line(line):
    words, labels = [], []
    for field in line.rstrip().split()[1:]:  # skip $START
        word, operations = field.split(SEP, 1)
        case, punct = "KEEP", "NONE"
        for operation in operations.split(OP_SEP):
            if operation == "$TRANSFORM_CASE_CAPITAL":
                case = "CAPITAL"
            elif operation == "$TRANSFORM_CASE_UPPER":
                case = "UPPER"
            elif operation.startswith("$REPLACE_"):
                target = operation[len("$REPLACE_"):]
                case = "UPPER" if target.isupper() else "CAPITAL"
            elif operation.startswith("$APPEND_"):
                candidate = operation[len("$APPEND_"):]
                if candidate in PUNCTS:
                    punct = candidate
        words.append(word)
        labels.append(LABEL2ID[f"{case}|{punct}"])
    return words, labels


def resolve_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.txt"))
    else:
        files = glob.glob(path)
    if not files:
        raise FileNotFoundError(f"No training data matched: {path}")
    return sorted(files)


def batches(path, batch_size, shuffle_files=False, seed=42):
    words_batch, labels_batch = [], []
    files = resolve_files(path)
    if shuffle_files:
        random.Random(seed).shuffle(files)
    for filename in files:
        with open(filename, encoding="utf-8") as source:
            for line in source:
                words, labels = parse_line(line)
                if not words:
                    continue
                words_batch.append(words)
                labels_batch.append(labels)
                if len(words_batch) == batch_size:
                    yield words_batch, labels_batch
                    words_batch, labels_batch = [], []
    if words_batch:
        yield words_batch, labels_batch


def encode(tokenizer, words_batch, labels_batch, max_length, device):
    encoded = tokenizer(words_batch, is_split_into_words=True, padding=True,
                        truncation=True, max_length=max_length,
                        return_tensors="pt")
    aligned = []
    for row, word_labels in enumerate(labels_batch):
        word_ids = encoded.word_ids(row)
        previous, row_labels = None, []
        for word_id in word_ids:
            if word_id is None or word_id == previous:
                row_labels.append(-100)
            else:
                row_labels.append(word_labels[word_id])
            previous = word_id
        aligned.append(row_labels)
    encoded["labels"] = torch.tensor(aligned)
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def evaluate(model, tokenizer, path, batch_size, max_length, device):
    model.eval()
    matrix = torch.zeros(len(LABELS), len(LABELS), dtype=torch.long)
    losses, batches_seen = 0.0, 0
    for words, labels in batches(path, batch_size):
        batch = encode(tokenizer, words, labels, max_length, device)
        output = model(**batch)
        predictions = output.logits.argmax(-1)
        mask = batch["labels"] != -100
        gold = batch["labels"][mask].cpu()
        pred = predictions[mask].cpu()
        matrix += torch.bincount(
            gold * len(LABELS) + pred,
            minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
        losses += output.loss.item()
        batches_seen += 1
    keep = LABEL2ID["KEEP|NONE"]
    edit_ids = [i for i in range(len(LABELS)) if i != keep]
    tp = matrix.diag()[edit_ids].sum().item()
    fp = matrix[:, edit_ids].sum().item() - tp
    fn = matrix[edit_ids, :].sum().item() - tp
    edit_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    f1s = []
    for idx in edit_ids:
        class_tp = matrix[idx, idx].item()
        class_fp = matrix[:, idx].sum().item() - class_tp
        class_fn = matrix[idx, :].sum().item() - class_tp
        f1s.append(2 * class_tp / max(1, 2 * class_tp + class_fp + class_fn))
    model.train()
    return {"eval_loss": losses / max(1, batches_seen),
            "edit_micro_f1": edit_f1,
            "edit_macro_f1": sum(f1s) / len(f1s)}


def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, step,
                    epoch, metrics, keep=3):
    path = os.path.join(output_dir, f"checkpoint-{step}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump({"step": step, "completed_epochs": epoch, **metrics}, f,
                  indent=2)
    torch.save({"step": step, "completed_epochs": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()},
               os.path.join(path, "training_state.pt"))
    checkpoints = sorted(
        (d for d in os.listdir(output_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.rsplit("-", 1)[1]))
    for old in checkpoints[:-keep]:
        shutil.rmtree(os.path.join(output_dir, old))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.tensorboard_dir,
                           flush_secs=args.tensorboard_flush_secs)
    writer.add_text("config", "\n".join(
        f"{key}: {value}" for key, value in vars(args).items()))
    device = torch.device("cuda")
    checkpoints = sorted(
        glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[1]))
    resume_path = checkpoints[-1] if args.resume and checkpoints else None
    model_source = resume_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForTokenClassification.from_pretrained(
        model_source, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID, ignore_mismatched_sizes=True).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = math.ceil(args.train_samples /
                                (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    step = micro_step = start_epoch = 0
    if resume_path:
        state = torch.load(os.path.join(resume_path, "training_state.pt"),
                           map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        start_epoch = state["completed_epochs"]
        print(f"Resumed {resume_path}: epoch={start_epoch} step={step}",
              flush=True)
    model.train(); optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        samples_seen = 0
        for words, labels in batches(args.train_file, args.batch_size,
                                     shuffle_files=True,
                                     seed=args.seed + epoch):
            if samples_seen >= args.train_samples:
                break
            samples_seen += len(words)
            batch = encode(tokenizer, words, labels, args.max_length, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.grad_accum
            loss.backward(); micro_step += 1
            if micro_step % args.grad_accum:
                continue
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_steps == 0:
                rate = step / (time.time() - started)
                train_loss = loss.item() * args.grad_accum
                print(f"step={step}/{total_steps} epoch={epoch+1} loss={train_loss:.4f} steps/s={rate:.3f}", flush=True)
                writer.add_scalar("train/loss", train_loss, step)
                writer.add_scalar("train/learning_rate",
                                  scheduler.get_last_lr()[0], step)
                writer.add_scalar("train/steps_per_second", rate, step)
            if step % args.eval_steps == 0:
                metrics = evaluate(model, tokenizer, args.dev_file,
                                   args.eval_batch_size, args.max_length, device)
                print(json.dumps({"step": step, **metrics}), flush=True)
                for name, value in metrics.items():
                    writer.add_scalar(f"eval/{name.removeprefix('eval_')}",
                                      value, step)
                writer.flush()
        # Checkpoint only once per epoch. Step-based evaluation is deliberately
        # kept separate so monitoring does not generate excessive disk I/O.
        metrics = evaluate(model, tokenizer, args.dev_file,
                           args.eval_batch_size, args.max_length, device)
        print(json.dumps({"epoch": epoch + 1, "step": step, **metrics}),
              flush=True)
        for name, value in metrics.items():
            writer.add_scalar(f"eval/{name.removeprefix('eval_')}",
                              value, step)
        save_checkpoint(model, tokenizer, optimizer, scheduler,
                        args.output_dir, step, epoch + 1, metrics)
        writer.flush()
        if step >= total_steps:
            break
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Fsoft-AIC/videberta-xsmall")
    p.add_argument("--train_file", default="data/capu_bilingual_15m/train")
    p.add_argument("--dev_file", default="data/capu_bilingual_15m/dev")
    p.add_argument("--output_dir", default="outputs/videberta-xsmall-capu-bilingual")
    p.add_argument("--train_samples", type=int, default=15_000_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=.05)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--tensorboard_dir",
                   default="outputs/videberta-xsmall-capu-bilingual/tensorboard")
    p.add_argument("--tensorboard_flush_secs", type=int, default=30)
    main(p.parse_args())
