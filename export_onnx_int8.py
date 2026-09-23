"""Export a Hugging Face token-classification model to ONNX and dynamic INT8."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForTokenClassification, AutoTokenizer


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask).logits


def main(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model).cpu().eval()
    tokenizer.save_pretrained(output)

    encoded = tokenizer(
        ["đây là một câu tiếng việt", "this is an english sentence"],
        padding=True, return_tensors="pt")
    wrapper = LogitsWrapper(model).eval()
    fp32_path = output / "model.fp32.onnx"
    torch.onnx.export(
        wrapper,
        (encoded["input_ids"], encoded["attention_mask"]),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(fp32_path))

    int8_path = output / "model.int8.onnx"
    quantize_dynamic(
        str(fp32_path), str(int8_path),
        # Per-channel quantization severely distorts DeBERTa's pruned word
        # embedding. Per-tensor keeps evaluation quality while retaining the
        # full dynamic-INT8 size reduction.
        per_channel=False, reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    checker_error = None
    try:
        onnx.checker.check_model(str(int8_path))
    except onnx.checker.ValidationError as error:
        # ORT dynamic quantization of DeBERTa graphs with an `If` subgraph can
        # leave cross-subgraph Identity inputs that the standalone ONNX checker
        # rejects, while ORT itself loads and executes the model correctly.
        checker_error = str(error)

    inputs = {key: value.numpy().astype(np.int64)
              for key, value in encoded.items()
              if key in ("input_ids", "attention_mask")}
    fp32_session = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"])
    fp32_logits = fp32_session.run(["logits"], inputs)[0]
    int8_logits = int8_session.run(["logits"], inputs)[0]
    metadata = {
        "source_model": args.model,
        "opset": args.opset,
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "compression_ratio": fp32_path.stat().st_size / int8_path.stat().st_size,
        "onnx_checker_valid": checker_error is None,
        "onnx_checker_error": checker_error,
        "smoke_max_logit_abs_delta": float(
            np.max(np.abs(fp32_logits - int8_logits))),
        "smoke_prediction_differences": int(np.count_nonzero(
            fp32_logits.argmax(-1) != int8_logits.argmax(-1))),
    }
    (output / "export.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--opset", type=int, default=17)
    main(parser.parse_args())
