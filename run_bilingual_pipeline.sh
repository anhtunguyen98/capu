#!/usr/bin/env bash
set -euo pipefail

cd /workspace/vicapu
prep_pid="${1:?usage: run_bilingual_pipeline.sh PREP_PID}"
while kill -0 "$prep_pid" 2>/dev/null; do
  sleep 30
done

test -f data/capu_bilingual_15m/manifest.json
python -u -m utils.audit_bilingual_data \
  --data_dir data/capu_bilingual_15m \
  2>&1 | tee logs/audit_bilingual_15m.log

mkdir -p outputs/videberta-xsmall-capu-bilingual/tensorboard
if ss -ltn | grep -q ':8007 '; then
  echo 'Port 8007 is already occupied' >&2
  exit 1
fi
tensorboard \
  --logdir outputs/videberta-xsmall-capu-bilingual/tensorboard \
  --host 0.0.0.0 --port 8007 \
  > logs/tensorboard-8007.log 2>&1 &
tensorboard_pid=$!
trap 'kill "$tensorboard_pid" 2>/dev/null || true' EXIT
sleep 3
kill -0 "$tensorboard_pid"

python -u train_hf.py \
  --model Fsoft-AIC/videberta-xsmall \
  --train_file data/capu_bilingual_15m/train \
  --dev_file data/capu_bilingual_15m/dev \
  --output_dir outputs/videberta-xsmall-capu-bilingual \
  --tensorboard_dir outputs/videberta-xsmall-capu-bilingual/tensorboard \
  --train_samples 15000000 \
  --epochs 3 \
  --learning_rate 5e-5 \
  --max_length 128 \
  --batch_size 64 \
  --eval_batch_size 128 \
  --grad_accum 4 \
  2>&1 | tee logs/train_bilingual_15m.log
