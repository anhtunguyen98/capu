# Capitalization and Punctuation for Automatic Speech Recognition

Automatic Speech Recognition (ASR) systems typically generate text with no punctuation and capitalization of the words. This repository provides code for training and predicting punctuation and capitalization for each word in a sentence to make ASR output more readable and to boost the performance of the named entity recognition, machine translation, or text-to-speech models. The model for this task was trained using a pre-trained BERT model. For every word in our training dataset, we’re going to predict:

- punctuation mark that should follow the word and
- whether the word should be capitalized

Main idea was introduced in the following paper with the official PyTorch implementation:
> [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://arxiv.org/abs/2005.12592)<br>
> [Grammarly](https://github.com/grammarly/gector)

It is mainly based on `AllenNLP` and `transformers`.
## Installation
The following command installs all necessary packages:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7.

## Datasets
This model can work with any text dataset. The raw dataset should be preprocessed into two files, one `source` file and one `target` file. The `target` file should contain final texts, whereas the `source` file is simply the lowercase version with punctuations removed from the `target` file.<br>

**Note**: Punctuations should be space-separated with words.<br>

To train the model data has to be preprocessed and converted to special format with the command:
```.bash
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```

## Train model
To train the model, simply run:
```.bash
python train.py --train_set TRAIN_SET --dev_set DEV_SET \
                --model_dir MODEL_DIR
```

`transformer_model` accepts any Hugging Face model id (or a local directory)
that can be loaded by `AutoTokenizer.from_pretrained` and
`AutoModel.from_pretrained`. For example:

```bash
python train.py --train_set TRAIN_SET --dev_set DEV_SET \
    --model_dir MODEL_DIR \
    --transformer_model sentence-transformers/all-MiniLM-L6-v2 \
    --special_tokens_fix 0 --use_fast 1 --tune_bert 1
```

Training writes `training_config.json` into `MODEL_DIR`. Prediction accepts the
whole model directory and reads the backbone configuration automatically:

```bash
python predict.py --model_path MODEL_DIR \
    --vocab_path MODEL_DIR/vocabulary \
    --input_file INPUT_FILE --output_file OUTPUT_FILE
```

For older checkpoints without `training_config.json`, pass the model explicitly
with `--transformer_model MODEL_ID`.

### Prepare Vietnamese news data

The news dataset can be converted without loading all 19.4M rows into memory:

```bash
python utils/prepare_news_data.py \
    --dataset vietgpt/binhvq_news_vi \
    --output_dir data/news_capu \
    --max_train 1000000 --max_dev 10000 --max_test 10000 \
    --output_format tagged --min_free_gb 10
```

The default `tagged` format creates training-ready `train.txt`, `dev.txt`, and
`test.txt` directly and avoids duplicate source/target files. Use
`--output_format both` only when those intermediate pairs are needed. The
generator checks free space periodically and stops while keeping completed
output if the configured reserve would be crossed. Based on the included sample,
10 million tagged examples require roughly 15--16 GB; checkpoints and optimizer
state require additional space.

Documents, rather than individual sentences, are assigned to a split to reduce
data leakage. When `--output_format parallel` is used, convert the pairs to GEC
tags with:

```bash
python utils/preprocess_data.py \
    -s data/news_capu/train.source \
    -t data/news_capu/train.target \
    -o data/news_capu/train.txt
python utils/preprocess_data.py \
    -s data/news_capu/dev.source \
    -t data/news_capu/dev.target \
    -o data/news_capu/dev.txt
```

Train ViDeBERTa xsmall with `--special_tokens_fix 0`:

```bash
python train.py --train_set data/news_capu/train.txt \
    --dev_set data/news_capu/dev.txt \
    --model_dir outputs/videberta-xsmall-capu \
    --transformer_model Fsoft-AIC/videberta-xsmall \
    --special_tokens_fix 0 --use_fast 1 --tune_bert 1
```
There are a lot of parameters to specify among them:
- `cold_steps_count` the number of epochs where we train only last linear layer
- `transformer_model {bert, distilbert, gpt2, roberta, transformerxl, xlnet, albert, xlm-r, phobert, ...}` model encoder
- `tn_prob` probability of getting sentences with no errors; helps to balance precision/recall
- `pieces_per_token` maximum number of subwords per token; helps not to get CUDA out of memory

## Model inference
To run your model on the input file use the following command:
```.bash
python predict.py --model_path MODEL_PATH [MODEL_PATH ...] \
                  --vocab_path VOCAB_PATH --input_file INPUT_FILE \
                  --output_file OUTPUT_FILE
```
Among parameters:
- `min_error_probability` - minimum error probability (as in the paper)
- `additional_confidence` - confidence bias (as in the paper)
- `special_tokens_fix` to reproduce some reported results of pretrained models

## Citation
If you find this work is useful for your research, please cite our paper:
```
@inproceedings{omelianchuk-etal-2020-gector,
    title = "{GECT}o{R} {--} Grammatical Error Correction: Tag, Not Rewrite",
    author = "Omelianchuk, Kostiantyn  and
      Atrasevych, Vitaliy  and
      Chernodub, Artem  and
      Skurzhanskyi, Oleksandr",
    booktitle = "Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = jul,
    year = "2020",
    address = "Seattle, WA, USA â†’ Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bea-1.16",
    pages = "163--170",
    abstract = "In this paper, we present a simple and efficient GEC sequence tagger using a Transformer encoder. Our system is pre-trained on synthetic data and then fine-tuned in two stages: first on errorful corpora, and second on a combination of errorful and error-free parallel corpora. We design custom token-level transformations to map input tokens to target corrections. Our best single-model/ensemble GEC tagger achieves an F{\_}0.5 of 65.3/66.5 on CONLL-2014 (test) and F{\_}0.5 of 72.4/73.6 on BEA-2019 (test). Its inference speed is up to 10 times as fast as a Transformer-based seq2seq GEC system.",
}
```
