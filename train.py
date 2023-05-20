import os
import json
import tqdm
import numpy as np
import datasets 
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    default_data_collator,
    get_scheduler,
    set_seed,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#train_ds = load_dataset("json", data_files="train.json")
#test_ds = load_dataset("json", data_files="test.json")
#print(train_ds["train"][0])

dataset = {}
dataset_train = load_dataset("json", data_files="train.json")
dataset_test = load_dataset("json", data_files="test.json")
dataset_eval = load_dataset("json", data_files="eval.json")
dataset['train'] = dataset_train['train']
dataset['eval'] = dataset_eval['train']
dataset['test'] = dataset_test['train']
mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

def tokenize_sample_data(data):
  # Max token size is 14536 and 215 for inputs and labels, respectively.
  # Here I restrict these token size.
  input_feature = mt5_tokenizer(data["article"], truncation=True, max_length=256)
  label = mt5_tokenizer(data["title"], truncation=True, max_length=64)
  return {
    "input_ids": input_feature["input_ids"],
    "attention_mask": input_feature["attention_mask"],
    "labels": label["input_ids"],
  }

tokenized_ds = dataset.map(
  tokenize_sample_data,
  remove_columns=["id", "title", "url", "article"],
  batched=True,
  batch_size=128)

print(tokenized_ds)

# see https://huggingface.co/docs/transformers/main_classes/configuration
mt5_config = AutoConfig.from_pretrained(
  "google/mt5-small",
  max_length=128,
  length_penalty=0.6,
  no_repeat_ngram_size=2,
  num_beams=15,
)

model = (AutoModelForSeq2SeqLM
         .from_pretrained("google/mt5-small", config=mt5_config)
         .to(device))

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
  mt5_tokenizer,
  model=model,
  return_tensors="pt")

import evaluate
import numpy as np
from nltk.tokenize import RegexpTokenizer

rouge_metric = evaluate.load("rouge")

# define function for custom tokenization
def tokenize_sentence(arg):
  encoded_arg = mt5_tokenizer(arg)
  return mt5_tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

# define function to get ROUGE scores with custom tokenization
def metrics_func(eval_arg):
  preds, labels = eval_arg
  # Replace -100
  labels = np.where(labels != -100, labels, mt5_tokenizer.pad_token_id)
  # Convert id tokens to text
  text_preds = mt5_tokenizer.batch_decode(preds, skip_special_tokens=True)
  text_labels = mt5_tokenizer.batch_decode(labels, skip_special_tokens=True)
  # Insert a line break (\n) in each sentence for ROUGE scoring
  # (Note : Please change this code, when you perform on other languages except for Japanese)
  text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
  text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
  sent_tokenizer_jp = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
  text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
  text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]
  # compute ROUGE score with custom tokenization
  return rouge_metric.compute(
    predictions=text_preds,
    references=text_labels,
    tokenizer=tokenize_sentence
  )

from torch.utils.data import DataLoader

sample_dataloader = DataLoader(
  tokenized_ds["test"].with_format("torch"),
  collate_fn=data_collator,
  batch_size=5)
for batch in sample_dataloader:
  with torch.no_grad():
    preds = model.generate(
      batch["input_ids"].to(device),
      num_beams=15,
      num_return_sequences=1,
      no_repeat_ngram_size=1,
      remove_invalid_values=True,
      max_length=128,
    )
  labels = batch["labels"]
  break

print(metrics_func([preds, labels]))


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
  output_dir = "mt5-summarize-ja",
  log_level = "error",
  num_train_epochs = 10,
  learning_rate = 5e-4,
  lr_scheduler_type = "linear",
  warmup_steps = 90,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size = 2,
  per_device_eval_batch_size = 1,
  gradient_accumulation_steps = 16,
  evaluation_strategy = "steps",
  eval_steps = 100,
  predict_with_generate=True,
  generation_max_length = 128,
  save_steps = 500,
  logging_steps = 10,
  push_to_hub = False
)


from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
  model = model,
  args = training_args,
  data_collator = data_collator,
  compute_metrics = metrics_func,
  train_dataset = tokenized_ds["train"],
  eval_dataset = tokenized_ds["validation"].select(range(20)),
  tokenizer = mt5_tokenizer,
)

trainer.train()