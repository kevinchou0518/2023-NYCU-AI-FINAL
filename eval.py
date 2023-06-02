import numpy as np
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
import evaluate

mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

rouge_metric = evaluate.load("rouge")


def tokenize_sentence(arg):
  encoded_arg = mt5_tokenizer(arg)
  return mt5_tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

def eval_func(eval_arg):
    text_preds, text_labels = eval_arg
    text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
    sent_tokenizer_tw = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_tw.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer_tw.tokenize(l))) for l in text_labels]
    return rouge_metric.compute(
        predictions=text_preds,
        references=text_labels,
        tokenizer=tokenize_sentence
    )