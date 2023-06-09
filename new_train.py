
import numpy as np
import datasets 
import json
from math import isnan
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import evaluate
from nltk.tokenize import RegexpTokenizer
import os
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoTokenizer,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Adafactor,
    get_scheduler
)
import evaluate
from nltk.tokenize import RegexpTokenizer
import os
import eval



def compute_rl_loss(data, logits, model, tokenizer, accelerator, device, compute_metrice):
    # Using origin model to generate the summary
    baseline_tokens = accelerator.unwrap_model(model).generate(
        data["input_ids"],
        attention_mask=data["attention_mask"],
        max_length=64,
        num_beams=1
    )

    # Using model to generae the summary with sampling method 
    sampled_tokens = accelerator.unwrap_model(model).generate(
        data["input_ids"],
        attention_mask=data["attention_mask"],
        max_length=64,
        do_sample=True,
        num_beams=1,
        top_k=0,
        top_p=0.2,
        temperature=0.5,
        output_scores=True,
        return_dict_in_generate=True,
    )
    transition_scores = model.compute_transition_scores(
        sampled_tokens.sequences, sampled_tokens.scores, normalize_logits=False
    )
    scores = []
    for i in transition_scores:
        sum = 0.0
        for j in i:
            if j > -10000000:
                sum += j
        scores.append(sum/10)
    scores = torch.tensor(scores).to(device)
    sampled_tokens = sampled_tokens.sequences
    # compute the score of baseline summary
    baseline_scores = []
    for pred, ref in zip(baseline_tokens, data['labels']):
        pred = torch.unsqueeze(pred, 0)
        ref = torch.unsqueeze(ref, 0)
        baseline_scores.append(compute_metrice([pred, ref]))
    baseline_rewards = torch.FloatTensor([(scores["rouge1"] + scores["rouge2"] + scores["rougeL"]) / 3 \
                        for scores in baseline_scores]).to(device)
    # compute the score of sample summary
    sampled_scores = []
    for pred, ref in zip(sampled_tokens, data['labels']):
        pred = torch.unsqueeze(pred, 0)
        ref = torch.unsqueeze(ref, 0)
        sampled_scores.append(compute_metrice([pred, ref]))
    sampled_rewards = torch.FloatTensor([(scores["rouge1"] + scores["rouge2"] + scores["rougeL"]) / 3 \
                        for scores in sampled_scores]).to(device)
    # compute the loss
    diff_rewards = (torch.Tensor(baseline_rewards) - torch.Tensor(sampled_rewards)).to(device)
    rl_loss = -(diff_rewards * scores).mean()
    return rl_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ prepare data """
    dataset = load_dataset("json", data_files={
        'train': "train.json",
        'eval':  "eval.json",
        'test': "test.json"
    })
    """ select tokenizer """
    mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    
    """ tokenize data """
    def tokenize_sample_data(data, indices):
        # restrict token size.
        input_feature = mt5_tokenizer(data["article"], truncation=True, max_length=256)
        label = mt5_tokenizer(data["title"], truncation=True, max_length=64)
        return {
            "input_ids": input_feature["input_ids"],
            "attention_mask": input_feature["attention_mask"],
            "labels": label["input_ids"],
            "indices": indices
        }
    
    tokenized_ds = dataset.map(
        tokenize_sample_data,
        remove_columns=["id", "title", "url", "article"],
        with_indices=True,
        batched=True,
        batch_size=128,
    )
    
    # define compute metrice function 
    rouge_metric = evaluate.load("rouge")

    def tokenize_sentence(arg):
        encoded_arg = mt5_tokenizer(arg)
        return mt5_tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)
    
    def compute_metrices(eval_arg, aggre=True):
        preds, labels = eval_arg
        preds, labels = preds.cpu(), labels.cpu()
        labels = np.where(labels.cpu().numpy() != -100, labels, mt5_tokenizer.pad_token_id)
        text_preds = mt5_tokenizer.batch_decode(preds, skip_special_tokens=True)
        text_labels = mt5_tokenizer.batch_decode(labels, skip_special_tokens=True)
        text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
        text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
        sent_tokenizer_tw = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
        text_preds = ["\n".join(np.char.strip(sent_tokenizer_tw.tokenize(p))) for p in text_preds]
        text_labels = ["\n".join(np.char.strip(sent_tokenizer_tw.tokenize(l))) for l in text_labels]
        return rouge_metric.compute(
            predictions=text_preds,
            references=text_labels,
            tokenizer=tokenize_sentence,
            use_aggregator=aggre
        )
    
    """ load model and data collator """
    mt5_config = AutoConfig.from_pretrained(
        "google/mt5-small",
        max_length=128,
        length_penalty=0.6,
        no_repeat_ngram_size=2,
        num_beams=10,  
    )
    
    model = (AutoModelForSeq2SeqLM
        .from_pretrained("google/mt5-small", config=mt5_config)
        .to(device)
    )
    data_collator = DataCollatorForSeq2Seq(
        mt5_tokenizer,
        model=model,
        return_tensors="pt"
    )

    """ use acclerator to enhance training process """
    accelerator = Accelerator()
    traindataset = tokenized_ds["train"]
    evaldataset = tokenized_ds["eval"]
    train_dataloader = DataLoader(
        traindataset, shuffle=True, collate_fn=data_collator, batch_size=4
    )
    eval_dataloader = DataLoader(
        evaldataset.select(range(20)), shuffle=True, collate_fn=data_collator, batch_size=4
    )

    optimizer = Adafactor(model.parameters() , lr = 5e-5 , weight_decay = 0.01 , relative_step = False , scale_parameter = False)
    
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    
    )
    
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=90,
        num_training_steps=2360,
    )

    train_info_loss = []
    train_info_eval = []
    path = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(path)

    """ training """
    max_rouge_score = 0.95
    for epoch in tqdm(range(5)):
        total_ml_loss, total_loss = 0, 0
        accum_loss = 0

        for step, data in tqdm(enumerate(train_dataloader, 1)):
            model.train()
            data = {k: v for k, v in data.items() if k != "indices"}
            outputs = model(**data)
            ml_loss = outputs.loss
            total_ml_loss += ml_loss.item()
            
            loss = ml_loss
            total_loss += loss.item()
            if len(train_dataloader) % 16 != 0 \
                    and len(train_dataloader) - step < 16:
                loss = loss / (len(train_dataloader) % 16)
            else:
                loss = loss / 16
            accum_loss += loss
            accelerator.backward(loss)
            if step % 16 == 0 or step == len(train_dataloader):
                accum_loss /= 16
                print("Loss: {:.5f}".format(accum_loss))
                info = {
                    'epoch':epoch,
                    'step': step,
                    'loss': accum_loss.detach().cpu().numpy().tolist()
                }
                train_info_loss.append(info)
                clip_grad_norm_(model.parameters(), max_norm=5) 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                model.eval()
                eval_metrices = {
                    'rouge1': 0,
                    'rouge2': 0,
                    'rougeL': 0,
                    'rougeLsum': 0
                }
                num = 0
                for batch in eval_dataloader:
                    num = num + 1
                    with torch.no_grad():
                        eval_preds = accelerator.unwrap_model(model).generate(
                            batch["input_ids"].to(device),
                            num_beams=15,
                            num_return_sequences=1,
                            no_repeat_ngram_size=1,
                            remove_invalid_values=True,
                            max_length=64,
                        )
                    eval_labels = batch["labels"]
                    temp_metrices = compute_metrices([eval_preds,eval_labels])
                    eval_metrices['rouge1'] += temp_metrices['rouge1']
                    eval_metrices['rouge2'] += temp_metrices['rouge2']
                    eval_metrices['rougeL'] += temp_metrices['rougeL']
                    eval_metrices['rougeLsum'] += temp_metrices['rougeLsum']
                    if num > 5:
                        break
                eval_metrices['rouge1'] /= num
                eval_metrices['rouge2'] /= num
                eval_metrices['rougeL'] /= num
                eval_metrices['rougeLsum'] /= num
                print(" ---evaluating--- ")
                print("step =",step)
                print(eval_metrices)
                temp_score = eval_metrices['rouge1'] + eval_metrices['rouge2'] + eval_metrices['rougeL']
                if temp_score > max_rouge_score:
                        max_rouge_score = temp_score
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained("temp_ml_model", save_function=accelerator.save)
                        print("save step:", step)
                        print("max_rouge_score", max_rouge_score)
                info = {
                    'epoch':epoch,
                    'step': step,
                    'rouge_score': eval_metrices
                }
                train_info_eval.append(info)
        filename = path + '/loss_train_epoch_' + str(epoch) + '.json'
        with open(filename, 'w+') as file:
            for data in train_info_loss:
                file.write(json.dumps(data , ensure_ascii=False) + "\n")
        filename = path + '/eval_train_epoch_' + str(epoch) + '.json'
        with open(filename, 'w+') as file:
            for data in train_info_eval:
                file.write(json.dumps(data , ensure_ascii=False) + "\n")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("final_ml_model", save_function=accelerator.save)


def rl_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ prepare data """
    dataset = load_dataset("json", data_files={
        'train': "train.json",
        'eval':  "eval.json",
        'test': "test.json"
    })
    """ select tokenizer """
    mt5_config = AutoConfig.from_pretrained(
        "google/mt5-small",
        max_length=128,
        length_penalty=0.6,
        no_repeat_ngram_size=2,
        num_beams=10,  
    )
    mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    model = (AutoModelForSeq2SeqLM
        .from_pretrained("google/mt5-small", config=mt5_config)
        .to(device)
    )
    """ tokenizer data """
    def tokenize_sample_data(data, indices):
        # restrict token size.
        input_feature = mt5_tokenizer(data["article"], truncation=True, max_length=256)
        label = mt5_tokenizer(data["title"], truncation=True, max_length=64)
        return {
            "input_ids": input_feature["input_ids"],
            "attention_mask": input_feature["attention_mask"],
            "labels": label["input_ids"],
            "indices": indices
        }
    
    tokenized_ds = dataset.map(
        tokenize_sample_data,
        remove_columns=["id", "title", "url", "article"],
        with_indices=True,
        batched=True,
        batch_size=128
    )
    
    # define compute metrice function 
    rouge_metric = evaluate.load("rouge")

    def tokenize_sentence(arg):
        encoded_arg = mt5_tokenizer(arg)
        return mt5_tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)
    
    def compute_metrices(eval_arg, aggre=True, preds_cpu=False, labels_cpu=False):
        preds, labels = eval_arg
        if(preds_cpu==False):
            preds = preds.cpu()
        if(labels_cpu==False):
            labels = labels.cpu()
        labels =np.where(labels != -100, labels, mt5_tokenizer.pad_token_id)
        text_preds = mt5_tokenizer.batch_decode(preds, skip_special_tokens=True)
        text_labels = mt5_tokenizer.batch_decode(labels, skip_special_tokens=True)
        text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
        text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
        sent_tokenizer_tw = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
        text_preds = ["\n".join(np.char.strip(sent_tokenizer_tw.tokenize(p))) for p in text_preds]
        text_labels = ["\n".join(np.char.strip(sent_tokenizer_tw.tokenize(l))) for l in text_labels]
        return rouge_metric.compute(
            predictions=text_preds,
            references=text_labels,
            tokenizer=tokenize_sentence,
            use_aggregator=aggre
        )

    data_collator = DataCollatorForSeq2Seq(
        mt5_tokenizer,
        model=model,
        return_tensors="pt"
    )
    accelerator = Accelerator()
    traindataset = tokenized_ds["train"]
    evaldataset = tokenized_ds["eval"]
    train_dataloader = DataLoader(
        traindataset, shuffle=True, collate_fn=data_collator, batch_size=4
    )
    eval_dataloader = DataLoader(
        evaldataset, shuffle=True, collate_fn=data_collator, batch_size=4
    )

    optimizer = Adafactor(model.parameters() , lr = 5e-5 , weight_decay = 0.01 , relative_step = False , scale_parameter = False)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    
    )
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=90,
        num_training_steps=2360,
    )
    train_info_loss = []
    train_info_eval = []
    path = datetime.now().strftime("%Y%m%d-%H%M%S")
    path += '_rl'
    os.makedirs(path)
    max_rouge_score = 0
    for epoch in tqdm(range(5)):
        total_rl_loss, total_loss = 0, 0
        total_ml_loss = 0
        accum_loss = 0.0
        for step, data in tqdm(enumerate(train_dataloader, 1)):
            model.train()
            data = {k: v for k, v in data.items() if k != "indices"}
            outputs = model(**data)
            rl_ratio = 0.1*epoch
            #rl_ratio = 1
            #print('rlr',rl_ratio)

            if rl_ratio > 0:
                rl_loss = compute_rl_loss(data, outputs.logits, model, mt5_tokenizer, accelerator, device, compute_metrices)
            else:
                rl_loss = torch.tensor(0)
            ml_loss = outputs.loss
            total_rl_loss += rl_loss.item()
            total_ml_loss += ml_loss.item()
            loss = rl_loss * rl_ratio + ml_loss * (1 - rl_ratio)
            total_loss += loss.item()
            if len(train_dataloader) % 16 != 0 \
                    and len(train_dataloader) - step < 16:
                loss = loss / (len(train_dataloader) % 16)
            else:
                loss = loss / 16
            accum_loss += loss
            accelerator.backward(loss)
            if step % 16 == 0 or step == len(train_dataloader):
                accum_loss /= 16
                print("Loss: {:.5f}".format(accum_loss))
                info = {
                    'epoch':epoch,
                    'step': step,
                    'loss': accum_loss.detach().cpu().numpy().tolist()
                }
                train_info_loss.append(info)
                accum_loss = 0
                clip_grad_norm_(model.parameters(), max_norm=5) 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                model.eval()
                eval_metrices = {
                    'rouge1': 0,
                    'rouge2': 0,
                    'rougeL': 0,
                    'rougeLsum': 0
                }
                num = 0
                for batch in eval_dataloader:
                    num = num + 1
                    with torch.no_grad():
                        eval_preds = accelerator.unwrap_model(model).generate(
                            batch["input_ids"].to(device),
                            num_beams=15,
                            num_return_sequences=1,
                            no_repeat_ngram_size=1,
                            remove_invalid_values=True,
                            max_length=64,
                        )
                    eval_labels = batch["labels"]
                    temp_metrices = compute_metrices([eval_preds,eval_labels])
                    eval_metrices['rouge1'] += temp_metrices['rouge1']
                    eval_metrices['rouge2'] += temp_metrices['rouge2']
                    eval_metrices['rougeL'] += temp_metrices['rougeL']
                    eval_metrices['rougeLsum'] += temp_metrices['rougeLsum']
                    if num > 5:
                        break
                eval_metrices['rouge1'] /= num
                eval_metrices['rouge2'] /= num
                eval_metrices['rougeL'] /= num
                eval_metrices['rougeLsum'] /= num
                temp_score = eval_metrices['rouge1'] + eval_metrices['rouge2'] + eval_metrices['rougeL']
                if temp_score > max_rouge_score and epoch == 4:
                        max_rouge_score = temp_score
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained("temp_mlrl_model", save_function=accelerator.save)
                        print("save step:", step)
                        print("max_rouge_score", max_rouge_score)
                info = {
                    'epoch':epoch,
                    'step': step,
                    'rouge_score': eval_metrices
                }
                train_info_eval.append(info)
                
                print(" ---evaluating--- ")
                print("step =",step)
                print(eval_metrices)
        filename = path + '/loss_train_epoch_' + str(epoch) + '.json'
        with open(filename, 'w+') as file:
            for data in train_info_loss:
                file.write(json.dumps(data , ensure_ascii=False) + "\n")
        filename = path + '/eval_train_epoch_' + str(epoch) + '.json'
        with open(filename, 'w+') as file:
            for data in train_info_eval:
                file.write(json.dumps(data , ensure_ascii=False) + "\n")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("final_mlrl_model", save_function=accelerator.save)
    return 

def main():
    #rl_train()
    train()
    return



if __name__ == "__main__":
    main()