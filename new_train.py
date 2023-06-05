from argparse import ArgumentParser, Namespace
import numpy as np
import datasets 
import json
from math import isnan
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical
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
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AdamW,
    Adafactor,
    get_scheduler
)
import evaluate
from nltk.tokenize import RegexpTokenizer
import os
import eval
from collections import deque
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
#os.environ["WANDB_DISABLED"] = "true"
def compute_rl_loss(data, logits, model, tokenizer, accelerator, device, compute_metrice):
    # Using origin model to generate the summary
    baseline_tokens = accelerator.unwrap_model(model).generate(
        data["input_ids"],
        attention_mask=data["attention_mask"],
        max_length=64,
        num_beams=1
    )
    #print(logits.size())
    #print(logits)
    #print(softmax(logits, 1))
    """
    baseline_tokens = accelerator.pad_across_processes(
        baseline_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    baseline_tokens = accelerator.gather(baseline_tokens)
    baseline_preds = tokenizer.batch_decode(baseline_tokens.cpu().numpy(), skip_special_tokens=True)
    """
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
    print(scores)
    sampled_tokens = sampled_tokens.sequences
    """
    sampled_tokens = accelerator.pad_across_processes(
        sampled_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    sampled_tokens = accelerator.gather(sampled_tokens)
    sampled_preds = tokenizer.batch_decode(sampled_tokens.cpu().numpy(), skip_special_tokens=True)
    """
    # compute the score of baseline summary
    baseline_scores = []
    for pred, ref in zip(baseline_tokens, data['labels']):

        pred = torch.unsqueeze(pred, 0)
        ref = torch.unsqueeze(ref, 0)
        baseline_scores.append(compute_metrice([pred, ref]))
        #print(baseline_scores)
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
    #loss_fct = CrossEntropyLoss(reduction="none")
    #if logits.shape[1] < sampled_tokens.shape[1]:
    #    sampled_tokens = sampled_tokens[:, :logits.shape[1]]
    #loss_input = logits[:, :sampled_tokens.shape[1], :].reshape(-1, logits.shape[-1])
    #loss_target = sampled_tokens.reshape(-1)
    #sampled_probs = -loss_fct(loss_input, loss_target).reshape(logits.shape[0], -1).sum(1)
    diff_rewards = (torch.Tensor(baseline_rewards) - torch.Tensor(sampled_rewards)).to(device)
    print(diff_rewards)
    rl_loss = (diff_rewards * scores).mean()
    print(rl_loss)
    return rl_loss
"""
class RLSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        if isnan(outputs.loss.item()):
            return (outputs.loss, outputs) if return_outputs else outputs.loss
        loss = self.policy_gradient(outputs.logits, inputs['labels'])
        return (loss, outputs) if return_outputs else loss
    def policy_gradient(self, logits, labels, gamma=0.9):
        
        def getreward(prediction, label):
            try:
                baseline = {
                    'rouge1': 0.22,
                    'rouge2': 0.085,
                    'rougeL': 0.02
                }
                #print(prediction)
                rouge_scores = eval.eval_func([prediction, label])
                combined_rouge = 0
                for i in baseline:
                    combined_rouge += rouge_scores[i] / baseline[i]
                #print(prediction, label)
                print(combined_rouge)
                return combined_rouge
            except ValueError:
                return 0
        device = logits.device
        eps = 1e-6
        all_policys = []
        all_rewards = []
        for logit, label_ids in zip(logits, labels):
            generated_sequence = []
            for idx, single_logit in enumerate(logit):
                word_distribution = Categorical(logits=single_logit)
                next_word = word_distribution.sample()
                generated_sequence.append(next_word)
                all_policys.append(
                    word_distribution.log_prob(next_word).unsqueeze(0))
                if generated_sequence[-1] == self.tokenizer.eos_token_id or idx >= self.args.generation_max_length:
                    break
            generated_sequence = torch.tensor(generated_sequence).to()
            generated_sequence = self.tokenizer.decode(
                generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            label_ids = label_ids.detach().cpu().tolist()
            label_ids = [l for l in label_ids if l > 0]
            label_text = self.tokenizer.decode(
                label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            R = 0 if generated_sequence == "" else getreward(
                generated_sequence, label_text)
            rewards = deque()
            for _ in range(idx + 1):
                rewards.appendleft(R)
                R *= gamma
            all_rewards.extend(rewards)

        all_policys = torch.cat(all_policys)
        all_rewards = torch.tensor(all_rewards).to(device)
        all_rewards = (all_rewards - all_rewards.mean()) / \
            (all_rewards.std() + eps)  # standardize
        # \times -1 = maximization
        loss = torch.sum(-1 * all_policys * all_rewards, -1)
        return -loss
"""
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
            if step % 16 == 0 or step == len(train_dataloader):
                print("Train | Loss: {:.5f}".format(total_loss / step))

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
    unwrapped_model.save_pretrained("iter_trained_for_summarization_tw", save_function=accelerator.save)


def rl_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ prepare data """
    dataset = load_dataset("json", data_files={
        'train': "train.json",
        'eval':  "eval.json",
        'test': "test.json"
    })
    """ select tokenizer """
    mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    
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
        #preds, labels = preds.cpu(), labels.cpu()
        #print(preds)
        #print(labels)
        labels =np.where(labels != -100, labels, mt5_tokenizer.pad_token_id)
        text_preds = mt5_tokenizer.batch_decode(preds, skip_special_tokens=True)
        text_labels = mt5_tokenizer.batch_decode(labels, skip_special_tokens=True)
        #print(text_preds)
        #print(text_labels)
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
    model = (AutoModelForSeq2SeqLM
         .from_pretrained("google/mt5-small")
         #.from_pretrained("./iter_trained_for_summarization_tw")
         .to(device))
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
    no_decay = ["bias", "LayerNorm.weight"]
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
    for epoch in tqdm(range(5)):
        total_rl_loss, total_loss = 0, 0
        total_ml_loss = 0
        #print(len(train_dataloader))
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
            #print(ml_loss)
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
            if step % 16 == 0 or step == len(train_dataloader):
                print("Train | Loss: {:.5f}".format(total_loss / step))
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
    modelpath = "rl_iter_trained_for_summarization_tw_" + path
    unwrapped_model.save_pretrained("mlplusrl_iter_trained_for_summarization_tw", save_function=accelerator.save)
    return 
def main():
    rl_train()
    #train()
    return



if __name__ == "__main__":
    main()