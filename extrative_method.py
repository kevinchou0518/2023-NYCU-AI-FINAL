# -*- coding: utf-8 -*-
import nltk
import numpy as np
from ckiptagger import WS, POS, NER
import eval
import json
from tqdm import tqdm

ws = WS("./ckiptagger_data")
pos = POS("./ckiptagger_data")
ner = NER("./ckiptagger_data")

def sent_tokenizer(texts):
    start = 0
    i = 0
    sentences = []
    punt_list = ',.:?!;~，。：？！；～'

    for text in texts:
        if text in punt_list and token not in punt_list:
            sentences.append(texts[start : i+1])
            start = i + 1
            i += 1
        else:
            i += 1
            token = list(texts[start : i+2]).pop()
    if start < len(texts):
        sentences.append(texts[start:])
    return sentences

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='UTF-8').readlines()]
    return stopwords

def score_sentences(sentences, topn_words):
    scores = []
    sentence_idx = -1
    slist = []
    for s in sentences:
        temp = ws(s.split(" "))
        slist.append(temp[0])
    for s in slist:                
        sentence_idx += 1
        word_idx = []
        for w in topn_words:
            try:
                word_idx.append(s.index(w))
            except ValueError:
                pass
        word_idx.sort()
        if len(word_idx) == 0:
            continue

        clusters = []
        cluster = [word_idx[0]]
        i = 1
        while i < len(word_idx):
            CLUSTER_THRESHOLD = 4
            if word_idx[i] - word_idx[i - 1] < CLUSTER_THRESHOLD:
                cluster.append(word_idx[i])
            else:
                clusters.append(cluster[:])
                cluster = [word_idx[i]]
            i += 1
        clusters.append(cluster)

        max_cluster_score = 0
        for c in clusters:
            significant_words_in_cluster = len(c)
            total_words_in_cluster = c[-1] - c[0] + 1
            score = 1.0 * significant_words_in_cluster * significant_words_in_cluster / total_words_in_cluster
            if score > max_cluster_score:
                max_cluster_score = score
        scores.append((sentence_idx, max_cluster_score))
    return scores

def results(texts, topn_wordnum, n):
    stopwords = stopwordslist("stopwords.txt")
    sentence = sent_tokenizer(texts)
  
    words = []
    for sen in sentence:
        senlist = []
        senlist.append(sen)
        for w in ws(senlist,):
            for i in range(len(w)):
                if w[i] not in stopwords:
                    if len(w[i]) > 1 and w[i] != '\t':
                        words.append(w[i])
    wordfre = nltk.FreqDist(words)
    topn_words = [w[0] for w in sorted(wordfre.items(), key = lambda d: d[1], reverse=True)][:topn_wordnum]
    scored_sentences = score_sentences(sentence, topn_words)
    avg = np.mean([s[1] for s in scored_sentences])
    std = np.std([s[1] for s in scored_sentences])
    mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences if score > (avg + 0.5 * std)]

    top_n_scored = sorted(scored_sentences, key = lambda s: s[1])[-n:]
    top_n_scored = sorted(top_n_scored, key = lambda s: s[0])
    c = dict(mean_scoredsentences = [sentence[idx] for (idx, score) in mean_scored])
    c1 = dict(topnsentences = [sentence[idx] for (idx, score) in top_n_scored])
    return c, c1

def summarize(art_list):
    topic_list = []
    for art in art_list:
        topn_wordnum = int(len(art) / 10)
        n = 2
        c, c1 = results(art, topn_wordnum, n)
        topic = ""
        for i in range(len(c1["topnsentences"])):
            topic += c1['topnsentences'][i]
        topic_list.append(topic)
    return topic_list

if __name__ == '__main__':
    with open('test.json', encoding='utf-8') as f:
        data = list(f)
    
    new_topic_list = []
    old_topic_list = []
    print("Generate new topic")
    for index in tqdm(range(len(data))):
        x = data[index]
        y = json.loads(x)
        texts = y['article']
        topn_wordnum = int(len(texts) / 10)
        n = 2
        c, c1 = results(texts, topn_wordnum, n)
        
        new_topic = ""
        for i in range(len(c1["topnsentences"])):
            new_topic += c1['topnsentences'][i]
        new_topic_list.append(new_topic)
        old_topic = y['title']
        old_topic_list.append(old_topic)
   
    with open('extract.json', 'w', encoding='UTF-8') as extract:
        i = 0
        for x in data:
            y = json.loads(x)
            index = y['id']
            content = {"id" : index, "new_title" : new_topic_list[i], "old_title" : old_topic_list[i]}
            extract.write(json.dumps(content, ensure_ascii=False) + "\n")
            i += 1
     