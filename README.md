# 2023-NYCU-AI-FINAL
# Chinese Article Summarization
###  Download PyTorch version
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### Setup playground
```
pip install -r requirements.txt
```
## Target: Doing Chinese article summarization with two different methods
 1.Extractive Method
 2.Abstractive Method
## Baseline: Extractive Method - Sentence Scoring Method
Implementation:
<br/>
1.Tokenize the sentence
<br/>
2.Get the high frequency words
<br/>
3.Score the sentence
<br/>
4.Article summarize
## Main Approach: Abstractive Method - Fine-tuning mt5-model
Using pre-train model md5 and transformer provided by google on [Hugging Face](https://huggingface.co/)
### Hyperparamters

<br/>
Fine-tuning the model
<br/>
with supervis
## Evaluating the result
Using ROUGE which is the most used package designed for automatic summarization
<br/>
[chinese rouge](https://github.com/cccntu/tw_rouge/tree/main) 
### Useful resource
1.[CkipTagger](https://github.com/ckiplab/ckiptagger)
<br/>
2.[mt5-model](https://huggingface.co/google/mt5-small)
<br/>
3.[jeiba](https://github.com/fxsjy/jieba)
<br/>
4.[Text2vec](https://github.com/shibing624/text2vec)
<br/>
### Useful Article
1.[Text Summarization Techniques(一) — 概述](https://medium.com/ml-note/%E8%87%AA%E5%8B%95%E6%96%87%E7%AB%A0%E6%91%98%E8%A6%81%E6%96%B9%E6%B3%95-e56dc2d2f6f4)
<br/>
2.[Reinforcement Learning for NLP](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture16-guest.pdf)
<br/>
### Related Paper
1.[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
<br/>
2.[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
<br/>
3.[mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/pdf/2010.11934.pdf)
<br/>.
4.[A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION](https://arxiv.org/pdf/1705.04304.pdf)
