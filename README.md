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
<br/>
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
Using pre-train transformer model mt5 provided by google 
<br/>
Fine-tuning the mt5-model
<br/>
with our dataset collect from [ETtoday](https://www.ettoday.net/?from=logo)
<br/>
### Hyperparameters
batch size: 4
<br/>
accumulate steps: 16
<br/>
epoch: 5
<br/>
learning rate: 5e-5
<br/>
optimizer: Adafactor with weight decay = 1e-2
<br/>
scheduler: Linear decay with warm up steps = 90
<br/>
## Evaluating the result
Using ROUGE which is the most used package designed for automatic summarization
<br/>
| Methods        | ROUGE-1           | ROUGE-2  | ROUGE-L |
| ------------- |:-------------:|:-------------:|:-------------:|
| sentence scoring method | 0.246016 | 0.105189 | 0.214126 |
| mt5 model without fine-tuning | 0.140646 | 0.064830 | 0.138213 |
| Supervised Learning | 0.389000 | 0.194737 | 0.343403 |
| Supervised Learning | 0.400387 | 0.198889 | 0.345834 |
<br/>
### Related Paper
1.[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
<br/>
2.[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
<br/>
3.[mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/pdf/2010.11934.pdf)
<br/>.
4.[A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION](https://arxiv.org/pdf/1705.04304.pdf)
## Related website
