# InferSent

**InferSent** ([code](https://github.com/facebookresearch/InferSent), [paper](https://arxiv.org/abs/1705.02364)) is a 
*sentence embeddings* method that provides semantic representations for English sentences.
 It is trained on natural language inference data and generalizes well to many different tasks. This module is heavily 
 based on [**the original implementation of InferSent**](https://github.com/facebookresearch/InferSent), but also 
 includes new API classes for exporting InferSent embeddings for input datasets/textual prompts.
 
This document is only to provide further information about interacting with the InferSent module. 
 **You do not need to take an further setup steps if you have successfully completed [this](https://github.com/natashamjaques/neural_chat#InferSent-Setup)**. 

## Dependencies

The dependencies of this python module include:

* [Pytorch](http://pytorch.org/)
* NLTK >= 3

## Download datasets
To get SNLI and MultiNLI [147MB], run:
```bash
cd dataset
./get_data.bash
```
This will download and preprocess SNLI/MultiNLI datasets. For MacOS, you may have to use *p7zip* instead of *unzip*.

## Use the pre-trained sentence encoder
We provide a simple interface to encode English sentences. **See [**encoder/demo.ipynb**](https://github.com/natashamjaques/neural_chat/tree/master/inferSent/encoder/demo.ipynb)
for a practical example.** Get started with the following steps:

*0) Make sure you have the NLTK tokenizer by running the following once:*
```python
import nltk
nltk.download('punkt')
```

*1) [Load the pre-trained model](https://github.com/natashamjaques/neural_chat/tree/master/inferSent/encoder/demo.ipynb) (in encoder/):*
```python
from models import InferSent
V = 1
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
```

*2) Set word vector path for the model:*
```python
W2V_PATH = '../dataset/GloVe/glove.840B.300d.txt'
infersent.set_w2v_path(W2V_PATH)
```

*3) Build the vocabulary of word vectors (i.e keep only those needed):*
```python
infersent.build_vocab(sentences, tokenize=True)
```
where *sentences* is your list of **n** sentences. You can update your vocabulary using *infersent.update_vocab(sentences)*, or directly load the **K** most common English words with *infersent.build_vocab_k_words(K=100000)*.
If **tokenize** is True (by default), sentences will be tokenized using NTLK.

*4) Encode your sentences (list of *n* sentences):*
```python
embeddings = infersent.encode(sentences, tokenize=True)
```
This outputs a numpy array with *n* vectors of dimension **4096**. Speed is around *1000 sentences per second* with batch size 128 on a single GPU.

*5) Visualize the importance that our model attributes to each word:*

We provide a function to visualize the importance of each word in the encoding of a sentence:
```python
infersent.visualize('A man plays an instrument.', tokenize=True)
```

## Reference

[1] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, [*Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*](https://arxiv.org/abs/1705.02364)

### Related work
* [J. R Kiros, Y. Zhu, R. Salakhutdinov, R. S. Zemel, A. Torralba, R. Urtasun, S. Fidler - SkipThought Vectors, NIPS 2015](https://arxiv.org/abs/1506.06726)
* [S. Arora, Y. Liang, T. Ma - A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017](https://openreview.net/pdf?id=SyK00v5xx)
* [Y. Adi, E. Kermany, Y. Belinkov, O. Lavi, Y. Goldberg - Fine-grained analysis of sentence embeddings using auxiliary prediction tasks, ICLR 2017](https://arxiv.org/abs/1608.04207)
* [A. Conneau, D. Kiela - SentEval: An Evaluation Toolkit for Universal Sentence Representations, LREC 2018](https://arxiv.org/abs/1803.05449)
* [S. Subramanian, A. Trischler, Y. Bengio, C. J Pal - Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning, ICLR 2018](https://arxiv.org/abs/1804.00079)
* [A. Nie, E. D. Bennett, N. D. Goodman - DisSent: Sentence Representation Learning from Explicit Discourse Relations, 2018](https://arxiv.org/abs/1710.04334)
* [D. Cer, Y. Yang, S. Kong, N. Hua, N. Limtiaco, R. St. John, N. Constant, M. Guajardo-Cespedes, S. Yuan, C. Tar, Y. Sung, B. Strope, R. Kurzweil - Universal Sentence Encoder, 2018](https://arxiv.org/abs/1803.11175)
* [A. Conneau, G. Kruszewski, G. Lample, L. Barrault, M. Baroni - What you can cram into a single vector: Probing sentence embeddings for linguistic properties, ACL 2018](https://arxiv.org/abs/1805.01070)
* [A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, S. Bowman - GLUE: A Multi-Task Benchmark and Analysis Platform
for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
