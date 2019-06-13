# Neural Chat
[PyTorch 0.4](https://github.com/pytorch/pytorch) Implementation of Neural Chat ([Approximating Interactive Human Evaluation with Self-Play for Open-Domain Dialog Systems](https://arxiv.org/abs/), [Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog](https://arxiv.org/abs/)). You can interact with the models here: http://neural.chat. 

This code is inspired by and built off of "A Hierarchical Latent Structure for Variational Conversation Modeling" ([code](https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling), [paper](https://arxiv.org/abs/1804.03424), [presentation](https://vimeo.com/277671819)). 

## Prerequisites
This section includes installation of required libraries, and downloading pre-trained models.

### Installation
Install Python packages
```
pip install -r requirements.txt
```

Setup python path to include repo
```
python setup.py develop
```

### InferSent Setup

> For more information about InferSent module, see [here](https://github.com/natashamjaques/neural_chat/tree/master/inferSent).

Download [GloVe](https://nlp.stanford.edu/projects/glove/) [2.18GB] (V1) or [fastText](https://fasttext.cc/docs/en/english-vectors.html) [5.83GB] (V2) vectors. We suggest using GloVe:
```bash
mkdir inferSent/dataset/GloVe
curl -Lo inferSent/dataset/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip inferSent/dataset/GloVe/glove.840B.300d.zip -d inferSent/dataset/GloVe/

mkdir inferSent/dataset/fastText
curl -Lo inferSent/dataset/fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip inferSent/dataset/fastText/crawl-300d-2M.vec.zip -d inferSent/dataset/fastText/
```

Download the pre-trained InferSent models (V1 trained with GloVe, V2 trained with fastText) [154MB each]:
```bash
curl -Lo inferSent/encoder/infersent1.pickle https://affect.media.mit.edu/neural_chat/inferSent/encoder/infersent1.pickle
curl -Lo inferSent/encoder/infersent2.pickle https://affect.media.mit.edu/neural_chat/inferSent/encoder/infersent2.pickle
```
Note that infersent1 is trained with GloVe (which have been trained on text preprocessed with the PTB tokenizer) and infersent2 is trained with fastText (which have been trained on text preprocessed with the MOSES tokenizer). The latter also removes the padding of zeros with max-pooling which was inconvenient when embedding sentences outside of their batches.

### TorchMoji Setup

> For more information about TorchMoji module, see [here](https://github.com/natashamjaques/neural_chat/tree/master/torchMoji).

Run the download script to downloads the pre-trained torchMoji weights [~85MB] from [here](https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=0) and put them in the `./torchMoji/model/` directory:
```
python torchMoji/scripts/download_weights.py
```

## Download & Preprocess data
The following scripts will:

1. Create directories `./datasets/reddit_casual/` and `./datasets/cornell/` respectively.

2. Download and preprocess conversation data inside each directory.

### for [Reddit Casual Conversations Dataset](https://www.reddit.com/r/CasualConversation/)

To download the pre-processed dataset, use:
```
python dataset_preprocess.py --dataset=reddit_casual --shortcut
```

Alternatively, if you'd like to download a smaller version, and do pre-processing steps on your end, use:
```
python dataset_preprocess.py --dataset=reddit_casual
    --max_sentence_length (maximum number of words in sentence; default: 30)
    --max_conversation_length (maximum turns of utterances in single conversation; default: 10)
    --max_vocab_size (maximum size of word vocabulary; default: 20000)
    --max_vocab_frequency (minimum frequency of word to be included in vocabulary; default: 5)
    --n_workers (number of workers for multiprocessing; default: os.cpu_count())
```

### for [Cornell Movie Dialogue Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

To download the pre-processed dataset, use:
```
python dataset_preprocess.py --dataset=cornell --shortcut
```

Alternatively, if you'd like to download a smaller version, and do pre-processing steps on your end, use:
```
python dataset_preprocess.py --dataset=reddit_casual
    --max_sentence_length (maximum number of words in sentence; default: 30)
    --max_conversation_length (maximum turns of utterances in single conversation; default: 10)
    --max_vocab_size (maximum size of word vocabulary; default: 20000)
    --max_vocab_frequency (minimum frequency of word to be included in vocabulary; default: 5)
    --n_workers (number of workers for multiprocessing; default: os.cpu_count())
```


## Training
Go to the model directory and set the save_dir in configs.py (this is where the model checkpoints will be saved)

We provide our implementation of VHCR, as well as our reference implementations for [HRED](https://arxiv.org/abs/1507.02221) and [VHRED](https://arxiv.org/abs/1605.06069).

To run training:
```
python train.py --data=<data> --model=<model> --batch_size=<batch_size>
```

For example:
1. Train HRED on Cornell Movie:
```
python train.py --data=cornell --model=HRED
```

2. Train VHRED with word drop of ratio 0.25 and kl annealing iterations 250000:
```
python train.py --data=ubuntu --model=VHRED --batch_size=40 --word_drop=0.25 --kl_annealing_iter=250000
```

3. Train VHCR with utterance drop of ratio 0.25:
```
python train.py --data=ubuntu --model=VHCR --batch_size=40 --sentence_drop=0.25 --kl_annealing_iter=250000
```

By default, it will save a model checkpoint every epoch to <save_dir> and a tensorboard summary.
For more arguments and options, see config.py.


## Evaluation
To evaluate the word perplexity:
```
python eval.py --model=<model> --checkpoint=<path_to_your_checkpoint>
```

For embedding based metrics, you need to download [Google News word vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), unzip it and put it under the datasets folder.
Then run:
```
python eval_embed.py --model=<model> --checkpoint=<path_to_your_checkpoint>
```


## Reference
If you use this code or the released Reddit dataset as part of any published research, please reference the following papers:

```
@article{ghandeharioun2019approximating,
  title={Approximating Interactive Human Evaluation with Self-Play for Open-Domain Dialog Systems},
  author={Ghandeharioun, Asma and Shen, Judy and Jaques, Natasha and Ferguson, Craig and Jones, Noah, and Lapedriza, Agata and Picard, Rosalind},
  journal={arXiv preprint arXiv:},
  year={2019}
}
```

```
@article{jaques2019way,
  title={Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog},
  author={Jaques, Natasha and Ghandeharioun, Asma and Shen, Judy and Ferguson, Craig and Jones, Noah, and Lapedriza, Agata and Gu, Shixiang and Picard, Rosalind},
  journal={arXiv preprint arXiv:},
  year={2019}
}
```


### Related work
* Park, Y., Cho, J., & Kim, G. (2018, June). [*A Hierarchical Latent Structure for Variational Conversation Modeling*](https://www.aclweb.org/anthology/N18-1162). In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) (pp. 1792-1801).
