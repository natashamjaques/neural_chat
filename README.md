# Neural Chat
[PyTorch 0.4](https://github.com/pytorch/pytorch) Implementation of Neural Chat: [Approximating Interactive Human Evaluation with Self-Play for Open-Domain Dialog Systems](https://arxiv.org/abs/1906.09308). You can interact with the models here: **http://neural.chat**.

This repository is accompanied by [Neural Chat Web](https://github.com/asmadotgh/neural_chat_web) to deploy a web server and host the models online.

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

### PyTorch Setup
Follow the instructions [here](https://pytorch.org/get-started/locally/) to download PyTorch version (0.4.0) or by running
```bash
pip3 install torch===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### InferSent Setup

> For more information about InferSent module, see [here](https://github.com/natashamjaques/neural_chat/tree/master/inferSent).

Download [GloVe](https://nlp.stanford.edu/projects/glove/) [2.18GB] (V1) and the pre-trained InferSent models trained with GloVe.

```bash
mkdir inferSent/dataset/GloVe
curl -Lo inferSent/dataset/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip inferSent/dataset/GloVe/glove.840B.300d.zip -d inferSent/dataset/GloVe/
curl -Lo inferSent/encoder/infersent1.pickle https://affect.media.mit.edu/neural_chat/inferSent/encoder/infersent1.pickle
```

You can instead download [fastText](https://fasttext.cc/docs/en/english-vectors.html) [5.83GB] (V2) vectors and the corresponding InferSent model. We suggest using GloVe:
```bach
mkdir inferSent/dataset/fastText
curl -Lo inferSent/dataset/fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip inferSent/dataset/fastText/crawl-300d-2M.vec.zip -d inferSent/dataset/fastText/
curl -Lo inferSent/encoder/infersent2.pickle https://affect.media.mit.edu/neural_chat/inferSent/encoder/infersent2.pickle
```

Note that infersent1 is trained with GloVe (which have been trained on text preprocessed with the PTB tokenizer) and infersent2 is trained with fastText (which have been trained on text preprocessed with the MOSES tokenizer). The latter also removes the padding of zeros with max-pooling which was inconvenient when embedding sentences outside of their batches.

### TorchMoji Setup

> For more information about TorchMoji module, see [here](https://github.com/natashamjaques/neural_chat/tree/master/torchMoji).

Run the download script to downloads the pre-trained torchMoji weights [~85MB] from [here](https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=0) and put them in the `./torchMoji/model/` directory:
```
python torchMoji/scripts/download_weights.py
```

## Download & Preprocess Data
The following scripts will:

1. Create directories `./datasets/reddit_casual/` and `./datasets/cornell/` respectively.

2. Download and preprocess conversation data inside each directory.

### for [Reddit Casual Conversations Dataset](https://www.reddit.com/r/CasualConversation/)

To download the pre-processed dataset [10.31GB], use:
```
python dataset_preprocess.py --dataset=reddit_casual --shortcut
```

Alternatively, if you'd like to download a smaller version [24.2MB], and do pre-processing steps on your end, use:
```
python dataset_preprocess.py --dataset=reddit_casual
    --max_sentence_length (maximum number of words in sentence; default: 30)
    --max_conversation_length (maximum turns of utterances in single conversation; default: 10)
    --max_vocab_size (maximum size of word vocabulary; default: 20000)
    --max_vocab_frequency (minimum frequency of word to be included in vocabulary; default: 5)
    --n_workers (number of workers for multiprocessing; default: os.cpu_count())
```

### for [Cornell Movie Dialogue Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

To download the pre-processed dataset [3.61GB], use:
```
python dataset_preprocess.py --dataset=cornell --shortcut
```

Alternatively, if you'd like to download a smaller version [9.9MB], and do pre-processing steps on your end, use:
```
python dataset_preprocess.py --dataset=reddit_casual
    --max_sentence_length (maximum number of words in sentence; default: 30)
    --max_conversation_length (maximum turns of utterances in single conversation; default: 10)
    --max_vocab_size (maximum size of word vocabulary; default: 20000)
    --max_vocab_frequency (minimum frequency of word to be included in vocabulary; default: 5)
    --n_workers (number of workers for multiprocessing; default: os.cpu_count())
```


## Training EI (Emotion+Infersent) Models
Go to the model directory and set the save_dir in configs.py (this is where the model checkpoints will be saved).

By default, it will save a model checkpoint every epoch to <save_dir> and a tensorboard summary.
For more arguments and options, see config.py.

Note that after training, you should only keep the single optimal checkpoint in the checkpoint directory for
[evaluation](#evaluation) and [interaction](#interacting-with-models) steps and remove the
remaining checkpoints.

We provide our implementation of EI (Emotion+Infersent) models built upon implementations for [VHCR](https://arxiv.org/pdf/1804.03424.pdf), [VHRED](https://arxiv.org/abs/1605.06069), and [HRED](https://arxiv.org/abs/1507.02221).

To run training:
```
python train.py --data=<data> --model=<model> --batch_size=<batch_size> [--emotion --infersent]
```

For example:
1. Train HRED-Infersent-only on Cornell Movie:
```
python model/train.py --data=cornell --model=HRED --infersent --infersent_weight=25000 --infersent_embedding_size=128
```

2. Train VHRED-Emotion-only on Reddit Casual Conversations:
```
python model/train.py --data=reddit_casual --model=VHRED --emotion --emo_weight=25 --emo_embedding_size=128
```

3. Train VHCR-EI on Reddit Casual Conversations:
```
python model/train.py --data=reddit_casual --model=VHCR --emotion --infersent --emo_weight=25 --emo_embedding_size=128 --infersent_weight=100000 --infersent_embedding_size=4000
```

## Evaluation

To evaluate the word perplexity:
```
python model/eval.py --mode=<mode> --checkpoint=<path_to_your_checkpoint_directory>
```

For embedding based metrics, you need to download [Google News word vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), unzip it and put it under the datasets folder.
Then run:
```
python model/eval_embed.py --mode=<mode> --checkpoint=<path_to_your_checkpoint_directory>
```

To evaluate sentiment and semantics using distance from torhcmoji and infersent inferred embedding:
```
python model/eval_novel.py --mode=<mode> --checkpoint=<path_to_your_checkpoint_directory>
```

## Interacting with Models

Use the following command to interact with / talk to a saved model checkpoint:
```
python model/interact.py --debug --checkpoint=<path_to_your_checkpoint_directory>
```

## Deploying chatbots live on a web server
This code is accompanied by another repository that implements the server portion of Neural Chat project.
Please refer to [**Neural Chat Web**](https://github.com/asmadotgh/neural_chat_web) for details on how to deploy your chatbots live on web.


## Reference
If you use this code or the released Reddit dataset, please reference one of the following papers:

For batch reinforcement learning in dialog systems, refer to:
```
@article{jaques2019way,
  title={Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog},
  author={Jaques, Natasha and Ghandeharioun, Asma and Shen, Judy and Ferguson, Craig and Jones, Noah, and Lapedriza, Agata and Gu, Shixiang and Picard, Rosalind},
  journal={arXiv preprint arXiv:},
```

For hierarchical reinforcement learning for open-domain dialog, refer to:
```
@article{saleh2019hier,
  title={Hierarchical Reinforcement Learning for Open-Domain Dialog},
  author={Saleh, Abdelrhman and Jaques, Natasha and Ghandeharioun, Asma and Shen, Judy and Picard, Rosalind},
  journal={arXiv preprint arXiv:0000000000},
  year={2019}
}
```

For interactive evaluation, use of Reddit dataset, miscellaneous use-cases, refer to [the following paper](https://arxiv.org/abs/1906.09308):
```
@article{ghandeharioun2019approximating,
  title={Approximating Interactive Human Evaluation with Self-Play for Open-Domain Dialog Systems},
  author={Ghandeharioun, Asma and Shen, Judy and Jaques, Natasha and Ferguson, Craig and Jones, Noah, and Lapedriza, Agata and Picard, Rosalind},
  journal={arXiv preprint arXiv:1906.09308},
  year={2019}
}
```

### Related Work
* Park, Y., Cho, J., & Kim, G. (2018, June). [*A Hierarchical Latent Structure for Variational Conversation Modeling*](https://www.aclweb.org/anthology/N18-1162). In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) (pp. 1792-1801).
* Serban, I. V., Sordoni, A., Lowe, R., Charlin, L., Pineau, J., Courville, A., & Bengio, Y. (2017, February). [*A hierarchical latent variable encoder-decoder model for generating dialogues*](https://arxiv.org/pdf/1605.06069.pdf). In Thirty-First AAAI Conference on Artificial Intelligence.
* Sordoni, A., Bengio, Y., Vahabi, H., Lioma, C., Grue Simonsen, J., & Nie, J. Y. (2015, October). [*A hierarchical recurrent encoder-decoder for generative context-aware query suggestion*](https://arxiv.org/pdf/1507.02221.pdf). In Proceedings of the 24th ACM International on Conference on Information and Knowledge Management (pp. 553-562). ACM.
