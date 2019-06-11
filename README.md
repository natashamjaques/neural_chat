# Variational Hierarchical Conversation RNN (VHCR)
[PyTorch 0.4](https://github.com/pytorch/pytorch) Implementation of ["A Hierarchical Latent Structure for Variational Conversation Modeling"](https://arxiv.org/abs/1804.03424) (NAACL 2018 Oral)
* [NAACL 2018 Oral Presentation Video](https://vimeo.com/277671819)

## Prerequisites
Install Python packages
```
pip install -r requirements.txt
```

Setup python path to include repo
```
python setup.py develop
```

## Download & Preprocess data
Following scripts will

1. Create directories `./datasets/cornell/` and `./datasets/ubuntu/` respectively.

2. Download and preprocess conversation data inside each directory.

### for [Cornell Movie Dialogue dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
```
python cornell_preprocess.py
    --max_sentence_length (maximum number of words in sentence; default: 30)
    --max_conversation_length (maximum turns of utterances in single conversation; default: 10)
    --max_vocab_size (maximum size of word vocabulary; default: 20000)
    --max_vocab_frequency (minimum frequency of word to be included in vocabulary; default: 5)
    --n_workers (number of workers for multiprocessing; default: os.cpu_count())
```

### for [Ubuntu Dialog Dataset](http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/)
```
python ubuntu_preprocess.py
    --max_sentence_length (maximum number of words in sentence; default: 30)
    --max_conversation_length (maximum turns of utterances in single conversation; default: 10)
    --max_vocab_size (maximum size of word vocabulary; default: 20000)
    --max_vocab_frequency (minimum frequency of word to be included in vocabulary; default: 5)
    --n_workers (number of workers for multiprocessing; default: os.cpu_count())
```

### Download pretrained deepmoji model

```
cd torchMoji
python scripts/download_weights.py
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

If you use this code or dataset as part of any published research, please refer the following paper.

```
@inproceedings{VHCR:2018:NAACL,
    author    = {Yookoon Park and Jaemin Cho and Gunhee Kim},
    title     = "{A Hierarchical Latent Structure for Variational Conversation Modeling}",
    booktitle = {NAACL},
    year      = 2018
    }
```