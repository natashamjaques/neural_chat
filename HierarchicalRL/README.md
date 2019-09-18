# Hierarchical Reinforcement Learning for Open-Domain Dialog

This code supports training conversational dialog models using hierarchical reinforcement learning, as described in this [paper](https://arxiv.org/abs/1909.07547). The main file is ```hrl_tune.py```, which is invoked using ```run_hrl.py```. The training code initializes a fixed user simulator network and a policy network from a pre-trained [VHRED](https://arxiv.org/abs/1605.06069) model checkpoint. The policy network interacts with the simulator to collect experience and learns to optimize for human-centered rewards through hierarchical reinforcement learning. The hierarchical nature of the framework we propose facilitates learning conversation-level rewards and achieves higher quality ratings according to human evaluators.   

Interaction data was collected using http://neural.chat. To deploy your own model on the web, please see the related repo https://github.com/asmadotgh/neural_chat_web.


## Prerequisites
See the top-level README for libraries and installation instructions. The rewards in ```hrl_tune.py``` require TorchMoji, Infersent, Universal Sentence Encoder, Toxicity, and the Google News Vectors. However, note that not all the requirements are necessary if you are only interested in a limited set of rewards.

### Universal Sentence Encoder Setup
> For more information about the Universal Sentence Encoder, see [here](https://tfhub.dev/google/universal-sentence-encoder-large/3).

If you are interested in the Universal Sentence Encoder (USE) semantic similarity reward run ```./UniversalSentenceEncoder/encoder_setup.py``` to download the pre-trained encoder using ```tensorflow-hub```.

### Toxicity Setup
If you are interested in the Toxicity reward download the Toxic Comments dataset from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and extract it to ```./datasets/Kaggle_Toxic/data/```. Then run ```./Toxicity/toxic.py``` to train the model used to assign toxicity rewards.


## Training
Refer to the top-level README for instructions on training a VHRED model. Make sure the arguments ```--emotion```, ```--infersent```, and the ```--calc_novel_embedding``` are all set to False. Once you have a pre-trained checkpoint, you can start fine-tuning your checkpoint using hierarchical reinforcement learning as follows:

```
python HierarchicalRL/run_hrl.py --vhrl -r 'reward_bot_deepmoji' 'reward_question' 'reward_repetition' -w 0.3 0.2 0.5 \
    --checkpoint=<path_to_your_checkpoint_directory>
```

There are three main training configurations:
  *  ```--vhrl``` which trains the manager and the worker jointly using a policy gradient objective.
  *  ```decoupled_vhrl``` which alternates between training the manager and the worker
  *  ```--reinforce``` which only applies rewards at the worker level using REINFORCE

Exactly one of these training modes need to be specified for correct training. The behavior of these configurations is further described in the paper. Refer to the arguments defined in ```run_hrl.py``` for training hyperparameters.

**Note**: If you get a tensorboard error at this stage run ```pip uninstall tensorboard```  

### Transformers
> For more information about ParlAI refer to the official documentation [here] (https://www.parl.ai/docs/index.html)

First you need to install ParlAI using:
```
python ParlAI/setup.py develop
```

We added a ```redditcasual``` task to ParlAI that can be used to train the transformer on reddit data. To train a transformer you can run:
```
python ParlAI/examples/train_model.py -m transformer/generator -t redditcasual --optimizer adam \
    --adam-eps 1e-9 --betas 0.9,0.98 -lr 1.0 --lr-scheduler noam --warmup-updates 10000 --display-examples True \
    -tr 300 --label-truncate 30 -histsz 10 -mf ParlAI/trained/transformer
```
Feel free to include other arguments and adjust the transformer hyperparameters when running this training script. However, note that you will need to install a more recent PyTorch version to use ParlAI as the version used by this repo (0.4.0) isn't supported.

## Interacting with RL Models

Use the following command to interact with / talk to a saved model checkpoint:
```
python model/interact.py --load_rl_ckpt --rl_ckpt_epoch 100 --checkpoint=<path_to_your_checkpoint_directory>
```

## Reference
If you use this code, please cite our work:
```
@article{saleh2019hier,
  title={Hierarchical Reinforcement Learning for Open-Domain Dialog},
  author={Saleh, Abdelrhman and Jaques, Natasha and Ghandeharioun, Asma and Shen, Judy and Picard, Rosalind},
  journal={arXiv preprint arXiv:1909.07547},
  year={2019}
}
```
