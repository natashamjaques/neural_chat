# Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog

This code supports training conversational dialog models from a static batch of data (with no exploration) using Q-learning and KL-control, as described in this [paper](https://arxiv.org/abs/). The main file is ```way_off_policy_batch_rl.py```, which is invoked using ```run_rl.py```. Using a pre-trained model checkpoint, and a .csv file of interaction data, the file samples batches of data from a replay buffer and trains with respect to the Q-learning loss. Several enhancements enable better learning:
  * KL-control to penalize divergence from the pre-trained prior.
  * Psi-learning, which uses a logsumexp instead of a max over the estimated future reward. Equivalent to entropy maximization and important for generating diverse samples.
  * Model averging, which allows you to use an aggregated prior computed by averaging the predictions of several models. This can be pre-computed using ```model_averaging.py```
  * Monte-carlo Target Q-value estimation, in which dropout is used to get a lower bound of the Target Q-value.

These options can be toggled using the hyperparameter settings in ```run_rl.py```. We also include code for a discrete version of [Batch Constrained Q-learning](https://arxiv.org/abs/1812.02900) in ```dbcq.py```.

Interaction data was collected using http://neural.chat. To deploy your own model on the web, please see the related repo https://github.com/asmadotgh/neural_chat_web.

## Prerequisites
See the top level README for libraries and installation instructions.

## Data format
Data for the 

## Training


### Rewards

First, calculate rewards offline:
```
python model/rl/rewards.py --raw --experience_path=<path_to_experience_csv_file> --save_path=<save_path>
```

Then, run:
```
python model/rl/run_rl.py -r 'reward_you' 'reward_what' -rw 2.0 1.5
```

## Interacting with RL Models

Use the following command to interact with / talk to a saved model checkpoint:
```
python model/interact.py --debug --checkpoint=<path_to_your_checkpoint_directory>
```

## Reference
If you use this code, please cite our work:
```
@article{jaques2019way,
  title={Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog},
  author={Jaques, Natasha and Ghandeharioun, Asma and Shen, Judy and Ferguson, Craig and Jones, Noah, and Lapedriza, Agata and Gu, Shixiang and Picard, Rosalind},
  journal={arXiv preprint arXiv:},
  year={2019}
}
```