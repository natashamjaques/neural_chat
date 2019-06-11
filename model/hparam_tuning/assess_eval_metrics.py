""" This file allows multiple hyperparameter turning jobs to be run on a server.
    After each job, an email is sent to notify desired people of its completion.

    Must specify a text job file that contains the names and specifications of
    each job. Each job has 5 lines, containing:
    1) the name, 2) the types of model settings to search, 3) the hparams that
    will stay constant, 4) the hparams and values to seach, 5) whether to take
    the 'cross_product' of all searched hparams values, or iterate through the 
    lists of values for each hparam 'in_order'.

    An example job file format is as follows:

        cornell_HRED
        ['emotion', 'infersent', 'emoinfer', 'input_only']
        --data=cornell --model=HRED --emo_weight=25 --infersent_weight=30000
        --encoder_hidden_size=[400,600,800] --decoder_hidden_size=[400,600,800]
        in_order

        cornell_VHRED
        ['emotion', 'infersent']
        --data=cornell --model=VHRED --emo_weight=25 --infersent_weight=30000
        --embedding_size=[400,500] --word_drop=[0,.1,.25] --dropout=[0,.1,.2]
        cross_product

    Usage: python hparam_tune.py --available_gpus 1 2 3
"""    
import pandas as pd
import ast
import copy
import argparse
import os
import subprocess
import sys
import socket
import threading
import time

from run_utils import *

MINIMUM_JOB_SECONDS = 1200  # 20 minutes
PRINT_LAST_X_LINES = 200
FINISHED = 3
ERROR = 2
SUCCESS = 0
WARNING = 1
DEFAULT_STEPS = 40
UNIVERSAL_HPARAMS = ' --extra_save_dir=hparam_tune' + \
                    ' --batch_size=32 --eval_batch_size=32' + \
                    ' --evaluate_embedding_metrics --n_epoch=' + \
                    str(DEFAULT_STEPS)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--models_file', type=str, 
                        default='./jobs_to_run/models_to_assess.txt')
    parser.add_argument('--dataframe_path', type=str, default=None,
                        help='Path where results df will be saved.')
    parser.add_argument('--dataset', type=str, default='valid')
    parser.add_argument('--available_gpus', nargs='+', required=True,
                        help='<Required> IDs of available GPUs', )

    # Parse arguments
    kwargs = parser.parse_args()

    # Namespace => Dictionary
    kwargs = vars(kwargs)

    return kwargs


def run_eval_job(checkpoint, kwargs_dict, gpu_id):
    """ Runs a system command for evaluating a model
    """
    job = 'CUDA_VISIBLE_DEVICES=' + str(gpu_id) + ' '
    job += 'python ../eval_embed.py --dataset=' + kwargs_dict['dataset'] + ' '
    job += '--checkpoint=' + checkpoint + ' '
    job += '--results_df_path=' + kwargs_dict['dataframe_path']
    print("\nRunning job", job)
    
    t0 = time.time()

    # execute the command
    p = subprocess.Popen(job, stdout=subprocess.PIPE, 
                         stderr=subprocess.STDOUT, shell=True)
    # Wait until command is finished to proceed.
    (output, err) = p.communicate()  
    p_status = p.wait()  # Get status of job

    # Record time taken
    t1 = time.time()
    total_secs = t1 - t0
    hours, mins, secs = get_secs_mins_hours_from_secs(total_secs)
    time_str = "Job ended. Total time taken: " + str(int(hours)) + "h " \
        + str(int(mins)) + "m " + str(int(secs)) + "s"
    print(time_str)
    
    print('Process', job, 'finished with code', p_status, 'output was:')
    try:
        output = output.decode("utf-8") if output is not None else ''
        err = err.decode("utf-8") if err is not None else ''
    except Exception as e:
        print("Could not decode model output or error")
        print(str(e))
    print(output)
    print(err)
    sys.stdout.flush()


def load_model_checkpoints(filename):
    f = open(filename, 'r')
    checkpoints = f.readlines()
    return [c.strip() for c in checkpoints]


def assess_models_on_gpus(checkpoints, kwargs_dict):
    all_gpus = kwargs_dict['available_gpus']
    active_checkpoint_ids = list(range(len(checkpoints)))
    currently_available_gpus = copy.deepcopy(all_gpus)
    active_threads = []
    checkpoints_iterator = 0

    last_avail_gpus = len(currently_available_gpus)

    while True:
        while len(currently_available_gpus) > 0:
            ckpt_id = active_checkpoint_ids[checkpoints_iterator]
            ckpt = checkpoints[ckpt_id]
            gpu = currently_available_gpus.pop()
            print("Popped gpu", gpu, "GPUs left:", currently_available_gpus)

            t = threading.Thread(target=run_eval_job, 
                                 name=ckpt+'_gpu'+str(gpu),
                                 args=(ckpt,kwargs_dict,gpu,))
            t.daemon = True
            t.start()
            active_threads.append(t)
            print("Active threads:")
            for t in active_threads:
                print('\t', t.name, "alive?", t.isAlive())

            # Pause a second before launching the next job to ensure model
            # directories created with different timestamps.
            time.sleep(2)

            checkpoints_iterator += 1
            if checkpoints_iterator >= len(active_checkpoint_ids):
                break
    
        # Filter out finished threads and find free GPUs
        active_threads = [t for t in active_threads if t.isAlive()]
        currently_available_gpus = get_available_gpus(active_threads, all_gpus)
        if last_avail_gpus != len(currently_available_gpus):
            print("active threads:", len(active_threads), 
                "free gpus:", currently_available_gpus)
        last_avail_gpus = len(currently_available_gpus)

        if len(active_threads) == 0:
            break
        else:
            print('Still waiting on the following threads')
            for t in active_threads:
                print('\t', t.name, "alive?", t.isAlive())
            print('Sleeping for 30 seconds...')
            time.sleep(30)


def get_available_gpus(active_threads, all_gpus):
    thread_names = [t.name for t in active_threads]
    active_gpus = []
    for t in thread_names:
        idx = t.find('_gpu')
        active_gpus.append(t[idx+4:])
    
    return [g for g in all_gpus if g not in active_gpus]


if __name__ == '__main__':
    hostname = socket.gethostname()
    
    kwargs_dict = parse_args()
    models_file = kwargs_dict['models_file']

    checkpoints_list = load_model_checkpoints(models_file)

     # Send initial test mail
    send_email("Starting to evaluate " + models_file + " on " + hostname,
               "This is a test email to confirm email updates are enabled " \
               + "for evaluating jobs in " + models_file + ". The models to " \
               + "evaluate are: " + str(checkpoints_list))

    # Run all jobs
    assess_models_on_gpus(checkpoints_list, kwargs_dict)

    # Send final email
    send_email("Finished assessing all models on " + hostname, 
               "Congratulations, all of the models in the file " + models_file + " have been assessed.")