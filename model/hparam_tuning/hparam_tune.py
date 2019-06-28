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
        ['baseline', emotion', 'infersent', 'emoinfer']
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

    parser.add_argument('--specification', type=str, 
                        default='./jobs_to_run/hparam_tune.txt')
    parser.add_argument('--available_gpus', nargs='+', required=True,
                        help='<Required> IDs of available GPUs', )

    # Parse arguments
    kwargs = parser.parse_args()

    # Namespace => Dictionary
    kwargs = vars(kwargs)

    return kwargs


class JobGroup:
    def __init__(self, name, model_types, stable_hparams, search_string,
                 search_type):
        self.name = name
        self.model_types = model_types
        self.stable_hparams = stable_hparams
        self.search_string = search_string
        self.search_type = search_type
        self.list_of_job_commands = []
        self.current_job_index = 0
        self.done = False

        self.stable_hparams += UNIVERSAL_HPARAMS

    def __str__(self):
        rep = "JobGroup " + self.name + " \n"
        rep += "model_types: " + str(self.model_types) + "\n"
        rep += "stable_hparams: " + self.stable_hparams + "\n"
        rep += "search_string: " + self.search_string + "\n"
        rep += "search_type: " + self.search_type + "\n"
        rep += "done: " + str(self.done) + "\n" 
        return rep
    
    def run_next_job(self, gpu_id):
        """ Runs a system command for a job, returns whether it succeeded and 
            output text to be emailed.

            Returns
                A code indicating whether the job was successful, and
                a string containing text about the job and job output to 
                be mailed to the user
        """
        if self.done:
            message = "No more jobs to run for job group " + self.name
            print(message)
            return FINISHED, message

        job = self.list_of_job_commands[self.current_job_index]
        job = 'CUDA_VISIBLE_DEVICES=' + str(gpu_id) + ' ' + job
        print("\nRunning job", job)
        message = 'Running job -> ' + job

        self.current_job_index += 1
        if self.current_job_index >= len(self.list_of_job_commands) -1:
            self.done = True
        
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
        message += "\n\n" + time_str + "\n\n"
        final_message = "The last " + str(PRINT_LAST_X_LINES) + \
                " lines of job output were:\n\n"
        
        try:
            output = output.decode("utf-8") if output is not None else ''
            err = err.decode("utf-8") if err is not None else ''
            print('Process', job, 'finished with code', p_status, 'output was:')
            print(output)
            print(err)
            lines = output.split('\n')
            tail = "\n".join(lines[-PRINT_LAST_X_LINES:])
            final_message += tail
        except Exception as e:
            print("Could not decode model output or error")
            print(str(e))
            final_message += "ERROR: Could not decode"
        sys.stdout.flush()

        if p_status != 0:
            message += "There was an error running the job:\n"
            message += err
            code = ERROR
        elif total_secs < MINIMUM_JOB_SECONDS:
            message += "The total time taken for the job was suspiciously short."
            print("Warning!", message)
            code = WARNING
        else:
            print("Job finished successfully!")
            code = SUCCESS

        return code, message, final_message

    def run_and_email(self, gpu_id):
        hostname = socket.gethostname()
        status, message, final_message = self.run_next_job(gpu_id)
        if status == FINISHED:
            title = "Congrats! All jobs in the group " + self.name \
                + " have been launched on " + hostname
        elif status == SUCCESS:
            title = "Success! Job is finished on " + hostname
        else:
            title = "Warning! Job finished too quickly " + hostname

        success = send_email(title, message + final_message)
        if not success:
            send_email(title, message)

    def create_list_of_jobs(self):
        self.default_job = 'python ../train.py ' + self.stable_hparams

        search_param_strs = self.search_string.split(' ') 
        self.search_params = {}
        for s in search_param_strs:
            name, values = s.split('=')
            self.search_params[name] = ast.literal_eval(values)

        for model_type in self.model_types:
            model_job = self.default_job + ' ' \
                + get_model_type_hparams(model_type)

            if self.search_type == 'in_order':
                # Check all lengths are the same
                lens = [len(v) for v in self.search_params.values()]
                assert all(l == lens[0] for l in lens)

                # Iterate through lists of hparam values and append jobs to run
                for i in range(lens[0]):
                    job = model_job
                    for k in self.search_params.keys():
                        job += ' ' + k + '=' + str(self.search_params[k][i])
                    self.list_of_job_commands.append(job)

            elif self.search_type == 'cross_product':
                self.recurse_and_append_params(
                    copy.deepcopy(self.search_params), {}, model_job)

            else:
                print("Error! Cannot recognize search type. Should be "
                      "'in_order' or 'cross_product'.")

    def recurse_and_append_params(self, param_settings_left, this_param_dict,
                                  base_job):
        """Recursively finds all hyperparameter combinations for cross_product.

        Performs breadth-first-search.

        Args:
            param_settings_left: A dictionary of lists. The keys are parameters
                (like '--embedding_size'), the values are the list of settings 
                for those parameters that need to be tested (like 
                [1.0, 10.0, 100.0]).
            this_param_dict: A dictionary containing a single setting for each 
                parameter. If a parameter is not in this_param_dict's keys, a 
                setting for it has not been chosen yet.
            base_job: A string representing the basic form of a job command to
                append hyperparams to.
        """
        for key in self.search_params.keys():
            if key in this_param_dict:
                continue
            else:
                this_setting = param_settings_left[key].pop()
                if len(param_settings_left[key]) > 0:
                    # Recursing on remaining parameters
                    self.recurse_and_append_params(
                        copy.deepcopy(param_settings_left), 
                        copy.deepcopy(this_param_dict), base_job)
                this_param_dict[key] = this_setting

        # Add job based on this param dict to list
        job = base_job
        for k in this_param_dict.keys():
            job += ' ' + k + '=' + str(this_param_dict[k])
        self.list_of_job_commands.append(job)


def get_model_type_hparams(model_type):
    if model_type == 'baseline':
        return ""
    elif model_type == 'emotion':
        return "--emotion"
    elif model_type == 'infersent':
        return "--infersent"
    elif model_type == 'emoinfer':
        return "--emotion --infersent"
    elif model_type == 'input_only':
        return "--context_input_only"
    else:
        print("Error! Cannot recognize model type", model_type)


def load_job_groups(filename):
    f = open(filename, 'r')
    lines = f.readlines()

    job_groups = []

    i = 0
    while i < len(lines):
        jobname = lines[i].replace('\n', '')
        model_types = ast.literal_eval(lines[i+1])
        stable_hparams = lines[i+2].replace('\n', '')
        search_str = lines[i+3].replace('\n', '')
        search_type = lines[i+4].replace('\n', '')
        job_group = JobGroup(jobname, model_types, stable_hparams, search_str,
                             search_type)
        job_group.create_list_of_jobs()
        print("Job group", job_group.name, "has", 
              len(job_group.list_of_job_commands), "jobs to run")
        job_groups.append(job_group)
        i = i+6

    return job_groups


def schedule_jobs_on_gpus(job_groups, all_gpus):
    active_job_groups_ids = list(range(len(job_groups)))
    currently_available_gpus = copy.deepcopy(all_gpus)
    active_threads = []
    job_group_iterator = 0

    last_avail_gpus = len(currently_available_gpus)

    while True:
        while len(currently_available_gpus) > 0:
            job_group_id = active_job_groups_ids[job_group_iterator]
            job_group = job_groups[job_group_id]
            gpu = currently_available_gpus.pop()

            t = threading.Thread(target=job_group.run_and_email, 
                                 name=job_group.name+'_gpu'+str(gpu),
                                 args=(gpu,))
            t.daemon = True
            t.start()
            active_threads.append(t)

            # Pause half a second before launching the next job to ensure model
            # directories created with different timestamps.
            time.sleep(0.5)

            if job_group.done:
                active_job_groups_ids.remove(job_group_id)
            if len(active_job_groups_ids) == 0:
                return
            job_group_iterator += 1
            job_group_iterator = job_group_iterator % len(active_job_groups_ids)

        # Filter out finished threads and find free GPUs
        active_threads = [t for t in active_threads if t.isAlive()]
        currently_available_gpus = get_available_gpus(active_threads, all_gpus)
        if last_avail_gpus != len(currently_available_gpus):
            print("active threads:", len(active_threads), 
                "free gpus:", currently_available_gpus)
        last_avail_gpus = len(currently_available_gpus)


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
    jobsfile = kwargs_dict['specification']

    job_groups = load_job_groups(jobsfile)

     # Send initial test mail
    send_email("Starting to run " + jobsfile + " jobs on " + hostname,
               "This is a test email to confirm email updates are enabled for " + jobsfile + " jobs.")

    # Run all jobs
    schedule_jobs_on_gpus(job_groups, kwargs_dict['available_gpus'])

    # Send final email
    send_email("ALL JOBS FINISHED!! on " + hostname, 
               "Congratulations, all of the jobs in the file " + jobsfile + " have finished running.")