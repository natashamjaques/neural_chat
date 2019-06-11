import argparse
import os
import pandas as pd
import numpy as np


def add_total_validation_loss(series):
    series['validation_total_loss'] = np.nansum([series['validation_emoji_loss'],
                                                series['validation_infersent_loss'],
                                                series['validation_loss']])
    return series


def clean_checkpoints(dir):
    """
    This function adds the total validation loss to the metrics.csv
    and removes the .pkl checkpoints that don't have the optimal validation loss.
    """

    OPTIMIZATION_COLS = ['validation_total_loss', 'validation_loss']
    # for _dataset_dir in os.listdir(dir):
    #     dataset_dir = os.path.join(dir, _dataset_dir)
    #     if not os.path.isdir(dataset_dir):
    #         print(f'{dataset_dir} is not a valid dataset_dir. Skipping.')
    #         continue
    for _model_category_dir in os.listdir(dir):
        model_category_dir = os.path.join(dir, _model_category_dir)
        if not os.path.isdir(model_category_dir):
            print(f'{model_category_dir} is not a valid model_category_dir. Skipping.')
            continue
        for _model_dir in os.listdir(model_category_dir):
            model_dir = os.path.join(model_category_dir, _model_dir)
            if not os.path.isdir(model_dir):
                print(f'{model_dir} is not a valid model_dir. Skipping.')
                continue
            metrics_fname = os.path.join(model_dir, 'metrics.csv')

            print('\n\nWorking on', model_dir)
            try:
                df = pd.read_csv(metrics_fname)
                cols = df.columns.values
                if 'validation_total_loss' not in cols or 'epoch' not in cols:
                    df.rename(columns={'Unnamed: 0': 'epoch'}, inplace=True)
                    df = df.apply(add_total_validation_loss, axis=1)
                    df.to_csv(metrics_fname)
                
                keeping_epochs = []
                for col in OPTIMIZATION_COLS:
                    df = df.sort_values(by=[col]).reset_index(drop=True)
                    keeping_epoch = df.iloc[0]['epoch']
                    keeping_epochs.append(keeping_epoch)
                print('Intending to keep epoch', keeping_epochs)

                files_to_remove = []
                numbers_to_remove = []
                for _pickle_file in os.listdir(model_dir):
                    pickle_file = os.path.join(model_dir, _pickle_file)
                    if not _pickle_file.endswith('.pkl'):
                        continue
                    pickle_number = int(_pickle_file[:-4])
                    if pickle_number not in keeping_epochs:
                        files_to_remove.append(pickle_file)
                        numbers_to_remove.append(pickle_number)

                print("Planning to remove the following files:")
                for f in files_to_remove: print(f)
                print("Corresponding to the following epochs:", 
                    sorted(numbers_to_remove))
                answer = input("Proceed? y/N: ")
                if answer.lower() != 'y':
                    print("Okay, not removing these files.")
                    continue
                
                print("Okay, removing those files.")
                for f in files_to_remove:
                    os.remove(f)
            
            except Exception as e:
                print("Exception occurred:")
                print(str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str)
    args = parser.parse_args()

    clean_checkpoints(args.checkpoint_dir)
