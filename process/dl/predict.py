import os
import numpy as np
import argparse as ap
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.models import load_model
from mne.io import concatenate_raws
import json
from colorama import *

from utils import preprocess_raw, load_one_subject, SUBJECT_AVAILABLES

VERBOSE = False
CHOICE_TRAINING = ['all', 'hands_vs_feet', 'left_vs_right', 'hands_vs_feet', 'left_vs_right']

def local_print(msg):
    if VERBOSE:
        print(msg)

def check_args(args):
    subject = args.subject
    model_path = args.model_path
    directory_dataset = args.directory_dataset
    VERBOSE = args.verbose
    type_training = args.type_training
    output_file = args.output_file
    
    if type_training not in CHOICE_TRAINING:
        raise ValueError(f'Type training must be in {CHOICE_TRAINING}')
    
    if subject == 'all':
        subject = SUBJECT_AVAILABLES
    elif ',' in subject:
        subject = [int(s) for s in subject.split(',')]
    else:
        subject = [int(subject)]

    # verify subject
    for s in subject:
        if s not in SUBJECT_AVAILABLES:
            raise ValueError(f'Invalid subject number. Availables: {SUBJECT_AVAILABLES}')
    
    dir_output_file = os.path.dirname(output_file)
    if not os.path.exists(dir_output_file) and dir_output_file != '':
        raise ValueError(f'output_file directory path not valid: {output_file}')

    if not os.path.exists(model_path):
        raise ValueError(f'Model path not exists: {model_path}')
    
    if not os.path.exists(directory_dataset):
        raise ValueError(f'Directory dataset not exists: {directory_dataset}')
    if not os.path.isdir(directory_dataset):
        raise ValueError(f'Directory dataset is not a directory: {directory_dataset}')
    if directory_dataset[-1] == '/':
        directory_dataset = directory_dataset[:-1]

    
    choosed_model = load_model(model_path)

    return choosed_model, subject, directory_dataset, type_training, output_file, VERBOSE


def load_data(subject_num, directory_dataset, type_training):
    raws = []
    if VERBOSE:
        print(f'Loading data...')
    
    raws = load_one_subject(raws, subject_num, directory_dataset, type_training)
    raws_result = concatenate_raws(raws)
    return raws_result


if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-t', '--type-training', type=str, help='Type training', required=False, choices=CHOICE_TRAINING, default='all')
    parser.add_argument('-o', '--output-file', type=str, help='Output file', required=False, default='results.json')
    parser.add_argument('-s', '--subject', type=str, help='Subject number, sequence of subjects (separated by comma) or all', required=True)
    parser.add_argument('-m', '--model-path', type=str, help=f'Model path', required=False, default='./output_model/model.h5')
    parser.add_argument('-d', '--directory-dataset', type=str, help='Directory dataset', required=False, default='../../files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()
    results = []
    choosed_model, subject, directory_dataset, type_training, output_file, VERBOSE = check_args(args)



    for s in subject:
        raw = load_data(s, directory_dataset, type_training)
        raw, events, event_dict, picks, epochs = preprocess_raw(raw, type_training)
        X = epochs.get_data()
        n_channels = X.shape[1]
        input_window_size = X.shape[2]
        input_shape = (1, n_channels, input_window_size)
        X = X.reshape(X.shape[0], 1, n_channels, input_window_size)
        X = normalize(X, axis=1, order=0)
        y = epochs.events[:, -1] - 1
        y = to_categorical(y)
        local_print(f'Predicting...')
        y_pred = choosed_model.predict(X)


        print(f"[+] Subject: S{s:03d}")
        local_print(f"y = {y}")
        local_print(f'Predictions: {y_pred}')
        predict_correct = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
        good_prediction = 0
        for i in range(len(predict_correct)):
            if predict_correct[i].all():
                good_prediction += 1
        results.append({
            'subject': s,
            'accuracy': good_prediction/len(predict_correct)
        })

        print("\nEpochs [prediction] [real] [correct]")
        for ii in range(X.shape[0]):
            color = Fore.GREEN if predict_correct[ii] else Fore.RED
            current_predi = f"[{predict_correct[ii]}]"
            print(f'Epoch {ii:<5} {np.argmax(y_pred[ii]):<10} {np.argmax(y[ii]):<5} {color}{Style.BRIGHT}{current_predi:<8}{Style.RESET_ALL}')
        
        print(f'\nGood predictions: {good_prediction} / {len(predict_correct)}')
        print(f'Bad predictions: {len(predict_correct) - good_prediction} / {len(predict_correct)}')
        print(f'Accuracy: {good_prediction/len(predict_correct)}')
    
    if VERBOSE:
        m = []
        print("\n")
        print("Summary:")
        print(f"Subject {'':<7}  {'':>12} accuracy")
        for r in results:
            print(f"{r['subject']:<10} {round(r['accuracy'], 3):>24}")
            m.append(r['accuracy'])
        m = np.array(m)
        print(f"Mean accuracy of {len(subject)} subjects: {np.mean(m):>19}")

    with open(output_file, 'w') as f:
        json.dump(results, f)
