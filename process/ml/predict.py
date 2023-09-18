import os
import sys
import random
import joblib
import numpy as np

import mne
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import glob

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier


from sklearn.pipeline import Pipeline
from sklearn.model_selection import  ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from mne_realtime import LSLClient, MockLSLStream
import argparse as ap
from colorama import *

import json

from utils import load_data, SUBJECT_AVAILABLES

VERBOSE = False
CHOICE_TRAINING = ['all', 'hands_vs_feet', 'left_vs_right', 'hands_vs_feet', 'left_vs_right']

def check_args(args):
    subject = args.subject
    experiment = args.experiment
    model_path = args.model_path
    subject = args.subject
    directory_dataset = args.directory_dataset
    output_file = args.output_file
    VERBOSE = args.verbose

    if subject == 'all':
        subject = SUBJECT_AVAILABLES
    elif ',' in subject:
        subject = [int(s) for s in subject.split(',')]
    else:
        subject = [int(subject)]

    dir_output_file = os.path.dirname(output_file)
    if not os.path.exists(dir_output_file) and dir_output_file != '':
        raise ValueError(f'output_file directory path not valid: {output_file}')
    
    # verify subject
    for s in subject:
        if s not in SUBJECT_AVAILABLES:
            raise ValueError(f'Invalid subject number. Availables: {SUBJECT_AVAILABLES}')
    if experiment not in CHOICE_TRAINING:
        raise ValueError(f'Experiment not valid. Availables experiments: {CHOICE_TRAINING}')
    if not os.path.exists(model_path):
        raise ValueError(f'Model path not exists: {model_path}')
    
    if not os.path.exists(directory_dataset):
        raise ValueError(f'Directory dataset not exists: {directory_dataset}')
    if not os.path.isdir(directory_dataset):
        raise ValueError(f'Directory dataset is not a directory: {directory_dataset}')
    if directory_dataset[-1] == '/':
        directory_dataset = directory_dataset[:-1]
    
    pipeline = joblib.load(model_path)

    return pipeline, subject, experiment, directory_dataset, output_file, VERBOSE


def get_epochs(raw):
    event_id = {'T1': 1, 'T2': 2}
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=VERBOSE)

    tmin = -0.5
    tmax = 2
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=VERBOSE)

    return epochs, event_dict, raw

if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-e', '--experiment', type=str, help='Type training', required=False, choices=CHOICE_TRAINING, default='hands_vs_feet')
    parser.add_argument('-o', '--output-file', type=str, help='Output file', required=False, default='results.json')
    parser.add_argument('-s', '--subject', type=str, help='Subject number, sequence of subjects (separated by comma) or all', required=True)
    parser.add_argument('-m', '--model-path', type=str, help=f'Model path', required=False, default='./output_model/model.joblib')
    parser.add_argument('-d', '--directory-dataset', type=str, help='Directory dataset', required=False, default='../../files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()

    pipeline, subject, experiment, directory_dataset, output_file, VERBOSE = check_args(args)

    results = []
    for s in subject:
        print(f"[+] Subject: S{s:03d}")
        raw = load_data(s, experiment, directory_dataset, VERBOSE=VERBOSE)
        epochs, event_dict, raw = get_epochs(raw)

        X = epochs.get_data()
        y = epochs.events[:, -1] - 1

        y_pred = pipeline.predict(X)

        # 
        predict_correct = y_pred == y
        good_prediction = np.sum(predict_correct)
        print("Epochs [prediction] [real] [correct]")
        for ii in range(X.shape[0]):
            color = Fore.GREEN if predict_correct[ii] else Fore.RED
            print(f'Epoch {ii:<5} {y_pred[ii]:<10} {y[ii]:<5} {color}{Style.BRIGHT}[{predict_correct[ii]}]{Style.RESET_ALL}')
        print(f'Accuracy: {good_prediction/X.shape[0]*100}%')
        results.append({'subject': s, 'accuracy': good_prediction/X.shape[0]})
        print(good_prediction/X.shape[0]*100)
        print('')

    with open(output_file, 'w') as f:
        json.dump(results, f)


        # exit(0)
        # good_prediction = 0
        # print("Epochs [prediction] [real] [correct]")
        # for ii in range(X.shape[0]):
        #     X_ = X[ii:ii+1, :, :]
        #     y_ = y[ii:ii+1]
        #     y_pred = pipeline.predict(X)
        #     print(y_pred)
        #     exit(0)
        #     predict_correct = y_pred[0] == y_[0]
        #     good_prediction += 1 if predict_correct else 0 
        #     color = Fore.GREEN if predict_correct else Fore.RED
        #     print(f'Epoch {ii:<5} {y_pred[0]:<10} {y_[0]:<5} {color}{Style.BRIGHT}[{predict_correct}]{Style.RESET_ALL}')
        # print(f'Accuracy: {good_prediction/X.shape[0]*100}%')
        # results.append({'subject': s, 'accuracy': good_prediction/X.shape[0]})
        # print('')
        
        # host = 'localhost'
        # n_epochs = 200
        # good_prediction = 0
        # with MockLSLStream(host, raw, 'eeg'):
        #     with LSLClient(info=raw.info, host=host, wait_max=5, buffer_size=1) as client:
        #         print("Epochs: [prediction] [real] [correct]")
        #         for ii in range(epochs.get_data().shape[0]):
        #             client_info = client.get_measurement_info()
        #             sfreq = int(client_info['sfreq'])
        #             epoch = client.get_data_as_epoch(n_samples=sfreq)

        #             X = epoch.get_data()
        #             y = epoch.events[:, -1]
        #             y_pred = choosed_model.predict(X)
        #             predict_correct = y_pred[0] == y[0]
        #             good_prediction += 1 if predict_correct else 0
        #             print(f'Epoch {ii:<5} {y_pred[0]:<3} {y[0]:<3} [{predict_correct}]\n')
        #         print(f'Accuracy: {good_prediction/n_epochs*100}%')
        # print('Streams closed')