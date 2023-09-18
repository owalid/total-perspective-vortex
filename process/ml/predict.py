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

from utils import load_data, SUBJECT_AVAILABLES

VERBOSE = False
EXPERIMENT_AVAILABLES = range(1, 15)

def check_args(args):
    subject = int(args.subject)
    experiment = int(args.experiment)
    model_path = args.model_path
    VERBOSE = args.verbose

    if subject not in SUBJECT_AVAILABLES:
        raise ValueError(f'Invalid subject number. Availables: {SUBJECT_AVAILABLES}')
    if experiment not in EXPERIMENT_AVAILABLES:
        raise ValueError(f'Invalid experiment number. Availables: {EXPERIMENT_AVAILABLES}')
    if not os.path.exists(model_path):
        raise ValueError(f'Model path not exists: {model_path}')
    
    choosed_model = joblib.load(model_path)

    return choosed_model, subject, experiment, VERBOSE


def get_epochs(raw):
    event_id = {'T1': 1, 'T2': 2}
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=VERBOSE)

    tmin = -0.5
    tmax = 4
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=VERBOSE)

    return epochs, event_dict, raw

if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-s', '--subject', type=int, help='Subject number', required=True)
    parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)
    parser.add_argument('-m', '--model-path', type=str, help=f'Model path', required=False, default='./output_model/model.joblib')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()

    choosed_model, subject, experiment, VERBOSE = check_args(args)


    raw = load_data(subject, experiment, VERBOSE=VERBOSE)
    epochs, event_dict, raw = get_epochs(raw)

    X = epochs.get_data()
    y = epochs.events[:, -1] - 1
    good_prediction = 0
    print("Epochs [prediction] [real] [correct]")
    for ii in range(X.shape[0]):
        X_ = X[ii:ii+1, :, :]
        y_ = y[ii:ii+1]
        y_pred = choosed_model.predict(X_)
        predict_correct = y_pred[0] == y_[0]
        good_prediction += 1 if predict_correct else 0 
        color = Fore.GREEN if predict_correct else Fore.RED
        print(f'Epoch {ii:<5} {y_pred[0]:<10} {y_[0]:<5} {color}{Style.BRIGHT}[{predict_correct}]{Style.RESET_ALL}')
    print(f'Accuracy: {good_prediction/X.shape[0]*100}%')
    
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