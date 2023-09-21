import os
import base64
import sys
import random
import joblib
import numpy as np
import time

import socket
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
    stream_mode = args.stream_mode
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
    
    if not stream_mode:
        if not os.path.exists(directory_dataset):
            raise ValueError(f'Directory dataset not exists: {directory_dataset}')
        if not os.path.isdir(directory_dataset):
            raise ValueError(f'Directory dataset is not a directory: {directory_dataset}')
        if directory_dataset[-1] == '/':
            directory_dataset = directory_dataset[:-1]
    
    pipeline = joblib.load(model_path)

    return pipeline, subject, experiment, directory_dataset, output_file, stream_mode, VERBOSE


def get_epochs(raw):
    event_id = {'T1': 1, 'T2': 2}
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=VERBOSE)

    tmin = -0.5
    tmax = 2
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=VERBOSE)

    return epochs, event_dict, raw


def process_with_stream(pipeline, s, experiment, results):
    try:
        address = ('localhost', 5000)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(address)
        client.sendall(f'{s}:{experiment}\n'.encode())
        good_prediction = 0
        predictions = 0
        end = False
        while not end:
            # now i will receive the data of each raw epoch from the server as json
            final_data = b''
            if predictions > 0:
                client.sendall(b'next')
            while True:
                l = client.recv(4096**2)
                final_data += l
                if final_data == b'end':
                    end = True
                    break
                if l[-4:] == b'\x00\x00\x00\x00': # \x00\x00\x00\x00 is the end of the data
                    final_data = final_data[:-4]
                    break
            if end:
                break
            time_start = time.time()
            data_b64 = final_data.decode('ascii')
            data_decoded = base64.b64decode(data_b64)
            data = json.loads(data_decoded)
            epoch_data = np.array(data['epoch'])
            label = data['label']
            sfreq = data['sfreq']

            predictions += 1
            # now i will convert the data to a mne.RawArray
            epoch_data = mne.filter.filter_data(epoch_data, sfreq=sfreq, l_freq=8, h_freq=30, verbose=False)
            epoch_data = np.array([epoch_data])
            y_pred = pipeline.predict(epoch_data)
            predict_correct = y_pred[0] == label
            good_prediction += 1 if predict_correct else 0
            color = Fore.GREEN if predict_correct else Fore.RED
            time_end = time.time()
            delta_time = time_end - time_start
            delta_time = f'{delta_time:.2f}'
            predict_correct = f"[{predict_correct}]"
            print(f'Epoch {predictions:<5} {y_pred[0]:<10} {label:<5} {color}{Style.BRIGHT}{predict_correct:<8}{Style.RESET_ALL} {delta_time:<5}s')
        results.append({'subject': s, 'accuracy': good_prediction/predictions})
        print(f'Accuracy: {good_prediction/predictions*100}')
        print("\n")
    except ConnectionRefusedError:
        raise ValueError('Stream not available, please start the stream server.')
    except Exception:
        raise ValueError("Unexpected error: please make sure that the stream server is running and dataset are available.")
    return results

def process_without_stream(pipeline, s, experiment, directory_dataset, results):
    raw = load_data(s, experiment, directory_dataset, VERBOSE=VERBOSE)
    epochs, event_dict, raw = get_epochs(raw)

    time_start = time.time()
    X = epochs.get_data()
    y = epochs.events[:, -1] - 1
    y_pred = pipeline.predict(X)

    predict_correct = y_pred == y
    good_prediction = np.sum(predict_correct)
    time_end = time.time()
    delta_time = time_end - time_start
    delta_time = f'{delta_time:.2f}'
    for ii in range(X.shape[0]):
        color = Fore.GREEN if predict_correct[ii] else Fore.RED
        current_predi = f"[{predict_correct[ii]}]"
        print(f'Epoch {ii:<5} {y_pred[ii]:<10} {y[ii]:<5} {color}{Style.BRIGHT}{current_predi:<8}{Style.RESET_ALL} {delta_time:<5}s')
    print(f'Accuracy: {good_prediction/X.shape[0]*100}%')
    results.append({'subject': s, 'accuracy': good_prediction/X.shape[0]})
    print(good_prediction/X.shape[0]*100)
    print('')
    return results



if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-strm', '--stream-mode', action='store_true', help='Stream mode, When this flag is enabled, the program will wait for the data from the server at port 5000.', default=False)
    parser.add_argument('-e', '--experiment', type=str, help='Type training', required=False, choices=CHOICE_TRAINING, default='hands_vs_feet')
    parser.add_argument('-o', '--output-file', type=str, help='Output file', required=False, default='results.json')
    parser.add_argument('-s', '--subject', type=str, help='Subject number, sequence of subjects (separated by comma) or all', required=True)
    parser.add_argument('-m', '--model-path', type=str, help=f'Model path', required=False, default='./output_model/model.joblib')
    parser.add_argument('-d', '--directory-dataset', type=str, help='Directory dataset', required=False, default='../../files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()

    pipeline, subject, experiment, directory_dataset, output_file, stream_mode, VERBOSE = check_args(args)
    
    results = []
    for s in subject:
        print(f"[+] Subject: S{s:03d}")
        print("Epochs [prediction] [real] [correct] [time]")
        if stream_mode:
            results = process_with_stream(pipeline, s, experiment, results)
        else:
            results = process_without_stream(pipeline, s, experiment, directory_dataset, results)

    # global accuracy
    print("Global accuracy:")
    print(f"Accuracy: {np.mean([r['accuracy'] for r in results])*100}%")
    with open(output_file, 'w') as f:
        json.dump(results, f)