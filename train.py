
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

import argparse as ap


from utils import load_data

MODELS_LIST = [
    ('gradient_boosting', GradientBoostingClassifier(n_estimators=100)),
    ('lda', LinearDiscriminantAnalysis(solver='svd')),
    ('svc', SVC(C=1, kernel='linear')),
    ('knn', KNeighborsClassifier(n_neighbors=4)),
    ('random_forest', RandomForestClassifier()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=400)),
    ('decision_tree', DecisionTreeClassifier(max_depth=100)),
    ('xgb', XGBClassifier(learning_rate=0.05, n_estimators=200))
]
MODEL_NAMES = [name for name, _ in MODELS_LIST]
MODEL_NAMES_STR = ','.join(MODEL_NAMES)

EXPERIMENT_AVAILABLES = range(1, 15)
SUBJECT_AVAILABLES = range(1, 110)

VERBOSE = False

def local_print(msg):
    if VERBOSE:
        print(msg)

def check_args(args):
    model_name = args.model
    subject = int(args.subject)
    experiment = int(args.experiment)
    output = args.output
    verbose = args.verbose
    no_save_model = args.no_save_model

    if model_name not in MODEL_NAMES:
        raise ValueError(f'Model name not valid. Availables models: {MODEL_NAMES_STR}')
    
    if subject not in SUBJECT_AVAILABLES:
        raise ValueError(f'Subject not valid. Availables subjects: {SUBJECT_AVAILABLES}')
    
    if experiment not in EXPERIMENT_AVAILABLES:
        raise ValueError(f'Experiment not valid. Availables experiments: {EXPERIMENT_AVAILABLES}')
    
    try:
        choosed_model = [model for name, model in MODELS_LIST if name == model_name][0]
    except IndexError:
        raise ValueError(f'Model name not valid. Availables models: {MODEL_NAMES_STR}')
    
    dir_output = os.path.dirname(output)
    if not os.path.exists(dir_output) and dir_output != '':
        raise ValueError(f'Output directory path not valid: {output}')
    
    return model_name, choosed_model, subject, experiment, output, verbose, no_save_model

def get_epochs(raw):
    event_id = {'T1': 1, 'T2': 2}
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=VERBOSE)

    tmin = -0.5
    tmax = 4
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=VERBOSE)

    return epochs, event_dict, raw

def get_X_y(epochs):
    X = epochs.get_data()
    y = epochs.events[:, -1] - 1

    return X, y

if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-s', '--subject', type=int, help='Subject number', required=True)
    parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)
    parser.add_argument('-m', '--model', type=str, help=f'Model name.\nAvailables models: {MODEL_NAMES_STR}', required=False, default='lda')
    parser.add_argument('-o', '--output', type=str, help='Output path file', required=False, default='output_model/model.joblib')
    parser.add_argument('-nsmdl', '--no-save-model', action='store_true', help='Save model', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()

    model_name, choosed_model, subject, experiment, output, VERBOSE, no_save_model = check_args(args)
    
    
    local_print(f'Using model: {model_name}')
    local_print(f'Using subject: {subject}')
    local_print(f'Using experiment: {experiment}')
    local_print(f'Using output: {output}')
    local_print(f'Using verbose: {VERBOSE}')
    local_print("\n")
    
    
    raw = load_data(subject, experiment, VERBOSE)
    epochs, event_dict, raw = get_epochs(raw)
    X, y = get_X_y(epochs)

    shuffle_split = ShuffleSplit(n_splits=7, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('csp', mne.decoding.CSP(n_components=4)),
        (f'clf {model_name}', choosed_model)
    ], verbose=VERBOSE)
    scores = cross_validate(pipeline, X, y, cv=shuffle_split, n_jobs=1, return_estimator=True, verbose=VERBOSE)
    
    score = scores['test_score']
    local_print("\n")
    local_print("Cross validation scores:")
    local_print(f"Raw: {score}")
    local_print(f"Accuracy: {np.mean(score)} (+/- {np.std(score)})")

    # pipeline = pipeline.fit(X, y)
    # save model
    if no_save_model:
        exit(1)
    joblib.dump(scores['estimator'][0], output)