import os
import json
import joblib
import numpy as np

import mne
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate, cross_val_score, ShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

import argparse as ap

from utils import load_data_all, load_data_one, SUBJECT_AVAILABLES
from decomposition.TurboCSP import TurboCSP

CHOICE_TRAINING = ['hands_vs_feet', 'left_vs_right', 'imagery_left_vs_right', 'imagery_hands_vs_feet', 'all']

MODELS_LIST = [
    ('gradient_boosting', GradientBoostingClassifier(n_estimators=100)),
    ('lda', LinearDiscriminantAnalysis(solver='svd', tol=0.0001)),
    ('svc', SVC(C=1, kernel='linear')),
    ('knn', KNeighborsClassifier(n_neighbors=2)),
    ('random_forest', RandomForestClassifier()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=400)),
    ('decision_tree', DecisionTreeClassifier(max_depth=50)),
    ('xgb', XGBClassifier(learning_rate=0.05, n_estimators=200))
]
MODEL_NAMES = [name for name, _ in MODELS_LIST]
MODEL_NAMES_STR = ','.join(MODEL_NAMES)

DECOMPOSITION_ALGORITHMS = [
    ('TurboCSP', TurboCSP(n_components=5)),
    ('MNECSP', mne.decoding.CSP(n_components=5))
]
DECOMPOSITION_ALGORITHMS_NAMES = [name for name, _ in DECOMPOSITION_ALGORITHMS]
DECOMPOSITION_ALGORITHMS_NAMES_STR = ','.join(DECOMPOSITION_ALGORITHMS_NAMES)

VERBOSE = False

def local_print(msg):
    if VERBOSE:
        print(msg)

def check_args(args):
    model_name = args.model
    subject = args.subject
    experiment = args.experiment
    output = args.output
    verbose = args.verbose
    save_model = args.save_model
    decomp_alg = args.decomposition_algorithm
    directory_dataset = args.directory_dataset
    pack_subj = args.pack_subj

    if subject == 'all':
        subject = SUBJECT_AVAILABLES
    elif ',' in subject:
        subject = [int(s) for s in subject.split(',')]
    else:
        subject = [int(subject)]

    if not os.path.exists(directory_dataset):
        raise ValueError(f'Directory dataset not exists: {directory_dataset}')
    if not os.path.isdir(directory_dataset):
        raise ValueError(f'Directory dataset is not a directory: {directory_dataset}')
    if directory_dataset[-1] == '/':
        directory_dataset = directory_dataset[:-1]

    if decomp_alg not in DECOMPOSITION_ALGORITHMS_NAMES:
        raise ValueError(f'Decomposition algorithm not valid. Availables algorithms: {DECOMPOSITION_ALGORITHMS_NAMES_STR}')
    
    if model_name not in MODEL_NAMES:
        raise ValueError(f'Model name not valid. Availables models: {MODEL_NAMES_STR}')
    
    if experiment not in CHOICE_TRAINING:
        raise ValueError(f'Experiment not valid. Availables experiments: {CHOICE_TRAINING}')
    
    if experiment == 'all':
        experiment = CHOICE_TRAINING[:-1]
    else:
        experiment = [experiment]
    
    try:
        choosed_model = [model for name, model in MODELS_LIST if name == model_name][0]
    except IndexError:
        raise ValueError(f'Model name not valid. Availables models: {MODEL_NAMES_STR}')
    
    try:
        choosed_decomp_alg = [alg for name, alg in DECOMPOSITION_ALGORITHMS if name == decomp_alg][0]
    except IndexError:
        raise ValueError(f'Decomposition algorithm not valid. Availables algorithms: {DECOMPOSITION_ALGORITHMS_NAMES_STR}')
    
    dir_output = os.path.dirname(output)
    if not os.path.exists(dir_output) and dir_output != '':
        raise ValueError(f'Output directory path not valid: {output}')
    
    return model_name, choosed_model, decomp_alg, choosed_decomp_alg, subject, experiment, output, save_model, directory_dataset, pack_subj, verbose

def get_epochs(raw):
    event_id = {'T1': 1, 'T2': 2}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=VERBOSE)

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id, -0.5, 3.5, proj=True, picks=picks, baseline=None, preload=True, verbose=VERBOSE)

    return epochs, event_id, raw

def get_X_y(epochs):
    X = epochs.get_data()
    y = epochs.events[:, -1] - 1

    return X, y

def process_model(X, y, epochs, choosed_decomp_alg, choosed_model, need_calculate_mean, VERBOSE):
    scaler = mne.decoding.Scaler(epochs.info) # Standardize channel data.
    res_accuracy = None
    
    cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    pipeline = Pipeline([
        (f'scaler', scaler),
        (f'decomposition {decomp_alg}', choosed_decomp_alg),
        (f'clf {model_name}', choosed_model)
    ], verbose=VERBOSE)
    if need_calculate_mean or VERBOSE:
        score = cross_val_score(pipeline, X, y, cv=cv, n_jobs=1, verbose=VERBOSE)

        res_accuracy = np.mean(score)
        local_print("\n")
        local_print("Cross validation scores:")
        local_print(f"Raw: {score}")
        local_print(f"Accuracy: {res_accuracy} (+/- {np.std(score)})")

    return res_accuracy, pipeline

if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-s', '--subject', type=str, help='Subject number, sequence of subjects (separated by comma) or all', required=True)
    parser.add_argument('-ps', '--pack-subj', action='store_true', help='Pack subject, to have one model by experiment', default=False, required=False)
    parser.add_argument('-e', '--experiment', type=str, help='Type training', required=False, choices=CHOICE_TRAINING, default='hands_vs_feet')
    parser.add_argument('-d', '--directory-dataset', type=str, help='Directory dataset', required=False, default='../../files')
    parser.add_argument('-m', '--model', type=str, help=f'Model name.\nAvailables models: {MODEL_NAMES_STR}', required=False, default='lda')
    parser.add_argument('-o', '--output', type=str, help='Output directory', required=False, default='./output/')
    parser.add_argument('-da', '--decomposition-algorithm', type=str, help=f'Decomposition algorithm.\nAvailable: {DECOMPOSITION_ALGORITHMS_NAMES_STR}', required=False, default='TurboCSP')
    parser.add_argument('-sv', '--save-model', action='store_true', help='Save model', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()

    model_name, choosed_model, decomp_alg, choosed_decomp_alg, subject, experiment, output, save_model, directory_dataset, pack_subj, VERBOSE = check_args(args)
    
    local_print(f'Using model: {model_name}')
    local_print(f'Using decomposition algorithm: {decomp_alg}')
    local_print(f'Using number of subject: {len(subject)}')
    local_print(f'Using experiment: {experiment}')
    local_print(f'Using output: {output}')
    local_print(f'Using verbose: {VERBOSE}')
    local_print("\n")
    
    subject_len = len(subject)
    need_calculate_mean = subject_len > 1
    results = {}
    pipeline = None
    for e in experiment:
        local_print(f'Experiment: {e}')
        experiment_results = []
        results[e] = {"mean": 0, "results": []}
        if pack_subj:
            raw = load_data_all(subject, e, directory_dataset, VERBOSE)
            epochs, event_dict, raw = get_epochs(raw)
            X, y = get_X_y(epochs)
            score, pipeline = process_model(X, y, epochs, choosed_decomp_alg, choosed_model, need_calculate_mean, VERBOSE)
            results[e]["results"].append({"subject": "all", "accuracy": score})
            experiment_results.append(score)
        else:
            for s in subject:
                raw = load_data_one(s, e, directory_dataset, VERBOSE)
                epochs, event_dict, raw = get_epochs(raw)
                X, y = get_X_y(epochs)
                score, pipeline = process_model(X, y, epochs, choosed_decomp_alg, choosed_model, need_calculate_mean, VERBOSE)
                experiment_results.append(score)
                results[e]["results"].append({"subject": s, "accuracy": score})

        if need_calculate_mean:
            mean = np.mean(experiment_results)
            std = np.std(experiment_results)
            print("\n")
            print("Cross validation scores:")
            print(f"Accuracy: {mean} (+/- {std})")
            results[e]["mean"] = mean

        if save_model and pipeline is not None:
            pipeline = pipeline.fit(X, y)
            prefix = f'_{e}.joblib' if pack_subj else f'_{e}_{subject[0]}.joblib'
            path = output + prefix
            print(f"[+] Saving model in {path}")
            joblib.dump(pipeline, path)

    if need_calculate_mean and len(results.values()) and len(list(results.values())[0]['results']) > 0:
        m = []
        print("\n")
        print(f"Experiment {'':<10}  {'':>8} mean accuracy")
        for k, v in results.items():
            m.append(v['mean'])
            print(f"{k:<30} {v['mean']:>20}")
        
        m = np.array(m)
        print(f"Mean accuracy of {len(experiment)} experiments: {np.mean(m)}")

    if VERBOSE:
        print("Result file saved in: ./results.json")
        with open(f"{output}results.json", 'w') as f:
            json.dump(results, f)