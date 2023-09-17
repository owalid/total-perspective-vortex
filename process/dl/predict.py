import os
import numpy as np
import argparse as ap
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.models import load_model
from mne.io import concatenate_raws
from utils import preprocess_raw, load_one_subject

VERBOSE = False
SUBJECT_AVAILABLES = range(1, 110)
CHOICE_TRAINING = ['all', 'hands_vs_feet', 'left_vs_right']

def check_args(args):
    subject = int(args.subject)
    model_path = args.model_path
    directory_dataset = args.directory_dataset
    VERBOSE = args.verbose
    type_training = args.type_training


    if type_training not in CHOICE_TRAINING:
        raise ValueError(f'Type training must be in {CHOICE_TRAINING}')
    if subject not in SUBJECT_AVAILABLES:
        raise ValueError(f'Invalid subject number. Availables: {SUBJECT_AVAILABLES}')
    if not os.path.exists(model_path):
        raise ValueError(f'Model path not exists: {model_path}')
    
    if not os.path.exists(directory_dataset):
        raise ValueError(f'Directory dataset not exists: {directory_dataset}')
    if not os.path.isdir(directory_dataset):
        raise ValueError(f'Directory dataset is not a directory: {directory_dataset}')
    if directory_dataset[-1] == '/':
        directory_dataset = directory_dataset[:-1]

    
    choosed_model = load_model(model_path)

    return choosed_model, subject, directory_dataset, type_training, VERBOSE


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
    parser.add_argument('-s', '--subject', type=int, help='Subject number', required=True)
    parser.add_argument('-m', '--model-path', type=str, help=f'Model path', required=False, default='./output_model/model.h5')
    parser.add_argument('-d', '--directory-dataset', type=str, help='Directory dataset', required=False, default='../../files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()

    choosed_model, subject, directory_dataset, type_training, VERBOSE = check_args(args)

    raw = load_data(subject, directory_dataset, type_training)
    raw, events, event_dict, picks, epochs = preprocess_raw(raw, type_training)
    X = epochs.get_data()
    n_channels = X.shape[1]
    input_window_size = X.shape[2]
    input_shape = (1, n_channels, input_window_size)
    X = X.reshape(X.shape[0], 1, n_channels, input_window_size)
    X = normalize(X, axis=1, order=0)
    y = epochs.events[:, -1] - 2
    y = to_categorical(y)
    print(f"y = {y}")
    if VERBOSE:
        print(f'Predicting...')
    y_pred = choosed_model.predict(X)


    # TODO MAKE FUNCTION TO PARSE PROBABILITIES OF PREDICTIONS
    print(f'Predictions: {y_pred}')
    predict_correct = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
    good_prediction = 0
    for i in range(len(predict_correct)):
        if predict_correct[i].all():
            good_prediction += 1
    print(f'Accuracy: {good_prediction/len(predict_correct)}')
    if VERBOSE:
        print(f'Good predictions: {good_prediction} / {len(predict_correct)}')
        print(f'Bad predictions: {len(predict_correct) - good_prediction}/ {len(predict_correct)}')
        print(f'Predictions: {predict_correct}')
        