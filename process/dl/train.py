import os
import numpy as np
import argparse as ap
from tensorflow.keras.utils import to_categorical, normalize
from mne.io import concatenate_raws

from models import cnn2d_classic, gcn_classic, cnn2d_advanced
from utils import preprocess_raw, load_one_subject, SUBJECT_AVAILABLES


CHOICE_TRAINING = ['all', 'hands_vs_feet', 'left_vs_right', 'imagery_hands_vs_feet', 'imagery_left_vs_right']

MODEL_LIST = [
    ('cnn2d_classic', cnn2d_classic),
    ('gcn_classic', gcn_classic),
    ('cnn2d_advanced', cnn2d_advanced),
]
MODEL_NAMES = [name for name, _ in MODEL_LIST]
MODEL_NAMES_STR = ', '.join(MODEL_NAMES)
VERBOSE = False

def get_train_test_data(nums_subjects, directory_dataset, ratio, type_training):
    raws_train = []
    raws_test = []
    if VERBOSE:
        print(f'Loading data...')
    
    # get nums_subjects first inside SUBJECT_AVAILABLES array
    nums_subjects_elms = SUBJECT_AVAILABLES[:nums_subjects]
    for ii in nums_subjects_elms:
        if ii <= nums_subjects*ratio:
            raws_train = load_one_subject(raws_train, ii, directory_dataset, type_training)
        else:
            raws_test = load_one_subject(raws_test, ii, directory_dataset, type_training)

    if VERBOSE:
        print(f"Len raws train: {len(raws_train)}")
        print(f"Len raws test: {len(raws_test)}")
    raw_train = concatenate_raws(raws_train)
    raw_test = concatenate_raws(raws_test)

    return raw_train, raw_test


def get_X_y(epochs_train, epochs_test, model_name, type_training):
    min_one = 2 if type_training == 'all' else 1
    X_train = epochs_train.get_data()
    y_train = epochs_train.events[:, -1] - min_one
    n_classes = len(np.unique(y_train))
    # n_classes = 1 if n_classes == 2 else n_classes
    y_train = to_categorical(y_train)

    X_test = epochs_test.get_data()
    y_test = epochs_test.events[:, -1] - min_one
    y_test = to_categorical(y_test)

    _, n_channels, input_window_size = X_train.shape

    if model_name == 'cnn2d_advanced':
        input_shape = (input_window_size, n_channels, 1)
        X_train = X_train.transpose(0,2,1)
        X_train = X_train.reshape(X_train.shape[0], input_window_size, n_channels, 1)
        X_train = normalize(X_train, axis=1, order=0)

        X_test = X_test.transpose(0,2,1)
        X_test = X_test.reshape(X_test.shape[0], input_window_size, n_channels, 1)
        X_test = normalize(X_test, axis=1, order=0)
    else:
        input_shape = (1, n_channels, input_window_size)
        X_train = X_train.reshape(X_train.shape[0], 1, n_channels, input_window_size)
        X_train = normalize(X_train, axis=1, order=0)
        X_test = X_test.reshape(X_test.shape[0], 1, n_channels, input_window_size)
        X_test = normalize(X_test, axis=1, order=0)

    return X_train, y_train, X_test, y_test, input_shape, n_channels, input_window_size, n_classes



def check_args(args):
    nums_subjects = args.nums_subjects
    ratio = args.ratio
    model_name = args.model
    output = args.output
    verbose = args.verbose
    save_model = not args.save_model
    batch_size = args.batch_size
    epochs = args.epochs
    directory_dataset = args.directory_dataset
    type_training = args.type_training

    if nums_subjects == 'all':
        nums_subjects = len(SUBJECT_AVAILABLES)

    nums_subjects = int(nums_subjects)

    if nums_subjects < 3:
        raise ValueError(f'Number of subjects must be greater than 1')

    if type_training not in CHOICE_TRAINING:
        raise ValueError(f'Type training must be in {CHOICE_TRAINING}')
    
    if model_name not in MODEL_NAMES:
        raise ValueError(f'Model name must be in {MODEL_NAMES_STR}')
    if nums_subjects < 1:
        raise ValueError(f'Number of subjects must be greater than 1')
    if nums_subjects > 110:
        raise ValueError(f'Number of subjects must be less than 110')
    if ratio <= 0 or ratio >= 1:
        raise ValueError(f'Ratio must be between 0 and 1')
    if not os.path.exists(directory_dataset):
        raise ValueError(f'Directory dataset not exists: {directory_dataset}')
    if not os.path.isdir(directory_dataset):
        raise ValueError(f'Directory dataset is not a directory: {directory_dataset}')
    if directory_dataset[-1] == '/':
        directory_dataset = directory_dataset[:-1]

    dir_output = os.path.dirname(output)
    if not os.path.exists(dir_output) and dir_output != '':
        raise ValueError(f'Output directory path not valid: {output}')

    try:
        model = [model for name, model in MODEL_LIST if name == model_name][0]
    except:
        raise ValueError(f'Model name must be in {MODEL_NAMES_STR}')
    
    return nums_subjects, ratio, model_name, model, output, verbose, save_model, batch_size, epochs, directory_dataset, type_training



if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-ns', '--nums-subjects', type=str, help='Numbers of subjects', required=True)
    parser.add_argument('-t', '--type-training', type=str, help='Type training', required=False, choices=CHOICE_TRAINING, default='all')
    parser.add_argument('-r', '--ratio', type=float, help='Percent of train', required=False, default=0.8)
    parser.add_argument('-e', '--epochs', type=int, help='Epochs', default=15, required=False)
    parser.add_argument('-bs', '--batch-size', type=int, help='Batch size', default=10, required=False)
    parser.add_argument('-d', '--directory-dataset', type=str, help='Directory dataset', required=False, default='../../files')
    parser.add_argument('-m', '--model', type=str, help=f'Model name.\nAvailables models: {MODEL_NAMES_STR}', required=False, default='gcn_classic')
    parser.add_argument('-o', '--output', type=str, help='Output path file', required=False, default='output_model/model.h5')
    parser.add_argument('-sv', '--save-model', action='store_true', help='Save model', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()

    nums_subjects, ratio, model_name, model, output, VERBOSE, save_model, batch_size, epochs, directory_dataset, type_training = check_args(args)
    raw_train, raw_test = get_train_test_data(nums_subjects, directory_dataset, ratio, type_training)
    raw_train, events_train, event_dict_train, picks_train, epochs_train = preprocess_raw(raw_train, type_training)
    raw_test, events_test, event_dict_test, picks_test, epochs_test = preprocess_raw(raw_test, type_training)

    X_train, y_train, X_test, y_test, input_shape, n_channels, input_window_size, n_classes = get_X_y(epochs_train, epochs_test, model_name, type_training)

    model = model(input_shape, n_channels, input_window_size, n_classes)
    loss = 'binary_crossentropy' if n_classes == 1 else 'categorical_crossentropy'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),  shuffle=False, verbose=1 if VERBOSE else 0)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=1 if VERBOSE else 0)

    if VERBOSE:
        print(f'Accuracy: {accuracy}')
        print(f'Loss: {loss}')

    if save_model:
        model.save(output)

