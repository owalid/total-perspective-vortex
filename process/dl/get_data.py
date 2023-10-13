import numpy as np
from tensorflow.keras.utils import to_categorical, normalize
from mne.io import concatenate_raws
from sklearn.model_selection import train_test_split

from utils import preprocess_raw, load_one_subject, SUBJECT_AVAILABLES


# option data order
def get_train_test_data_order(nums_subjects, directory_dataset, test_size, type_training, VERBOSE):
    raws_train = []
    raws_test = []
    if VERBOSE:
        print(f'Loading data...')
    
    # get nums_subjects first inside SUBJECT_AVAILABLES array
    nums_subjects_elms = SUBJECT_AVAILABLES[:nums_subjects]
    for ii in nums_subjects_elms:
        if ii >= nums_subjects*test_size:
            raws_train = load_one_subject(raws_train, ii, directory_dataset, type_training)
        else:
            raws_test = load_one_subject(raws_test, ii, directory_dataset, type_training)

    if VERBOSE:
        print(f"Len raws train: {len(raws_train)}")
        print(f"Len raws test: {len(raws_test)}")
    raw_train = concatenate_raws(raws_train)
    raw_test = concatenate_raws(raws_test)

    raw_train, epochs_train = preprocess_raw(raw_train, type_training)
    raw_test, epochs_test = preprocess_raw(raw_test, type_training)
    X_train, y_train, X_test, y_test = get_X_y_order(epochs_train, epochs_test, type_training, VERBOSE)
    return X_train, y_train, X_test, y_test

def get_X_y_order(epochs_train, epochs_test, type_training, VERBOSE):
    min_one = 2 if type_training == 'all' else 1
    X_train = epochs_train.get_data()
    y_train = epochs_train.events[:, -1] - min_one
    y_train = to_categorical(y_train)

    X_test = epochs_test.get_data()
    y_test = epochs_test.events[:, -1] - min_one
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test
# end option data order

# option data random
def get_X_y_random(epochs, type_training, test_size, VERBOSE):
    min_one = 2 if type_training == 'all' else 1
    X = epochs.get_data()
    y = epochs.events[:, -1] - min_one
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test

def get_train_test_data_random(nums_subjects, directory_dataset, test_size, type_training, VERBOSE):
    raws = []

    if VERBOSE:
        print(f'Loading data...')
    
    nums_subjects_elms = SUBJECT_AVAILABLES[:nums_subjects]
    for ii in nums_subjects_elms:
        raws = load_one_subject(raws, ii, directory_dataset, type_training)
    
    raws = concatenate_raws(raws)
    raws, epochs = preprocess_raw(raws, type_training)
    X_train, y_train, X_test, y_test = get_X_y_random(epochs, type_training, test_size, VERBOSE)
    return X_train, y_train, X_test, y_test
# end option data random

def get_inputs_layers(X_train, y_train, X_test, model_name, VERBOSE):
    n_classes = len(np.unique(y_train))
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

    return X_train, X_test, input_shape, n_channels, input_window_size, n_classes
