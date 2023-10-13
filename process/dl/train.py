import os
import argparse as ap

from models import cnn2d_classic, gcn_classic, cnn2d_advanced
from get_data import *
from utils import SUBJECT_AVAILABLES


CHOICE_TRAINING = ['all', 'hands_vs_feet', 'left_vs_right', 'imagery_hands_vs_feet', 'imagery_left_vs_right']

MODEL_LIST = [
    ('cnn2d_classic', cnn2d_classic),
    ('gcn_classic', gcn_classic),
    ('cnn2d_advanced', cnn2d_advanced),
]
MODEL_NAMES = [name for name, _ in MODEL_LIST]
MODEL_NAMES_STR = ', '.join(MODEL_NAMES)
VERBOSE = False


def check_args(args):
    nums_subjects = args.nums_subjects
    model_name = args.model
    output = args.output
    verbose = args.verbose
    save_model = not args.save_model
    batch_size = args.batch_size
    epochs = args.epochs
    directory_dataset = args.directory_dataset
    type_training = args.type_training
    order = args.order
    test_size = args.test_size

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
    if test_size <= 0 or test_size >= 1:
        raise ValueError(f'test_size must be between 0 and 1')
    if test_size >= 0.5:
        print(f'Warning: test_size is greater than 0.5, this can cause overfitting')
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
    
    return nums_subjects, test_size, model_name, model, output, verbose, save_model, batch_size, epochs, directory_dataset, type_training, order



if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-ns', '--nums-subjects', type=str, help='Numbers of subjects', required=True)
    parser.add_argument('-t', '--type-training', type=str, help='Type training', required=False, choices=CHOICE_TRAINING, default='hands_vs_feet')
    parser.add_argument('-ts', '--test-size', type=float, help='Test size (default 0.3)', required=False, default=0.3)
    parser.add_argument('-e', '--epochs', type=int, help='Epochs', default=15, required=False)
    parser.add_argument('-bs', '--batch-size', type=int, help='Batch size', default=10, required=False)
    parser.add_argument('-d', '--directory-dataset', type=str, help='Directory dataset', required=False, default='../../files')
    parser.add_argument('-m', '--model', type=str, help=f'Model name.\nAvailables models: {MODEL_NAMES_STR}', required=False, default='gcn_classic')
    parser.add_argument('-o', '--output', type=str, help='Output path file', required=False, default='output_model/model.h5')
    parser.add_argument('-sv', '--save-model', action='store_true', help='Save model', default=False)
    parser.add_argument('-ord', '--order', action='store_true', help='Keep order subjects and not shuffle epochs', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)
    args = parser.parse_args()

    nums_subjects, test_size, model_name, model, output, VERBOSE, save_model, batch_size, epochs, directory_dataset, type_training, order = check_args(args)

    if order:
        X_train, y_train, X_test, y_test = get_train_test_data_order(nums_subjects, directory_dataset, test_size, type_training, VERBOSE)
    else:
        X_train, y_train, X_test, y_test = get_train_test_data_random(nums_subjects, directory_dataset, test_size, type_training, VERBOSE)

    X_train, X_test, input_shape, n_channels, input_window_size, n_classes = get_inputs_layers(X_train, y_train, X_test, model_name, VERBOSE)
    print( input_shape, n_channels, input_window_size, n_classes)
    model = model(input_shape, n_channels, input_window_size, n_classes)
    loss = 'binary_crossentropy' if n_classes == 1 else 'categorical_crossentropy'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),  shuffle=False, verbose=1 if VERBOSE else 0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1 if VERBOSE else 0)

    if VERBOSE:
        print(f'Accuracy: {accuracy}')
        print(f'Loss: {loss}')

    if save_model:
        model.save(output)

