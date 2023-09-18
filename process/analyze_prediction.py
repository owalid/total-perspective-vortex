import os
import json
import matplotlib.pyplot as plt
import argparse as ap

def check_data(data):
    if not isinstance(data, list):
        raise ValueError('Data must be a list')

    if len(data) == 0:
        raise ValueError('Data is empty')
    
    if not isinstance(data[0], dict):
        raise ValueError('Data must be a list of dict')
    
    if 'subject' not in data[0]:
        raise ValueError('Data must have subject key')
    
    if 'accuracy' not in data[0]:
        raise ValueError('Data must have accuracy key')
    

def plot_from_json_data(data):
    check_data(data)

    # sort data by accuracy
    data = sorted(data, key=lambda d: d['subject'], reverse=True)

    subjects = [d['subject'] for d in data] 
    accuracies = [d['accuracy'] for d in data]  
    
    # print mean accuracy for all subjects
    print(f'Mean accuracy: {sum(accuracies) / len(accuracies)}')
    plt.ylim(0, 1) 
    plt.ylim(top=1.1)
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.bar(subjects, accuracies)
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.show()

    plt.ylim(0, 1) 
    plt.ylim(top=1.1)
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.scatter(subjects, accuracies)
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-i', '--input_file', type=str, help='Input file with results in json format', required=True)
    args = parser.parse_args()

    input_file = args.input_file

    if not os.path.exists(input_file):
        raise ValueError(f'Input file not exists: {input_file}')
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    plot_from_json_data(data)