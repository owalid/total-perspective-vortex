import mne
from mne.io import concatenate_raws, read_raw_edf
import glob

def load_data(subject, experiment, VERBOSE=False):
    '''
    =========  ===================================
    run        task
    =========  ===================================
    1          Baseline, eyes open
    2          Baseline, eyes closed
    3, 7, 11   Motor execution: left vs right hand
    4, 8, 12   Motor imagery: left vs right hand
    5, 9, 13   Motor execution: hands vs feet
    6, 10, 14  Motor imagery: hands vs feet
    =========  ===================================
    '''
    experiments = [
        [1],
        [2],
        [3, 7, 11],
        [4, 8, 12],
        [5, 9, 13],
        [6, 10, 14]
    ]
    subject = f'S{subject:03d}'
    
    current_experiment = [e for i, e in enumerate(experiments) if experiment in e][0]
    files = glob.glob(f'./files/{subject}/*.edf')
    raws = []
    for i in current_experiment:
        current_file = files[i-1]
        r = read_raw_edf(current_file, preload=True, stim_channel='auto', verbose=VERBOSE)
        raws.append(r)
    raw = concatenate_raws(raws)
    raw = filter_data(raw, VERBOSE)
    return raw


def filter_data(raw, VERBOSE=False):
    # notch filter for power line noise
    notch_freq = 60
    raw.notch_filter(notch_freq, fir_design='firwin', verbose=VERBOSE)

    # band pass filter
    raw.filter(8, 40, fir_design='firwin', skip_by_annotation='edge', verbose=VERBOSE)

    # ICA
    ica = mne.preprocessing.ICA(n_components=10, random_state=97, max_iter=800, verbose=VERBOSE)
    ica.fit(raw, verbose=VERBOSE)

    return raw