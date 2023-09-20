import mne
from mne.io import concatenate_raws, read_raw_edf
import glob

SUBJECT_AVAILABLES = range(1, 110)
SUBJECT_AVAILABLES = list(SUBJECT_AVAILABLES)
SUBJECT_AVAILABLES.remove(88)
SUBJECT_AVAILABLES.remove(92)
SUBJECT_AVAILABLES.remove(100)


EXPERIMENTS = {
    'hands_vs_feet': [3, 7, 11],
    'left_vs_right': [5, 9, 13],
    'imagery_left_vs_right': [4, 8, 12],
    'imagery_hands_vs_feet': [6, 10, 14],
}


def load_data(subject, experiment, directory_dataset, VERBOSE=False):
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
    subject = f'S{subject:03d}'

    files = glob.glob(f'{directory_dataset}/{subject}/*.edf')
    raws = []

    for i in EXPERIMENTS[experiment]:
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
    raw.filter(8, 45, fir_design='firwin', skip_by_annotation='edge', verbose=VERBOSE)

    # ICA
    ica = mne.preprocessing.ICA(n_components=10, random_state=97, max_iter=800, verbose=VERBOSE)
    ica.fit(raw, verbose=VERBOSE)

    return raw