from mne.preprocessing import ICA
from mne.io import concatenate_raws, read_raw_edf
import glob

SUBJECT_AVAILABLES = range(1, 110)
SUBJECT_AVAILABLES = list(SUBJECT_AVAILABLES)
SUBJECT_AVAILABLES.remove(88)
SUBJECT_AVAILABLES.remove(92)
SUBJECT_AVAILABLES.remove(100)


EXPERIMENTS = {
    'hands_vs_feet': [5, 9, 13],
    'left_vs_right': [3, 7, 11],
    'imagery_left_vs_right': [4, 8, 12],
    'imagery_hands_vs_feet': [6, 10, 14],
}


def load_data(subjects, experiment, directory_dataset, VERBOSE=False):
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
    raws = []
    for subject in subjects:
        subject = f'S{subject:03d}'

        files = glob.glob(f'{directory_dataset}/{subject}/*.edf')
        files.sort()
        for i in EXPERIMENTS[experiment]:
            current_file = files[i-1]
            r = read_raw_edf(current_file, preload=True, stim_channel='auto', verbose=VERBOSE)
            raws.append(r)

    raw = concatenate_raws(raws)
    raw = filter_data(raw, VERBOSE)
    return raw

def load_data_all(subjects, experiment, directory_dataset, VERBOSE=False):
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
    raws = []
    for subject in subjects:
        subject = f'S{subject:03d}'

        files = glob.glob(f'{directory_dataset}/{subject}/*.edf')
        files.sort()
        for i in EXPERIMENTS[experiment]:
            current_file = files[i-1]
            r = read_raw_edf(current_file, preload=True, stim_channel='auto', verbose=VERBOSE)
            raws.append(r)

    raw = concatenate_raws(raws)
    raw = filter_data(raw, VERBOSE)
    return raw

def load_data_one(subject, experiment, directory_dataset, VERBOSE=False):
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
    raws = []

    subject = f'S{subject:03d}'

    files = glob.glob(f'{directory_dataset}/{subject}/*.edf')
    files.sort()

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
    raw.filter(7, 30, fir_design='firwin', skip_by_annotation='edge', verbose=VERBOSE)


    # # ICA
    # ica = ICA(n_components=20, random_state=97, max_iter=500, verbose=VERBOSE)
    # ica.fit(raw, verbose=VERBOSE)

    # components_to_excludes, _ = ica.find_bads_eog(raw, ch_name='Fpz.')

    # if components_to_excludes is not None and len(components_to_excludes) > 0:
    #     ica.exclude = components_to_excludes
    #     raw = ica.apply(raw)

    return raw