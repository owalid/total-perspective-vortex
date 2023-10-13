import mne
from mne.io import read_raw_edf
import glob

SUBJECT_AVAILABLES = range(1, 110)
SUBJECT_AVAILABLES = list(SUBJECT_AVAILABLES)
SUBJECT_AVAILABLES.remove(88)
SUBJECT_AVAILABLES.remove(92)
SUBJECT_AVAILABLES.remove(100)

EXPERIMENTS = {
    'hands_vs_feet': {
        'experiments': [5, 9, 13],
        'events': {'T1': 1, 'T2': 2}
    },
    'left_vs_right': {
        'experiments': [3, 7, 11],
        'events': {'T1': 1, 'T2': 2}
    },
    'imagery_left_vs_right': {
        'experiments': [4, 8, 12],
        'events': {'T1': 1, 'T2': 2}
    },
    'imagery_hands_vs_feet': {
        'experiments': [6, 10, 14],
        'events': {'T1': 1, 'T2': 2}
    },
    'all': {
        'experiments': [5, 9, 13, 3, 7, 11],
        'events': {'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}
    },
}


def load_one_subject(current_raws, subject_num, directory_dataset, type_training):
    subject = f'S{subject_num:03d}'
    files = glob.glob(f'{directory_dataset}/{subject}/*.edf')
    files.sort()
    for i in EXPERIMENTS[type_training]['experiments']:
        current_file = files[i]
        r = read_raw_edf(current_file, preload=True, stim_channel='auto')
        if type_training == 'all':
            events, _ = mne.events_from_annotations(r)
            if i in [5, 9, 13]:
                new_labels_events = {1:'rest', 2:'T1', 3:'T2'} # action
            elif i in [3, 7, 11]:
                new_labels_events = {1:'rest', 2:'T3', 3:'T4'}
            new_annot = mne.annotations_from_events(events=events, event_desc=new_labels_events, sfreq=r.info['sfreq'], orig_time=r.info['meas_date'])
            r.set_annotations(new_annot)
        current_raws.append(r)
    return current_raws

def preprocess_raw(raw, type_training):
    # filters
    notch_freq = 60
    raw.notch_filter(notch_freq, fir_design='firwin')

    low_cutoff = 8
    high_cutoff = 40
    raw.filter(low_cutoff, high_cutoff, fir_design='firwin')

    events, event_dict = mne.events_from_annotations(raw)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, exclude='bads')

    event_id = EXPERIMENTS[type_training]['events']
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id)
    tmin = -0.5  # Time before event in seconds
    tmax = 4.5  # Time after event in seconds
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

    return raw, epochs