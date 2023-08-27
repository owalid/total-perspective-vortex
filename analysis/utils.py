import mne
from mne.datasets import eegbci
import numpy as np
from mne.preprocessing import ICA

def preprocess_data(raw_obj):
    original_raw = raw_obj.copy()

    # filters
    notch_freq = 60
    original_raw.notch_filter(notch_freq, fir_design='firwin')

    low_cutoff = 8
    high_cutoff = 40
    original_raw.filter(low_cutoff, high_cutoff, fir_design='firwin')

    events, event_dict = mne.events_from_annotations(original_raw)
    print(original_raw.info)
    print(event_dict)
    picks = mne.pick_types(original_raw.info, meg=True, eeg=True, stim=False, eog=False, exclude='bads')
    eegbci.standardize(original_raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    original_raw.set_montage(montage)

    ## ICA
    n_components = 10
    ica = ICA(n_components=n_components, random_state=97, max_iter=800)
    ica.fit(original_raw)
    ica.plot_components()
    components_to_excludes, scores = ica.find_bads_eog(original_raw, ch_name='Fpz')
    if components_to_excludes is not None and len(components_to_excludes) > 0:
        ica.exclude = components_to_excludes
        original_raw = ica.apply(original_raw)
    else:
        print("No components to exclude")

    event_id = {'left_hand': 1, 'right_hand': 2}
    events, event_dict = mne.events_from_annotations(original_raw, event_id=event_id)
    tmin = -0.5  # Time before event in seconds
    tmax = 4.  # Time after event in seconds
    epochs = mne.Epochs(original_raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

    return original_raw, events, event_dict, picks, epochs