import mne
import sys
import json
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import glob


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import numpy as np
from mne.preprocessing import ICA

from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    sys.stdout = open('/dev/null', 'w')
    sys.stderr = open('/dev/null', 'w')

    files = glob.glob('../../files/S001/*.edf')
    '''
    https://github.com/mne-tools/mne-python/blob/main/mne/datasets/eegbci/eegbci.py#L110
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

    candidates = json.load(open('candidates.json', 'r'))
    raws = []
    f = [5,9,13]
    # ,6,10,14]
    for i in f:
        print(i)
        current_file = files[i-1]
        r = read_raw_edf(current_file, preload=True, stim_channel='auto')
        events, _ = mne.events_from_annotations(r)
        if i in [5, 9, 13]:
            new_labels_events = {1:'rest', 2:'action_hand', 3:'action_feet'} # action
        else:
            new_labels_events = {1:'rest', 2:'imagine_hand', 3:'imagine_feet'} # imagine
        new_annot = mne.annotations_from_events(events=events, event_desc=new_labels_events, sfreq=r.info['sfreq'], orig_time=r.info['meas_date'])
        r.set_annotations(new_annot)
        raws.append(r)
        
    raw_obj = concatenate_raws(raws)
    final_results = []

    for candidate in candidates:
        original_raw = raw_obj.copy()

        if candidate['notch']:
            # filters
            notch_freq = 60
            original_raw.notch_filter(notch_freq, fir_design='firwin')

        low_cutoff = candidate['low_cutoff']
        high_cutoff = candidate['high_cutoff']
        original_raw.filter(low_cutoff, high_cutoff, fir_design='firwin')


        events, event_dict = mne.events_from_annotations(original_raw)
        picks = mne.pick_types(original_raw.info, meg=True, eeg=True, stim=False, eog=False, exclude='bads')
        eegbci.standardize(original_raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        original_raw.set_montage(montage)

        ## ICA
        if candidate['ica']:
            n_components = 10
            ica = ICA(n_components=n_components, random_state=97, max_iter=800)
            ica.fit(original_raw)
            # ica.plot_components()
            components_to_excludes, scores = ica.find_bads_eog(original_raw, ch_name='Fpz')
            # if components_to_excludes is not None and len(components_to_excludes) > 0:
            #     ica.plot_properties(original_raw, picks=components_to_excludes)
            # else:
            #     print("No components to exclude")
            # original_raw.compute_psd().plot()

        event_id = {'action_hand': 1, 'action_feet': 2}
        events, event_dict = mne.events_from_annotations(original_raw, event_id=event_id)

        tmin = candidate['tmin']
        tmax = candidate['tmax']
        epochs = mne.Epochs(original_raw, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
        
        # _, event_dict = mne.events_from_annotations(original_raw)

        X = epochs.get_data()
        y = epochs.events[:, -1] - 1

        n_splits = 5  # Number of shuffle-split iterations
        test_size = 0.2  # Proportion of data to be used as the test set in each iteration
        shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

        models = [
            ('Gradient Boosting', GradientBoostingClassifier(), {'model__n_estimators': [50, 100]}),
            ('Linear discriminant analysis', LinearDiscriminantAnalysis(), {'model__solver': ['svd']}),
            ('SVM', SVC(), {'model__C': [0.5, 1, 3], 'model__kernel': ['linear']}),
            ('KNN', KNeighborsClassifier(), {'model__n_neighbors': [4,5,6]}),
            ('Random Forest', RandomForestClassifier(), {'model__n_estimators': [50,100]}),
            ('MLP', MLPClassifier(), {'model__hidden_layer_sizes': [(100, 50), (200, 100)]}),
            ('Decision Tree', DecisionTreeClassifier(), {'model__max_depth': [50, 100]}),
            ('XGB', XGBClassifier(), {'model__n_estimators': [200, 300], 'model__learning_rate': [0.05]})
        ]

        pipelines = []
        csp = CSP()
        for name, model, param_grid in models:
            pipeline = Pipeline([
                ('csp', csp),
                ('model', model)
            ])
            param_grid['csp__n_components'] = [5, 6, 7, 8, 9, 10, 15, 20, 50]
            pipelines.append((name, pipeline, param_grid))

        results = []
        for name, pipeline, param_grid in pipelines:
            grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=shuffle_split, scoring='accuracy', n_jobs=1)
            grid_search.fit(X, y)
            results.append((name, grid_search))
        
        res_grid = []
        for name, grid_search in results:
            res_grid.append({'name': name, 'best_params_': grid_search.best_params_, 'best_score_': grid_search.best_score_})
        candidate_id = candidate['id']
        with open(f'./results/{candidate_id}.json', 'w+') as f:
            json.dump(res_grid, f)
    final_results.append(res_grid)
    final_results[-1].insert(0, {'candidate': candidate})    
    # final_results = []

    # for candidate in candidates:
    #     candidate_id = candidate['id']
    #     with open(f'./results/{candidate_id}.json', 'r') as f:
    #         res_grid = json.load(f)
    #     final_results.append(res_grid)
    #     final_results[-1].insert(0, {'candidate': candidate})    

    with open(f'./results/final_results.json', 'w+') as f:
        json.dump(final_results, f, indent=4)