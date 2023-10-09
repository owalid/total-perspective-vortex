import numpy as np
import mne
from sklearn.decomposition import FastICA

def get_epochs_erd_ers_mrcp(raw ):
    event_id = {'T1': 1, 'T2': 2}
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id)

    # Epoch your data for different analyses
    tmin_erd = -2
    tmax_erd = 0
    tmin_ers = 4.1
    tmax_ers = 5.1
    tmin_mrcp = -2
    tmax_mrcp = 0

    raw_obj_erd = raw.copy()
    raw_obj_erd.filter(8, 30, fir_design='firwin')
    raw_obj_ers = raw.copy()
    raw_obj_ers.filter(8, 30, fir_design='firwin')
    raw_obj_mrcp = raw.copy()
    raw_obj_mrcp.filter(None, 3, fir_design='firwin')

    erd_epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin_erd, tmax=tmax_erd, baseline=None)
    ers_epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin_ers, tmax=tmax_ers, baseline=None)
    mrcp_epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin_mrcp, tmax=tmax_mrcp, baseline=None)

    return erd_epochs, ers_epochs, mrcp_epochs

def get_features_from_epoch_one(epoch_data, ica):
    current_data = epoch_data
    activation_vector = ica.fit(current_data)
    activation_vector = activation_vector.components_
    mean_current_data = np.mean(current_data, axis=0)
    activation_vector -= mean_current_data
    mean_activation_vector = np.mean(activation_vector, axis=1)
    std_activation_vector = np.std(activation_vector, axis=1)
    power_activation_vector = np.mean(np.abs(activation_vector) ** 2, axis=1)
    energy_activation_vector = np.sum(np.abs(activation_vector) ** 2, axis=1)
    variance_activation_vector = np.var(activation_vector, axis=1)
    rms_activation_vector = np.sqrt(np.mean(activation_vector ** 2, axis=1))

    epoch_features = np.hstack((mean_activation_vector, std_activation_vector, energy_activation_vector, power_activation_vector, variance_activation_vector, rms_activation_vector))
    return epoch_features


def get_X_y_mu_beta(raw):
    erd_epochs, ers_epochs, mrcp_epochs = get_epochs_erd_ers_mrcp(raw)
    n_components = None
    feature_matrix = []
    
    ica = FastICA(n_components=n_components, random_state=97, max_iter=800)

    for ii in range(erd_epochs.get_data().shape[0]):
        current_data_erd = erd_epochs.get_data()[ii]
        features_erd = get_features_from_epoch_one(current_data_erd, ica)
        feature_matrix.append(features_erd)

    yy_erd = erd_epochs.events[:, -1] - 1

    for ii in range(ers_epochs.get_data().shape[0]):
        current_data_ers = ers_epochs.get_data()[ii]
        features_ers = get_features_from_epoch_one(current_data_ers, ica)
        feature_matrix.append(features_ers)
    yy_ers = ers_epochs.events[:, -1] - 1

    for ii in range(mrcp_epochs.get_data().shape[0]):
        current_data_mrcp = mrcp_epochs.get_data()[ii]
        features_mrcp = get_features_from_epoch_one(current_data_mrcp, ica)
        feature_matrix.append(features_mrcp)
    yy_mrcp = mrcp_epochs.events[:, -1] - 1

    y = np.hstack((yy_erd, yy_ers, yy_mrcp))
    
    feature_matrix = np.array(feature_matrix)
    
    return feature_matrix, y, None