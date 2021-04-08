# %%
import os.path as op
from collections import Counter

import os
import numpy as np
import pandas as pd
import mne

from mne_bids import BIDSPath
from mne_bids import get_entity_vals, read_raw_bids


from joblib import Parallel, delayed
from autoreject import get_rejection_threshold

import library as lib
import config as cfg

username = os.environ.get('USER')
if "hashemi" in username:
    AGE_FNAME = "/datasabzi/hashemi/Cam-CAN-preprocessing/age.csv"
else:
    AGE_FNAME = "/storage/store/work/hjanati/datasets/data/camcan/age.csv"

username = os.environ.get('USER')
if "hashemi" in username:
    mne_camcan_freesurfer_path = (
        '/storage/store/data/camcan-mne/freesurfer')
else:
    mne_camcan_freesurfer_path = (
        'datasabzi/hashemi/freesurfer/subjects/')

derivative_path = './derivatives/'

if "hashemi" in username:
    BIDS_ROOT = "/datasabzi/data/CamCAN_old/cc700/mri/pipeline/release004/BIDSsep/megraw_passive"
else:
    BIDS_ROOT = "/storage/store/data/camcan/BIDSsep/passive"
    
kind = "passive"  # can be "smt"
N_JOBS = 1

task_info = {
    'passive': {
        'event_id': [{
            'Aud300Hz': 6, 'Aud600Hz': 7, 'Aud1200Hz': 8, 'Vis': 9}],
        'epochs_params': [{
            'tmin': -0.2, 'tmax': 0.7, 'baseline': (-.2, None),
            'decim': 8}],
        'lock': ['stim']
    },
    'task': {
        'event_id': [
            {'AudVis300Hz': 1, 'AudVis600Hz': 2, 'AudVis1200Hz': 3},
            {'resp': 8192}],
        'epochs_params': [
            {'tmin': -0.2, 'tmax': 0.7, 'baseline': (-.2, None),
             'decim': 8},
            {'tmin': -0.5, 'tmax': 1,
             'baseline': (.8, 1.0), 'decim': 8}],
        'lock': ['stim', 'resp'],
    }
}


def _get_global_reject_ssp(raw):
    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    if len(eog_epochs) >= 5:
        reject_eog = get_rejection_threshold(eog_epochs, decim=8)
        del reject_eog['eog']
    else:
        reject_eog = None

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    if len(ecg_epochs) >= 5:
        reject_ecg = get_rejection_threshold(ecg_epochs[:200], decim=8)
    else:
        reject_eog = None

    if reject_eog is None:
        reject_eog = reject_ecg
    if reject_ecg is None:
        reject_ecg = reject_eog
    return reject_eog, reject_ecg


def _run_maxfilter(raw):

    calibration = './Cam-CAN_sss_cal.dat'
    cross_talk = './Cam-CAN_ct_sparse.fif'

    bads, _ = mne.preprocessing.find_bad_channels_maxwell(
        raw, calibration=calibration, cross_talk=cross_talk
    )
    raw.info['bads'] = bads

    raw = lib.preprocessing.run_maxfilter(
        raw, coord_frame='head',
        calibration=calibration,
        cross_talk=cross_talk
    )
    return raw


def _compute_add_ssp_exg(raw):
    reject_eog, reject_ecg = _get_global_reject_ssp(raw)

    proj_eog = mne.preprocessing.compute_proj_eog(
        raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)

    proj_ecg = mne.preprocessing.compute_proj_ecg(
        raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)

    raw.add_proj(proj_eog[0])
    raw.add_proj(proj_ecg[0])


def _get_global_reject_epochs(raw, events, event_id, epochs_params):
    epochs = mne.Epochs(
        raw, events, event_id=event_id, proj=False,
        **epochs_params)
    epochs.load_data()
    epochs.pick_types(meg=True)
    epochs.apply_proj()
    reject = get_rejection_threshold(epochs, decim=8)
    return reject


def _compute_evoked(subject, kind):

    fname = BIDSPath(root=BIDS_ROOT, subject=subject, session=kind,
                     task=kind, datatype='meg', extension='.fif')

    raw = read_raw_bids(fname)
    mne.channels.fix_mag_coil_types(raw.info)
    raw = _run_maxfilter(raw)
    raw.filter(0.1, 45)
    _compute_add_ssp_exg(raw)

    out = {}
    for ii, event_id in enumerate(task_info[kind]['event_id']):
        epochs_params = task_info[kind]['epochs_params'][ii]
        lock = task_info[kind]['lock'][ii]
        events = mne.find_events(
            raw, uint_cast=True, min_duration=2. / raw.info['sfreq'])

        if kind == 'task' and lock == 'resp':
            event_map = np.array(
                [(k, v) for k, v in Counter(events[:, 2]).items()])
            button_press = event_map[:, 0][np.argmax(event_map[:, 1])]
            if event_map[:, 1][np.argmax(event_map[:, 1])] >= 50:
                events[events[:, 2] == button_press, 2] = 8192
            else:
                raise RuntimeError('Could not guess button press')

        reject = _get_global_reject_epochs(
            raw,
            events=events,
            event_id=event_id,
            epochs_params=epochs_params)

        epochs = mne.Epochs(
            raw, events=events, event_id=event_id, reject=reject,
            preload=True,
            **epochs_params)

        evokeds = list()
        for kk in event_id:
            evoked = epochs[kk].average()
            evoked.comment = kk
            evokeds.append(evoked)

        # tmax is 0.05 to account for the shift error of 50ms in camcan
        noise_covs = mne.compute_covariance(epochs, tmin=None, tmax=0.05,
                                            verbose=False, n_jobs=1,
                                            projs=None)

        out_path = op.join(derivative_path, subject)
        if not op.exists(out_path):
            os.makedirs(out_path)
        epo_fname = op.join(out_path,
                            '%s_%s_sensors-epo.fif' % (kind, lock))
        cov_fname = op.join(out_path,
                            '%s_%s_sensors-cov.fif' % (kind, lock))
        ave_fname = op.join(out_path,
                            '%s_%s_sensors-ave.fif' % (kind, lock))

        mne.write_evokeds(ave_fname, evokeds)
        mne.write_cov(cov_fname, noise_covs)

        epochs.save(epo_fname, overwrite=True)

        out.update({lock: (kind, epochs.average().nave)})

    return out


def _run_all(subject, kind):
    mne.utils.set_log_level('warning')
    print(subject)
    error = 'None'
    result = dict()
    # try:
    result = _compute_evoked(subject, kind)
    # except Exception as err:
    #     error = repr(err)
    #     print(error)

    # out = dict(subject=subject, kind=kind, error=error)
    # out.update(result)
    # return out

# %%

subjects = get_entity_vals(BIDS_ROOT, entity_key='subject')
subjects = subjects[:3]  # take one only
out = Parallel(n_jobs=N_JOBS)(delayed(_run_all)(subject=subject, kind=kind)
                                for subject in subjects)
out_df = pd.DataFrame(out)
out_df.to_csv(op.join(derivative_path,
                        'log_compute_evoked_%s.csv' % kind))

# %%
