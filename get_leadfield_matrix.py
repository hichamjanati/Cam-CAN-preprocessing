import os
from typing import overload
import numpy as np
import mne
from mne.datasets import sample
from joblib import Parallel, delayed

DATA_PATH = '/datasabzi/anuja/freesurfer'
SUBJECTS_DIR = DATA_PATH + '/subjects'
PATH_TO_SAVE_LEADFIELDS = '/datasabzi/anuja/leadfields'

def get_leadfield_matrix(subject, subjects_dir, src_ref, trans, raw_fname, save_dir=PATH_TO_SAVE_LEADFIELDS, save=True):
    """
    Makes forward solution and gets the leadfield matrix for the given subject.
    """
    try:
        src = mne.morph_source_spaces(src_ref, subject_to=subject, verbose=None, subjects_dir=subjects_dir)
        # conductivity = (0.3,)  # for single layer
        conductivity = (0.3, 0.006, 0.3)  # for three layers
        model = mne.make_bem_model(subject=subject, ico=4,
                                conductivity=conductivity,
                                subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                        meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                        verbose=True)
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=True)

        leadfield_matrix = fwd["sol"]["data"]

        if save:
            path = os.path.join(save_dir, subject)
            if not os.path.exists(path):
                os.makedirs(path)
            fwd_fname = os.path.join(path, "%s-fwd.fif" % subject)
            lead_fname = os.path.join(path, "%s-ldf.npy" % subject)
            mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
            np.save(lead_fname, leadfield_matrix)
        
        else:
            return fwd, leadfield_matrix

    except:
        pass

if __name__ == "__main__":
    SUBJECTS_TRAIN = ['CC120008', 'CC110033', 'CC110101', 'CC110187', 'CC110411', 'CC110606', 'CC112141', 'CC120049', 'CC120061', 'CC120120']  
    SUBJECTS_TEST = ['CC120182', 'CC120264', 'CC120309', 'CC120313', 'CC120319', 'CC120376', 'CC120469', 'CC120550', 'CC120218', 'CC120166']  
    SUBJECTS = SUBJECTS_TRAIN + SUBJECTS_TEST

    get_trans = lambda subject: '/datasabzi/anuja/trans-files/trans/sub-%s-trans.fif'%(subject)
    get_raw_fname = lambda subject: '/datasabzi/data/CamCAN_feb21/BIDSsep/passive/sub-%s/ses-passive/meg/sub-%s_ses-passive_task-passive_meg.fif'%(subject, subject)

    src_ref = mne.setup_source_space(subject="fsaverage", spacing='ico4', subjects_dir=SUBJECTS_DIR, add_dist=False)

    Parallel(n_jobs=25)(delayed(get_leadfield_matrix)(subject, SUBJECTS_DIR, src_ref, get_trans(subject), get_raw_fname(subject)) for subject in SUBJECTS)