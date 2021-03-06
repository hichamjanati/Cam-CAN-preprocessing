import os

import numpy as np

import mne
from mne.parallel import parallel_func

import config as cfg


def compute_fwd(subject, src_ref, info, trans_fname, bem_fname,
                meg=True, eeg=True, mindist=3, subjects_dir=None,
                n_jobs=1, verbose=None):
    src = mne.morph_source_spaces(src_ref, subject_to=subject,
                                  verbose=verbose,
                                  subjects_dir=subjects_dir)
    bem = mne.read_bem_solution(bem_fname, verbose=verbose)
    fwd = mne.make_forward_solution(info, trans=trans_fname, src=src,
                                    bem=bem, meg=meg, eeg=eeg,
                                    mindist=mindist, verbose=verbose,
                                    n_jobs=n_jobs)
    return fwd


dataset_name = "camcan"
delete_all = False
save_dir = "/storage/store/work/hjanati/datasets/%s/leadfields" \
    % dataset_name
subfolders = ["ico"]

age_max = 30
n_subjects = 20
subjects_dir = cfg.get_subjects_dir(dataset_name)
subjects = cfg.get_subjects_list(dataset_name, 0, age_max)[:n_subjects]
os.environ['SUBJECTS_DIR'] = subjects_dir

trans_fnames = [cfg.get_trans_fname(dataset_name, subject)
                for subject in subjects]
raw_fnames = [cfg.get_raw_fname(dataset_name, subject)
              for subject in subjects]

bem_fnames = [cfg.get_bem_fname(dataset_name, subject)
              for subject in subjects]
resolution = 4
spacing = "ico%d" % resolution

src_ref = mne.setup_source_space(subject="fsaverage",
                                 spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=False)

n_jobs = n_subjects
parallel, run_func, _ = parallel_func(compute_fwd, n_jobs=n_jobs)


fwds = parallel(run_func(s, src_ref, raw, trans, bem)
                for s, trans, raw, bem in zip(subjects, trans_fnames,
                                              raw_fnames, bem_fnames))

for sub, fwd in zip(subjects, fwds):
    fwd = mne.convert_forward_solution(fwd, surf_ori=True,
                                       force_fixed=True,
                                       use_cps=True,
                                       verbose=False)
    leadfield_matrix = fwd["sol"]["data"]
    fwd_fname = os.path.join(save_dir, "%s-fwd.fif" % sub)
    lead_fname = os.path.join(save_dir, "%s-ldf.npy" % sub)
    mne.write_forward_solution(fwd_fname, fwd)
    np.save(lead_fname, leadfield_matrix)
