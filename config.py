import os
import os.path as op
import pickle
import pandas as pd
import mne

exclude_subjects = dict(camcan=["CC220352"],
                        ds117=["sub001", "sub005", "sub016"])


def get_params(dataset):
    if os.path.exists("/home/parietal/"):
        subjects_dir = get_subjects_dir(dataset)
        data_path = "~/data/%s/" % dataset
        data_path = op.expanduser(data_path)
        subject = get_subjects_list(dataset)[0]
        info_fname = get_raw_fname(dataset, subject)
    else:
        data_path = "~/Dropbox/neuro_transport/code/"
        data_path += "mtw_experiments/meg/%s/" % dataset
        data_path = op.expanduser(data_path)
        subjects_dir = data_path + "subjects/"
        info_fname = "/Users/hichamjanati/Documents/work/mne-python/mne/"
        info_fname += "io/tests/data/test_raw.fif"
    info = mne.io.read_info(info_fname, verbose=False)
    meg_ind = mne.pick_types(info, eeg=False)
    info = mne.pick_info(info, meg_ind)
    grad_ind = mne.pick_types(info, meg="grad")

    params = dict(data_path=data_path, grad_indices=grad_ind, info=info,
                  subjects_dir=subjects_dir)
    return params


def get_subjects_dir(dataset):
    if os.path.exists("/home/parietal/"):
        if dataset == "camcan":
            subjects_dir = "/storage/store/data/camcan-mne/freesurfer/"
        elif dataset == "ds117":
            subjects_dir = "/storage/store/work/agramfort/mne-biomag-group-demo/"
            subjects_dir += "subjects/"
        else:
            raise ValueError("Unknown dataset %s." % dataset)
    else:
        data_path = "~/Dropbox/neuro_transport/code/"
        data_path += "mtw_experiments/meg/%s/" % dataset
        data_path = op.expanduser(data_path)
        subjects_dir = data_path + "subjects/"
    return subjects_dir


def get_trans_fname(dataset_name, subject):
    if dataset_name == "camcan":
        path = "/storage/store/data/camcan-mne/trans/"
        path += "sub-%s-trans.fif" % subject
    elif dataset_name == "ds117":
        path = "/storage/store/work/agramfort/mne-biomag-group-demo/"
        path += "ds117/%s/MEG/%s-trans.fif" % (subject, subject)
    else:
        raise ValueError("Unknown dataset %s." % dataset_name)

    return path


def get_bem_fname(dataset_name, subject):
    if dataset_name == "camcan":
        path = "/storage/store/data/camcan-mne/freesurfer/"
        path += "%s/bem/%s-meg-bem.fif" % (subject, subject)
    elif dataset_name == "ds117":
        path = "/storage/store/work/agramfort/mne-biomag-group-demo/"
        path += "subjects/%s/bem/%s-5120-bem-sol.fif" % (subject, subject)
    else:
        raise ValueError("Unknown dataset %s." % dataset_name)
    return path


def get_raw_fname(dataset_name, subject, task_type="passive"):
    if dataset_name == "camcan":
        path = "/storage/store/data/camcan/camcan47/cc700/meg/pipeline/"
        path += "release004/data/aamod_meg_get_fif_00001/%s/%s/" % (subject,
                                                                    task_type)
        path += "%s_raw.fif" % task_type
    elif dataset_name == "ds117":
        path = "/storage/store/work/agramfort/mne-biomag-group-demo/"
        path += "ds117/%s/MEG/run_01_raw.fif" % subject
    else:
        raise ValueError("Unknown dataset %s." % dataset_name)
    return path


def get_ave_fname(dataset_name, subject, task_type="passive"):
    if dataset_name == "camcan":
        path = "/storage/store/work/hjanati/datasets/data/camcan/meg/"
        path += "%s/%s_stim_sensors-ave.fif" % (subject, task_type)
    elif dataset_name == "ds117":
        path = "/storage/store/work/agramfort/mne-biomag-group-demo/"
        path += "MEG/%s/%s_highpass-NoneHz-ave.fif" % (subject, subject)
    else:
        raise ValueError("Unknown dataset %s." % dataset_name)
    return path


def get_cov_fname(dataset_name, subject, task_type="passive"):
    if dataset_name == "camcan":
        path = "/storage/store/work/hjanati/datasets/data/camcan/meg/"
        path += "%s/%s_stim_sensors-cov.fif" % (subject, task_type)
    elif dataset_name == "ds117":
        path = "/storage/store/work/agramfort/mne-biomag-group-demo/"
        path += "MEG/%s/%s_highpass-NoneHz-cov.fif" % (subject, subject)
    else:
        raise ValueError("Unknown dataset %s." % dataset_name)
    return path


def get_epo_fname(dataset_name, subject, task_type="passive"):
    if dataset_name == "camcan":
        path = "/storage/store/work/hjanati/datasets/data/camcan/meg/"
        path += "%s/%s_stim_sensors-epo.fif" % (subject, task_type)
    elif dataset_name == "ds117":
        path = "/storage/store/work/agramfort/mne-biomag-group-demo/"
        path += "MEG/%s/%s_highpass-NoneHz-epo.fif" % (subject, subject)
    else:
        raise ValueError("Unknown dataset %s." % dataset_name)
    return path


def get_fwd_fname(dataset_name, subject, resolution="ico4"):
    path = "/storage/store/work/hjanati/datasets/data/%s/bem/" % dataset_name
    path += "%s-%s-fwd.fif" % (subject, resolution)
    return path


def get_subjects_list(dataset_name, age_min=0, age_max=100, raw_only=False,
                      ave_only=False):
    if os.path.exists("/home/parietal/"):
        if dataset_name == "camcan":
            df = pd.read_csv("/storage/store/work/hjanati/datasets/data"
                             "/camcan/age.csv")
            path = "/storage/store/data/camcan-mne/trans/"
            df = df[(df.age < age_max) & (df.age > age_min)]
            all_subjects = list(df.Observations)
            subjects = []
            for subject in all_subjects:
                fname0 = get_raw_fname(dataset_name, subject)
                check0 = os.path.exists(fname0)
                if raw_only and check0:
                    subjects.append(subject)
                    continue
                fname4 = get_ave_fname(dataset_name, subject)
                check4 = os.path.exists(fname4)
                if ave_only and check4:
                    subjects.append(subject)
                    continue
                fname1 = get_bem_fname(dataset_name, subject)
                fname2 = path + "../freesurfer/%s/surf/lh.white" % subject
                fname3 = get_trans_fname(dataset_name, subject)
                check1 = os.path.exists(fname1)
                check2 = os.path.exists(fname2)
                check3 = os.path.exists(fname3)
                check5 = subject not in exclude_subjects[dataset_name]

                if check1 * check2 * check3 * check0 * check4 * check5:
                    subjects.append(subject)

        elif dataset_name == "ds117":
            subjects = ["sub%03d" % i for i in range(1, 20)
                        if "sub%03d" % i not in exclude_subjects[dataset_name]]
        else:
            raise ValueError("Unknown dataset %s." % dataset_name)
        fname = "/storage/store/work/hjanati/datasets/data/%s/info/"\
            % dataset_name
        fname += "subjects.list"
        f = open(fname, "wb")
        pickle.dump(subjects, f)
    else:
        data_path = "~/Dropbox/neuro_transport/code/"
        data_path += "mtw_experiments/meg/%s/" % dataset_name
        data_path = os.path.expanduser(data_path)
        f = open(data_path + "info/subjects.list", "rb")
        subjects = pickle.load(f)

    return subjects


if __name__ == "__main__":
    subjects_camcan = get_subjects_list("camcan")
    subjects_ds117 = get_subjects_list("ds117")
