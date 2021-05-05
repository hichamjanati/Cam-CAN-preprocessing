import os
from mne.bem import make_watershed_bem
from joblib import Parallel, delayed

# the paths to Freesurfer reconstructions
DATA_PATH = '/datasabzi/anuja/freesurfer'
SUBJECTS_DIR = DATA_PATH + '/subjects'

def generate_bem(subject, subjects_dir):
    """
    Creates BEM surfaces using the FreeSurfer watershed algorithm for the given subject.
    """
    make_watershed_bem(subject, subject_dir, overwrite=True)

if __name__ == "__main__":
    subjects = []
    for subject in os.listdir(SUBJECT_DIR):
        if subject.startswith("CC"):
            subjects.append(subject)
    
    Parallel(n_jobs=25)(delayed(generate_bem)(subject, SUBJECT_DIR) for subject in subjects)
