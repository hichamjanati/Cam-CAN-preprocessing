import os
from mne.bem import make_watershed_bem

# the paths to Freesurfer reconstructions
DATA_PATH = '/datasabzi/anuja/freesurfer'
SUBJECT_DIR = DATA_PATH + '/subjects'

def generate_bem(subject, subject_dir):
    """
    Creates BEM surfaces using the FreeSurfer watershed algorithm for the given subject.
    """
    make_watershed_bem(subject, subject_dir)

if __name__ == "__main__":
    for subject in os.listdir(SUBJECT_DIR):
        if subject.startswith("CC"):
            generate_bem(subject, SUBJECT_DIR)
            
