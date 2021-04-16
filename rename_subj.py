import os
os.chdir('/datasabzi/hashemi/freesurfer/subjects/test_subj/')
# Am I in the correct directory?

print(os.getcwd())
# Print all the current file names
for f in os.listdir():
    print(f)
    new_filename = f.split('-')[1]
    print(new_filename)
    os.rename(f, new_filename)