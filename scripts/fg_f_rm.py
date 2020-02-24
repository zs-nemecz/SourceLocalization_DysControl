import os.path as op
import mne
import numpy as np
import time
from mne.datasets import fetch_fsaverage
from scipy import stats as stats

data_folder = op.join('D:\\', 'Zsuzsa', 'source_localization', 'fixed_gaze')
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
n_subjects = len(subjects)
conditions = ['normal', 'armenian', 'phase_rand']
condition0_stc_data = None
condition1_stc_data = None
condition2_stc_data = None

## Set parameters of interest here##
############################################
noise_cov= 'merged_nc'
inverse_model = 'inverse_model_auto'
method ='MNE'
methods = ['MNE', 'dSPM', 'eLORETA', 'sLORETA']
tmin = 0.0
tmax = 0.450
p_threshold = 0.05
n_permutations = 1000
###########################################

start_time = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(start_time)))
for c, condition in enumerate(conditions):
    stc_data = None
    for s, subject in enumerate(subjects):

        if inverse_model == 'inverse_model_auto':
            stc_file = op.join(data_folder, condition, noise_cov, method, subject)
        else:
            stc_file = op.join(data_folder, condition, noise_cov, method, subject + '_fixed_inverse')
        stc = mne.read_source_estimate(stc_file, subject)
        stc.crop(tmin, tmax)
        tstep = stc.tstep

        if np.all(stc_data) == None:
            stc_data = stc.data
        else:
            stc_data = np.dstack((stc_data, stc.data))

# TODO reshape the arrays
    if condition == conditions[0]:
        condition0_stc_data = stc_data
    elif condition == conditions[1]:
        condition1_stc_data = stc_data
    else:
        condition2_stc_data = stc_data

end_time = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(end_time)))
elapsed_time = end_time - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
print(condition0_stc_data.shape)
