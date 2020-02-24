### Create noise covariant matrices from fixed gaze data epochs
### Steps:
### 1. Setup, add path and subject names
### 2. Read preprocessed EEG epochs
### 3. Create regularized and non-regularized noise covariance matrices
### 4. Write matrices to files

### 1. Setup, add path and subject names
import os.path as op
import numpy as np
import mne

condition = input('Add experimental condition: \'phase_rand\', \'normal\', or \'armenian\' \n')

data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
out_folder = op.join('noise_cov', condition)
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
for subject in subjects:
### 2. Read preprocessed EEG epochs
    data_file = op.join(data_folder, subject, 'nat_ica', 'erp_eeg_' + condition +'_v2_avgref.set')
    epochs=mne.io.read_epochs_eeglab(data_file)

### 3. Create regularized and non-regularized noise covariance matrices
    print('\nComputing non-regularized cov-mat for subject ', subject)
    noise_cov_baseline = mne.compute_covariance(epochs, tmax=0.) #non-regularized
    print('\nComputing regularized cov-mat for subject ', subject)
    reg_noise_cov_baseline = mne.compute_covariance(epochs, tmax=0., method='auto', rank=None, verbose=True) #regularized
    ##Plot
    # noise_cov_baseline.plot(epochs.info, proj=False)
    # reg_noise_cov_baseline.plot(epochs.info, proj=False)

### 4. Write matrices to files
    print('\nSaving matrices for subject ', subject)
    noise_cov_file = op.join(out_folder, subject+'_noise-cov.fif')
    reg_noise_cov_file = op.join(out_folder, subject+'_reg_noise-cov.fif')
    mne.write_cov(noise_cov_file,noise_cov_baseline)
    mne.write_cov(reg_noise_cov_file,reg_noise_cov_baseline)
    print('Done.')
    print('==========================================================================================================================================================')
