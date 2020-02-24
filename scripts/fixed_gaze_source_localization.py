### Source localization with template anatomy and standard electrode locations
### Steps:
### 1. Setup and add path and subject names
### 2. Read noise covariance matrix from preprocessed resting state data
### 3. Read preprocessed EEG epochs
### 4. Compute the evoked response
### 5. Read previously created forward solution (see create_template_fwd.py)
### 6. Make inverse operator
### 7. Compute inverse solution


### 1. Setup and add path and subject names
import os.path as op
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

condition = input('Add experimental condition: \'phase_rand\', \'normal\', or \'armenian\' \n')
cov_matrix = input('Choose covariant matrix: \'non_reg\' or \'reg\' \n')

data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
out_folder = op.join('fixed_gaze', condition)
noise_cov_folder = op.join('noise_cov/', condition)
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
inverse_solvers = ['MNE']#, 'dSPM', 'sLORETA', 'eLORETA')
loose = 0.2
depth = 0.8
print('Inverse operator characteristics \nLoose: {} \nDepth: {}'.format(loose,depth))

for subject in subjects:
    ### 2. Read previously created forward solution (see create_template_fwd.py)
    fwd = mne.read_forward_solution('template-fwd.fif')

    ### 3. Read noise covariance matrix from preprocessed resting state data
    #######TODO#### make file reading nicer
    if cov_matrix == 'non_reg':
        noise_cov = mne.read_cov(op.join(noise_cov_folder, subject+ '_noise-cov.fif'))
    else:
        noise_cov = mne.read_cov(op.join(noise_cov_folder, subject + '_reg_noise-cov.fif'))

    ### 4. Read preprocessed EEG epochs
    data_file = op.join(data_folder, subject, 'nat_ica', 'erp_eeg_' + condition  + '_v2_avgref.set')
    epochs = mne.io.read_epochs_eeglab(data_file)
    epochs.set_eeg_reference(projection=True)  # needed for inverse modeling

    ### 5. Compute the evoked response
    evoked = epochs.average().pick('eeg')
    evoked_file = op.join(out_folder, 'evoked', subject + '_' + condition + '-ave.fif')
    evoked.condition = condition
    evoked.save(evoked_file)
    del epochs  # to save memory


    ### 6. Make inverse operator
    print('\nCreating inverse operator for subject ', subject)
    info = evoked.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=loose, depth=depth) # IMPORTANT, default loose value is 0.2
    del noise_cov
    del fwd

    ### 7. Compute inverse solution
    snr = 3.
    lambda2 = 1. / snr ** 2
    for method in inverse_solvers:
        print('\nComputing inverse solutions with method {} for subject {}'.format(method, subject))
        if method == 'eLORETA':
            stc = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None, verbose=True)
        else:
            stc, residual = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None, return_residual=True, verbose=True)
            res_file = op.join(out_folder, 'with_'+ cov_matrix +'_noise_cov', method, 'residual', subject + '-ave.fif')
            residual.save(res_file)
            del residual

        stc_file = op.join(out_folder, 'with_'+ cov_matrix + '_noise_cov', method, subject)
        print('\nSaving inverse solution with method {} for subject {} \n to file: {}'.format(method, subject, stc_file))
        stc.save(stc_file)
        del stc
    print('=========================================================================================================================================')
