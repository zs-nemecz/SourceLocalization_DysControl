import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set up dirs needed for all visualizations
task1_dir = 'fixed_gaze/'
task2_dir = 'natural_reading/'
data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
conditions = ['phase_rand', 'armenian', 'normal']
subjects = ['587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
print('Number of participants: ', len(subjects))
conditions = ['phase_rand', 'armenian', 'normal']
task = 'fixed_gaze'

for subject in subjects:
    for condition in conditions:
        if task == 'fixed_gaze':
            evoked_dir = op.join(task1_dir, condition, 'evoked')
            evoked_file =  op.join(evoked_dir, subject + '_' + condition + '-ave.fif')
            title = 'Subject: ' + subject + '\nTask: ' + task + '\nCondition: '+ condition
            file = 'noise_cov/plots/' + subject + '_FG_merged_' + condition + '_white.png'
        else:
            evoked_dir = op.join(task2_dir, 'evoked')
            title = 'Subject: ' + subject + '\nTask: ' + task + '\nNoise Cov Mat from: FG Merged conditions'
            evoked_file =  op.join(evoked_dir, subject + '-ave.fif')
            file = 'noise_cov/plots/' + subject + '_NR_fg_merged_white.png'
        evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False)
        cov_mat_file = op.join('noise_cov','all', subject +'_reg_noise-cov.fif')
        print('\n===============================================================================\n')
        print(cov_mat_file)
        print('\n===============================================================================\n')
        noise_cov = mne.read_cov(cov_mat_file)
        fig_whit = mne.viz.plot_evoked_white(evoked,noise_cov, show=False)
        fig_whit.suptitle(title, x = 0.85, y=0.99)

        fig_whit.savefig(file)
        plt.close(fig_whit)
        del fig_whit
