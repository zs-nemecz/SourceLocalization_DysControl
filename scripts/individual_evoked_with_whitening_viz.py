import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set up dirs needed for all visualizations
data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
conditions = ['phase_rand', 'armenian', 'normal']
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
print('Number of participants: ', len(subjects))
noise_cov_files = ['noise-cov', 'reg_noise-cov']
noise_cov_types = ['with_non_reg_noise_cov', 'with_reg_noise_cov']
noise_cov_file = noise_cov_files[0]
noise_cov_type = noise_cov_types[0]
conditions = ['phase_rand', 'armenian', 'normal']
task = 'natural_reading'

for num, subject in enumerate(subjects):
    print('\n===============================================================================\n')
    print('Subject number: ', num)
    print(subject)
    print('\n===============================================================================\n')
    fig_ev, axs = plt.subplots(3,1, figsize = (12.0, 16.0), sharey=True, tight_layout=False)
    for row, condition in enumerate(conditions):
        cov_mat_file = op.join('noise_cov',condition, subject +'_'+ noise_cov_file + '.fif')
        noise_cov = mne.read_cov(cov_mat_file)
        if task == 'fixed_gaze':
            evoked_dir = op.join(task, condition, 'evoked')
            evoked_file =  op.join(evoked_dir, subject + '_' + condition + '-ave.fif')
            evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False)
            fig = evoked.plot(show=False, spatial_colors=True, axes = axs[row], noise_cov = noise_cov)
        else:
            evoked_dir = op.join(task, 'evoked')
            evoked_file =  op.join(evoked_dir, subject + '-ave.fif')
            evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False)
            evoked.plot(show=False, spatial_colors=True, axes = axs[row], noise_cov = noise_cov)

        title =  'Subject: ' + subject + '\nTask: ' + task + '\nCov Mat: ' + noise_cov_file
        axs[row].set_title(condition.capitalize(), x=0.8)
        if row < 2:
            axs[row].set_xlabel('')
    fig_ev.suptitle(title, x=0.5,y=0.99,weight='demibold')
    plt.subplots_adjust(top=0.90, hspace=0.2)
    plt.show()
    file =  op.join(task,'plots', subject + '_evoked_ '+ noise_cov_file + '.png')
    # fig_ev.savefig(file, bbox_inches='tight')
    plt.close(fig_ev)
    del fig_ev,axs
