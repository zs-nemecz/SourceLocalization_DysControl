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
noise_cov_files = ['noise-cov', 'reg_noise-cov']
noise_cov_types = ['with_non_reg_noise_cov', 'with_reg_noise_cov']
conditions = ['phase_rand', 'armenian', 'normal']
task = 'fixed_gaze'

for subject in subjects:
    for condition in conditions:
        fig_whit, axs = plt.subplots(1,2,figsize = (18.0, 6.0), tight_layout=True)
        if task == 'fixed_gaze':
            evoked_dir = op.join(task1_dir, condition, 'evoked')
            title = 'Subject: ' + subject + '\nTask: ' + task + '\nCondition: '+ condition
        else:
            evoked_dir = op.join(task2_dir, 'evoked')
            title = 'Subject: ' + subject + '\nNoise Cov Mat from: '+condition + '\nTask: ' + task
        fig_whit.suptitle(title, x=0.5,y=0.99,weight='demibold')
        col = 0
        for noise_cov_type, noise_cov_file in zip(noise_cov_types, noise_cov_files):
            if task == 'fixed_gaze':
                evoked_file =  op.join(evoked_dir, subject + '_' + condition + '-ave.fif')
            else:
                evoked_file =  op.join(evoked_dir, subject + '-ave.fif')
            evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False)
            cov_mat_file = op.join('noise_cov',condition, subject +'_'+ noise_cov_file + '.fif')
            print('\n===============================================================================\n')
            print(cov_mat_file)
            print('\n===============================================================================\n')
            noise_cov = mne.read_cov(cov_mat_file)
            fig1 = mne.viz.plot_evoked_white(evoked,noise_cov, show=False)
            fig1.suptitle(noise_cov_file, x = 0.2, y=0.99)
            if task=='fixed_gaze':
                img1 = 'noise_cov/plots/original_images/' + subject + '_FG_white_' + condition + '_' + noise_cov_file + '.png'
            else:
                img1 = 'noise_cov/plots/original_images/' + subject + '_NR_white_' + condition + '_' + noise_cov_file + '.png'
            fig1.savefig(img1, bbox_inches='tight')
            plt.close(fig1)
            img1 = mpimg.imread(img1)
            axs[col].imshow(img1)
            axs[col].set_axis_off()
            col += 1
            del fig1,img1

        # plt.show()
        if task=='fixed_gaze':
            file1 = 'noise_cov/plots/' + subject + '_FG_' + condition + '_white.png'
        else:
            file1 = 'noise_cov/plots/' + subject + '_NR_' + condition + '_white.png'
        fig_whit.savefig(file1, bbox_inches='tight')
        plt.close(fig_whit)
        del fig_whit,axs
