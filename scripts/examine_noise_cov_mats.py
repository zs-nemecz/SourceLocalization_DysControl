import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set up dirs needed for all visualizations
task1_dir = '../fixed_gaze/'
data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
conditions = ['phase_rand', 'armenian', 'normal']
subjects = ['515155', '366394', '133288']
noise_cov_files = ['noise-cov', 'reg_noise-cov']
noise_cov_types = ['with_non_reg_noise_cov', 'with_reg_noise_cov']
conditions = ['phase_rand', 'armenian', 'normal']

            # evoked_file =  op.join(task1_dir, condition, evoked, subject + '-ave.fif')
            # evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False)
            # evoked.plot

for subject in subjects:
    fig_cov, axs_cov = plt.subplots(2,3, figsize = (18.0, 15.0), tight_layout=True)
    fig_cov.suptitle('Subject: ' + subject, x=0.3, y=0.99,fontweight='demibold')
    fig_svd, axs_svd = plt.subplots(2,3, figsize = (18.0, 15.0), tight_layout=True)
    fig_svd.suptitle('Subject: ' + subject, x=0.3, y=0.99, fontweight='demibold')
    row = 0
    for noise_cov_type, noise_cov_file in zip(noise_cov_types, noise_cov_files):
        for col, condition in enumerate(conditions):
            data_file = op.join(data_folder, subject, 'nat_ica', 'erp_eeg_' + condition +'_v2_avgref.set')
            epochs = mne.io.read_epochs_eeglab(data_file)
            cov_mat_file = op.join('noise_cov',condition, subject +'_'+ noise_cov_file + '.fif')
            print('\n===============================================================================\n')
            print(cov_mat_file)
            print('\n===============================================================================\n')
            noise_cov = mne.read_cov(cov_mat_file)
            if col == 2:
                fig1, fig2 = mne.viz.plot_cov(noise_cov, epochs.info, show_svd=True, show=False)
            else:
                fig1, fig2 = mne.viz.plot_cov(noise_cov, epochs.info, show_svd=True, show=False, colorbar = False)
            fig1.suptitle(noise_cov_file, x = 0.2, y=0.99)
            fig2.suptitle(noise_cov_file, x = 0.2, y=0.99)
            img1 = 'noise_cov/plots/original_images/' + subject + '_' + condition + '_' + noise_cov_file + '.png'
            img2 = 'noise_cov/plots/original_images/' + subject + '_svd_' + condition + '_' + noise_cov_file + '.png'
            fig1.savefig(img1)
            fig2.savefig(img2)
            img1 = mpimg.imread(img1)
            axs_cov[row,col].imshow(img1)
            axs_cov[row,col].set_axis_off()
            axs_cov[row,col].set_title(condition.capitalize())

            img2 = mpimg.imread(img2)
            axs_svd[row,col].imshow(img2)
            axs_svd[row,col].set_axis_off()
            axs_svd[row,col].set_title(condition.capitalize())

        row = row + 1
    plt.show()
    file1 = 'noise_cov/plots/' + subject + '.png'
    file2 = 'noise_cov/plots/' + subject + '_svd.png'
    fig_cov.savefig(file1)
    fig_svd.savefig(file2)
    del fig_cov, fig_svd, fig1, fig2
