import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set up dirs needed for all visualizations
task1_dir = '../fixed_gaze/'
data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
conditions = ['phase_rand', 'armenian', 'normal']
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '181585', '269257', '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
conditions = ['phase_rand', 'armenian', 'normal']

for subject in subjects:
    fig_cov, axs_cov = plt.subplots(1,3, figsize = (12.0, 6.0), tight_layout=True)
    fig_cov.suptitle('Subject: ' + subject, x=0.3, y=0.99,fontweight='demibold')
    fig_svd, axs_svd = plt.subplots(1,3, figsize = (12.0, 6.0), tight_layout=True)
    fig_svd.suptitle('Subject: ' + subject, x=0.3, y=0.99, fontweight='demibold')
    cov_mat_file = op.join('noise_cov','all', subject +'_reg_noise-cov.fif')
    noise_cov = mne.read_cov(cov_mat_file)
    for col, condition in enumerate(conditions):
        data_file = op.join(data_folder, subject, 'nat_ica', 'erp_eeg_' + condition +'_v2_avgref.set')
        epochs = mne.io.read_epochs_eeglab(data_file)
        print('\n===============================================================================\n')
        print(cov_mat_file)
        print('\n===============================================================================\n')
        if col == 2:
            fig1, fig2 = mne.viz.plot_cov(noise_cov, epochs.info, show_svd=True, show=False)
        else:
            fig1, fig2 = mne.viz.plot_cov(noise_cov, epochs.info, show_svd=True, show=False, colorbar = False)
        img1 = 'noise_cov/plots/original_images/' + subject + '_merged_conditions_reg_noise-cov.png'
        img2 = 'noise_cov/plots/original_images/' + subject + '_svd_merged_conditions_reg_noise-cov.png'
        fig1.savefig(img1, bbox_inches='tight')
        fig2.savefig(img2, bbox_inches='tight')
        plt.close()
        img1 = mpimg.imread(img1)
        axs_cov[col].imshow(img1)
        axs_cov[col].set_axis_off()
        axs_cov[col].set_title(condition.capitalize())

        img2 = mpimg.imread(img2)
        axs_svd[col].imshow(img2)
        axs_svd[col].set_axis_off()
        axs_svd[col].set_title(condition.capitalize())

    # plt.show()
    plt.close()
    file1 = 'noise_cov/plots/' + subject + '_conditions_merged.png'
    file2 = 'noise_cov/plots/' + subject + '_conditions_merged_svd.png'
    fig_cov.savefig(file1,  bbox_inches='tight')
    fig_svd.savefig(file2,  bbox_inches='tight')
    del fig_cov, fig_svd, fig1, fig2
