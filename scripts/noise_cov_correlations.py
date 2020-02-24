import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#todo: felső v alsó háromszög
task1_dir = '../fixed_gaze/'
data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
conditions = ['phase_rand', 'armenian', 'normal']
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
print('Number of participants: ', len(subjects))
noise_cov_files = ['noise-cov', 'reg_noise-cov']
conditions = ['phase_rand', 'armenian', 'normal']
reg_coefs = [[],[],[]]
non_reg_coefs= [[],[],[]]
for noise_cov_file in noise_cov_files:
    for subject in subjects:
        print('\n===============================================================================\n')
        print('SUBJECT ', subject)
        vars = []
        for condition in conditions:
            cov_mat_file = op.join('noise_cov',condition, subject +'_'+ noise_cov_file + '.fif')
            noise_cov = mne.read_cov(cov_mat_file, verbose=False)
            # var = np.triu(noise_cov.data)
            var = np.reshape(noise_cov.data, -1)
            vars.append(var)
        table = np.array([vars[0], vars[1], vars[2]])
        coef = np.corrcoef(table)
        print('Correlation coefficients:\n', coef)
        if noise_cov_file == 'noise-cov':
            non_reg_coefs[0].append(coef[0,1]) #random and armenian
            non_reg_coefs[1].append(coef[0,2]) # random and normal
            non_reg_coefs[2].append(coef[1,2]) # armenian and normal
        else:
            reg_coefs[0].append(coef[0,1]) #random and armenian
            reg_coefs[1].append(coef[0,2]) # random and normal
            reg_coefs[2].append(coef[1,2]) # armenian and normal
    print('\n***************************************************************************************\n')
figure, axs = plt.subplots(1,3, sharey=True)
axs[0].plot(non_reg_coefs[0], 'r+')
axs[0].plot(reg_coefs[0], 'bx')
axs[0].set_title('Random - Armenian')
axs[1].plot(non_reg_coefs[1], 'r+')
axs[1].plot(reg_coefs[1], 'bx')
axs[1].set_title('Random - Normal')
axs[2].plot(non_reg_coefs[2], 'r+')
axs[2].plot(reg_coefs[2], 'bx')
axs[2].set_title('Armenian - Normal')
plt.show()
