import os.path as op
import numpy as np
from scipy import stats as stats
import mne
from mne.datasets import fetch_fsaverage
import time
import logging
logging.basicConfig(filename='D:/Zsuzsa/source_localization/fixed_gaze/results/ch_log.txt', level=logging.INFO, filemode='a')

data_folder = op.join('D:\\', 'Zsuzsa', 'source_localization', 'fixed_gaze')
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
n_subjects = len(subjects)
logging.info('Number of subjects included: {}'.format(n_subjects))

## Set parameters of interest here##
############################################
conditions = ['normal', 'armenian', 'phase_rand']
contrasts = [['normal', 'phase_rand'], ['normal', 'armenian'], ['armenian', 'phase_rand']]
methods = ['MNE', 'dSPM', 'eLORETA', 'sLORETA']
noise_cov= 'merged_nc'
tmins = [0.0] # [0.102, 0.102, 0.104, 0.112, 0.170, 0.170, 0.176, 0.178, 0.240, 0.244, 0.326, 0.326, 0.328, 0.332]
tmaxs = [0.450] # [0.122, 0.132, 0.124, 0.132, 0.190, 0.198, 0.196, 0.198, 0.260, 0.264, 0.346, 0.352, 0.348, 0.352]
inverse_model = 'inverse_model_auto'
lh_vertices = None
start_time = time.time()
logging.info(time.strftime("%H:%M:%S", time.gmtime(start_time)))
for tmin, tmax in zip(tmins, tmaxs):
    time_start = str(int(tmin * 1000))
    time_end = str(int(tmax * 1000))
    for method in methods:
        logging.info(method)
        cross_hemi = {}
        out_folder = op.join(data_folder, 'results', 'merged_nc', 'cross_hemi', method)
        for condition in conditions:
            stc_data = None
            file = condition + '_' + time_start + '-' + time_end
            for subject in subjects:
                stc_file = op.join(data_folder, condition, noise_cov, method, subject)
                stc = mne.read_source_estimate(stc_file, 'fsaverage')
                stc.crop(tmin, tmax)
                lh_vertices = stc.lh_data.shape[0]
                tstep = stc.tstep
                ## Morph to fsaverage_sym
                stc = mne.compute_source_morph(stc, subject_to='fsaverage_sym', smooth=5,
                                               warn=False,
                                               subjects_dir=subjects_dir).apply(stc)
                # Compute a morph-matrix mapping the right to the left hemisphere,
                # and vice-versa.
                morph = mne.compute_source_morph(stc, 'fsaverage_sym', 'fsaverage_sym',
                                                spacing=stc.vertices, warn=False,
                                                subjects_dir=subjects_dir, xhemi=True,
                                                verbose='error')  # creating morph map
                stc_xhemi = morph.apply(stc)
                if np.all(stc_data) == None:
                    stc_data = stc.data
                    stc_xhemi_data = stc_xhemi.data
                else:
                    stc_data = np.dstack((stc_data, stc.data))
                    stc_xhemi_data = np.dstack((stc_xhemi_data, stc_xhemi.data))
            logging.info('Collected {} stc data files from {} condition'.format(stc_data.shape[2], condition))
            X = stc_data[:, :, :] - stc_xhemi_data[:,:,:]
            X = X[:lh_vertices, :, :]
            cross_hemi[condition] = X

            out = stats.ttest_1samp(X, 0, axis=2) ## compute across participants
            ts = [out[0]]
            ps = [out[1]]
            ts.append(ts[-1])
            ps.append(mne.stats.fdr_correction(ps[0])[1])

            stc_file = op.join(data_folder, condition, noise_cov, method, '842608')
            stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin,tmax)

            stc_file = op.join(data_folder, 'normal', noise_cov, method, 'normal_average')
            stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin,tmax)
            stc.data[:lh_vertices, :] = ts[0]
            stc.data[lh_vertices:, :].fill(0)
            stc.save(op.join(out_folder,'uncorrected','ttest_tval_' + file))
            stc.data[:lh_vertices, :] = ps[1]
            stc.data[lh_vertices:, :].fill(1)
            stc.save(op.join(out_folder,'fdr','fdrttest_pval_' + file))
            stc.data[:lh_vertices, :] = ps[0]
            stc.save(op.join(out_folder,'uncorrected','ttest_pval_' + file))
        logging.info('Collected cross hemi contrasts for {} conditions'.format(len(cross_hemi)))
        logging.info([c for c in cross_hemi])

        for contrast in contrasts:
            print('Comparing {} with {} condition'.format(contrast[0], contrast[1]))
            file = contrast[0] + '_' + contrast[1] + '_' + time_start + '-' + time_end
            C = cross_hemi[contrast[0]] - cross_hemi[contrast[1]]
            out = stats.ttest_1samp(C, 0, axis=2)  ## compute across participants
            ts = [out[0]]
            ps = [out[1]]
            ts.append(ts[-1])
            ps.append(mne.stats.fdr_correction(ps[0])[1])

            stc_file = op.join(data_folder, contrast[0], noise_cov, method, '842608')
            stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin, tmax)

            stc_file = op.join(data_folder, 'normal', noise_cov, method, 'normal_average')
            stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin, tmax)
            stc.data[:lh_vertices, :] = ts[0]
            stc.data[lh_vertices:, :].fill(0)
            stc.save(op.join(out_folder, 'uncorrected', 'ttest_tval_' + file))
            stc.data[:lh_vertices, :] = ps[1]
            stc.data[lh_vertices:, :].fill(1)
            stc.save(op.join(out_folder, 'fdr', 'fdrttest_pval_' + file))
            stc.data[:lh_vertices, :] = ps[0]
            stc.save(op.join(out_folder, 'uncorrected', 'ttest_pval_' + file))
end_time = time.time()
logging.info(time.strftime("%H:%M:%S", time.gmtime(end_time)))
elapsed_time = end_time - start_time
logging.info('Elapsed time:')
logging.info(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
logging.info('--------------------------------------------------------------------------------------------------------')
