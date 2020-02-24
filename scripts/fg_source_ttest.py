#Note: reading of stc files/ condition could be done only once
import os.path as op
from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from scipy import stats as stats
import mne
from mne.datasets import fetch_fsaverage

methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
tmin = 0.112
tmax = 0.132
time_start = str(int(tmin*1000))
time_end = str(int(tmax*1000))
contrasts = [['normal', 'phase_rand'],['normal','armenian'],['armenian','phase_rand']]
noise_cov = 'merged_nc'

data_folder = '/home/zsuzsanna/Documents/MTA/source_localization/fixed_gaze/'
src_fname = '/home/zsuzsanna/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-ico-5-src.fif'
out_folder = op.join(data_folder,'results', 'merged_nc')
## subject excluded '181585'
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
print('Number of participants included: ',len(subjects))
tstep = None
for method in methods:
    out_folder = op.join(data_folder,'results', 'merged_nc', method)
    for conditions in contrasts:
        condition0_stc_data = None
        condition1_stc_data = None
        file = conditions[0] +'-'+ conditions[1] +'_' + time_start +'-'+ time_end
        for condition in conditions:
            stc_data = None
            for subject in subjects:
                stc_file = op.join(data_folder, condition, noise_cov, method, subject)
                stc = mne.read_source_estimate(stc_file, subject)
                stc.crop(tmin, tmax)
                tstep = stc.tstep
                if np.all(stc_data) == None:
                    stc_data = stc.data
                else:
                    stc_data = np.dstack((stc_data, stc.data))
            print('Collected {} stc data files from {} condition'.format(stc_data.shape[2], condition))
            if condition == conditions[0]:
                condition0_stc_data = stc_data
            else:
                condition1_stc_data = stc_data
        print('Subtracting {} condition from condition {}'.format(conditions[1],conditions[0]))
        X = condition0_stc_data[:, :, :] - condition1_stc_data[:,:,:]
        out = stats.ttest_1samp(X, 0, axis=2) ## compute across participants
        ts = [out[0]]
        ps = [out[1]]
        ts.append(ts[-1])
        ps.append(mne.stats.fdr_correction(ps[0])[1])

        stc_file = op.join(data_folder, 'normal', noise_cov, method, 'normal_average')
        stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin,tmax)
        stc.data = ts[0]
        stc.save(op.join(out_folder,'ttest_tval_' + file))
        stc.data = ps[1]
        stc.save(op.join(out_folder,'fdrttest_pval_' + file))
        stc.data = ps[0]
        stc.save(op.join(out_folder,'ttest_pval_' + file))
