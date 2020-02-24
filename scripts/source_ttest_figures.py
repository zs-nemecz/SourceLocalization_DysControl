import os.path as op
from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from scipy import stats as stats
import mne
from mne.datasets import fetch_fsaverage

method = 'MNE'
tmin = 0.00
tmax = 0.450
conditions = ['armenian','phase_rand']
noise_cov = 'with_non_reg_noise_cov'

data_folder = '/home/zsuzsanna/Documents/MTA/source_localization/fixed_gaze/'
src_fname = '/home/zsuzsanna/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-ico-5-src.fif'
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
print('Number of participants included: ',len(subjects))

condition0_stc_data = None
condition1_stc_data = None
tstep = None

# for condition in conditions:
#     stc_data = None
#     for subject in subjects:
#         stc_file = op.join(data_folder, condition, noise_cov, method, subject)
#         stc = mne.read_source_estimate(stc_file, subject)
#         stc.crop(tmin, tmax)
#         tstep = stc.tstep
#         if np.all(stc_data) == None:
#             stc_data = stc.data
#         else:
#             stc_data = np.dstack((stc_data, stc.data))
#     print('Collected {} stc data files from {} condition'.format(stc_data.shape[2], condition))
#     if condition == conditions[0]:
#         condition0_stc_data = stc_data
#     else:
#         condition1_stc_data = stc_data
# print('Subtracting {} condition from condition {}'.format(conditions[1],conditions[0]))
# X = condition0_stc_data[:, :, :] - condition1_stc_data[:,:,:]
# out = stats.ttest_1samp(X, 0, axis=2) ## compute across participants
# ts = [out[0]]
# ps = [out[1]]
# ts.append(ts[-1])
# ps.append(mne.stats.fdr_correction(ps[0])[1])
#
# stc_file = op.join(data_folder, 'normal', noise_cov, method, 'normal_average')
# stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin,tmax)
# fdr_stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin,tmax)
# fdr_stc.data = ps[1] * -1
# stc.data = ps[0] * -1
# stc.save('visualizations/fixed_gaze_source_ttest_' + conditions[0] +'-'+ conditions[1])
# fdr_stc.save('visualizations/fixed_gaze_source_fdrttest_' + conditions[0] +'-'+ conditions[1])

##Read p vals from stc if already saved
stc = mne.read_source_estimate('visualizations/fixed_gaze_source_ttest_' + conditions[0] +'-'+ conditions[1], 'fsaverage')

# time_points = [0.130, 0.190, 0.260] # Time points of interest
# time_points = [0.090, 0.100, 0.112, 0.122, 0.130, 0.140] # Time points of interest - P1
# time_points = [0.160, 0.170, 0.180, 0.185, 0.190, 0.200, 0.210] # Time points of interest - N1
# time_points = [0.230, 0.240, 0.250, 0.255, 0.260, 0.270, 0.280] # Time points of interest - N2
# time_points = [0.310, 0.320, 0.330, 0.340, 0.345, 0.350, 0.360, 0.370] # Time points of interest - P2
time_points = [0.090, 0.100, 0.112, 0.122, 0.130, 0.140, 0.160, 0.170, 0.180, 0.185, 0.190, 0.200, 0.210, 0.230, 0.240, 0.250, 0.255, 0.260, 0.270, 0.280, 0.310, 0.320, 0.330, 0.340, 0.345, 0.350, 0.360, 0.370]

mlab.options.offscreen = True
for time_point in time_points:
    time = str(int(time_point*1000))
    for view, hemi in zip(['caudal', 'ventral', 'lateral', 'lateral'],['both', 'both','lh', 'rh']):
        surfer_kwargs = dict(hemi=hemi, views=view, clim=dict(kind='value', lims=[-0.06, -0.04,0.0]),
                        time_unit='s', smoothing_steps=5, colorbar = True, size=(1200, 1200), time_label = None)
        file1 = 'visualizations/fixed_gaze_source_ttest_' + conditions[0] +'-'+ conditions[1]+ '_' + view +'_' + hemi + '_t'+time+'.png'
        fig, axs = plt.subplots(1,1, sharex=True, sharey = True, tight_layout=True, figsize = (12.8, 9.6))
        fig.suptitle('Time point: ' + time + ' ms')
        brain = stc.plot(initial_time = time_point, **surfer_kwargs)
        img = mlab.screenshot(antialiased = True)
        mlab.close()
        axs.imshow(img)
        axs.set_axis_off()
        axs.set_title(conditions[0].capitalize() + '-' + conditions[1].capitalize(), loc='left')
        fig.savefig(file1)
