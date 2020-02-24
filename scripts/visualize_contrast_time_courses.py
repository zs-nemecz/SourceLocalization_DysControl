import os
import os.path as op
from mayavi import mlab
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt

data_folder = '/home/zsuzsanna/Documents/MTA/source_localization/fixed_gaze/'
conditions = ['armenian', 'phase_rand']
noise_cov = 'with_non_reg_noise_cov'
method = 'dSPM'
out_folder = op.join(data_folder, 'results', noise_cov)
video_file = op.join(out_folder, method + '_' + conditions[0] + '_' + conditions[1] + '_contrast.mov')
condition1_file = op.join(data_folder, conditions[0] + '_condition',noise_cov, method, conditions[0] + '_average')
condition2_file = op.join(data_folder, conditions[1] + '_condition',noise_cov, method, conditions[1] + '_average')

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

stc1 = mne.read_source_estimate(condition1_file, 'fsaverage').crop(0.0, 0.300)
stc2 = mne.read_source_estimate(condition2_file, 'fsaverage').crop(0.0, 0.300)

contrast = stc1.data - stc2.data
stc1.data = contrast

surfer_kwargs = dict(
    hemi='split', subjects_dir=subjects_dir, #clim=dict(kind='value', lims=lim),
    views=['lateral', 'caudal', 'ventral'],
    initial_time= 0.00, time_unit='s', size=(1200, 1200), smoothing_steps=5)

brain = stc1.plot(**surfer_kwargs)
brain.save_movie(video_file, tmin=0.0, tmax= 0.300, time_dilation = 50, framerate = 24)
mlab.show()
