### This script creates visualizations for each subject's noise covariant matrices and source estimates
### 1. Create plots for cov matrices
### 2. Figure for peak source estimate with both cov matrices
### 3. Animations in sensor and source space
### 4. Avarage data
import os
import os.path as op
from mayavi import mlab
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt
from mne.minimum_norm import make_inverse_operator, apply_inverse

mlab.options.offscreen = True

results_folder = '/home/zsuzsanna/Documents/MTA/source_localization/natural_reading/with_non_reg_noise_cov'
noise_cov_folder = '/home/zsuzsanna/Documents/MTA/source_localization/noise_cov/'

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# noise_cov = mne.read_cov(noise_cov_folder + subject+'_reg_noise-cov.fif')

# average_stc_file = op.join(results_folder, 'dSPM', 'average')
# average_stc = mne.read_source_estimate(average_stc_file, 'fsaverage')
# vertno_max, time_max = average_stc.get_peak()
# fs_dir = fetch_fsaverage(verbose=True)
# subjects_dir = op.dirname(fs_dir)
# surfer_kwargs = dict(
#     hemi='split', subjects_dir=subjects_dir, views=['lateral', 'caudal', 'ventral'],
#     initial_time= time_max, time_unit='s', size=(592, 608), smoothing_steps=5)
#
# brain = average_stc.plot(**surfer_kwargs)
# brain.save_movie('test_video.gif')
# brain.add_text(0.1, 0.9,  'MNE (plus location of maximal activation)', 'title',
#                font_size=14)
# ##add foci of peak activation
# brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
#                scale_factor=0.6, alpha=0.5)

### 3. Animations in sensor and source space


### 4. Avarage data
inverse_solvers = ['eLORETA', ''] #'MNE','dSPM','sLORETA','eLORETA'
lims = [[0.4, 0.45, 1.1], []] #[6.5e-12, 7.0e-12, 1.6e-11 ], [16, 17, 31], [5, 5.8, 14], [0.4, 0.45, 1.1]
# #fixed gaze lims = ([1.25e-11, 1.5e-11, 2.7e-11 ], [6, 7, 12], [3, 4, 6], [0.2, 0.25, 0.4]) #[1.25e-11, 1.5e-11, 2.7e-11 ],[6, 7, 12],[3, 4, 6],[0.2, 0.25, 0.4]

for method, lim in zip(inverse_solvers, lims):
    video_file = op.join(results_folder, method, method + '_average.mov')
    # animation_file = op.join(results_folder, method, method + '_average.gif')
    # image_files = op.join(results_folder, method, method)
    average_stc_file = op.join(results_folder, method, 'average')
    average_stc = mne.read_source_estimate(average_stc_file, 'fsaverage')
    # vertno_max, time_max = average_stc.get_peak()

    ## parietal view
    # par_surf_kwargs =     surfer_kwargs = dict(
    #         hemi='split', subjects_dir=subjects_dir,  clim=dict(kind='value', lims=lim),
    #         views=['parietal'],
    #         initial_time= 0.186, time_unit='s', size=(1800, 1800), smoothing_steps=5)
    #
    # brain_par = average_stc.plot(**par_surf_kwargs)
    # brain_par.add_text(0.75, 0.2, method, 'title', font_size=23)
    # brain_par.save_image(image_files + '_par_view_n2_average.png')

    ## other views
    surfer_kwargs = dict(
        hemi='split', subjects_dir=subjects_dir, clim=dict(kind='value', lims=lim),
        views=['lateral', 'caudal', 'ventral'],
        initial_time= 0.00, time_unit='s', size=(1200, 1200), smoothing_steps=5)
    brain = average_stc.plot(**surfer_kwargs)
    brain.add_text(0.7, 0.2, method, 'title', font_size=23)

    brain.save_movie(video_file, tmin=0.0, tmax= 0.600, time_dilation = 34, framerate = 24)
    del brain
    # brain.save_movie(animation_file, tmin=0.16, tmax= 0.280, time_dilation = 200, framerate = 24)
    # brain.save_image(image_files + '_normal_n1_average.png')
