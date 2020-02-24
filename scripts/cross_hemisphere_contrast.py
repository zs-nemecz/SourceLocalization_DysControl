### Visualize cross-hemisphere contrast - Natural Reading task
import mne
import os.path as op
import matplotlib.pyplot as plt
from mayavi import mlab
from mne.datasets import fetch_fsaverage

## Setup path and file names
data_folder = 'natural_reading'
method = 'MNE'
noise_cov = 'with_non_reg_noise_cov'
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
video_file = op.join(data_folder, 'results', noise_cov, method + '_cross_hemisphere.mov')
stc_file = op.join(data_folder, noise_cov, method, 'average')

stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(0.00,0.40)

# Morph the data to fsaverage_sym, for which we have left_right
# registrations:
stc = mne.compute_source_morph(stc,'fsaverage', 'fsaverage_sym', smooth=5,
                               warn=True,
                               subjects_dir=subjects_dir).apply(stc)

# Compute a morph-matrix mapping the right to the left hemisphere,
# and vice-versa.
morph = mne.compute_source_morph(stc, 'fsaverage_sym', 'fsaverage_sym',
                                 spacing=stc.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error')  # creating morph map
stc_xhemi = morph.apply(stc)

# Subtract them and plot the result:
diff = stc - stc_xhemi
fig = diff.plot(hemi='split', views = ['lateral', 'caudal', 'ventral'], subjects_dir=subjects_dir, initial_time=0.0,
          size=(1200, 1200))
## Save video of time course
fig.save_movie(video_file, tmin=0.0, tmax= 0.400, time_dilation = 50, framerate = 24)
