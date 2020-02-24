import os.path as op
import numpy as np
import mne
from mayavi import mlab
import matplotlib.pyplot as plt
from mne.datasets import fetch_fsaverage

###########################################################################################xx
# Contrast function
def create_cross_hemi_contrast(stc, tmin = None, tmax = None):

    # Read files if input is string
    if isinstance(stc, str):
        stc = mne.read_source_estimate(stc, 'fsaverage')

    # Crop
    stc.crop(tmin, tmax)

    # Load FsAverage anatomy
    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = op.dirname(fs_dir)

    # Morph the data to fsaverage_sym, for which we have left_right registrations:
    stc = mne.compute_source_morph(stc,'fsaverage', 'fsaverage_sym', smooth=5, warn=True,
                                   subjects_dir=subjects_dir).apply(stc)
    # Compute a morph-matrix mapping the right to the left hemisphere,
    # and vice-versa.
    morph = mne.compute_source_morph(stc, 'fsaverage_sym', 'fsaverage_sym',
                                     spacing=stc.vertices, warn=False,
                                     subjects_dir=subjects_dir, xhemi=True,
                                     verbose='error')  # creating morph map
    stc_xhemi = morph.apply(stc)

    # Create contrast
    contrast = stc - stc_xhemi

    # Return stc with contrast data
    return contrast
###########################################################################################

# STC files
# Set up dirs needed for all visualizations
experiment_dir = 'natural_reading/'
method = 'MNE'
noise_cov = 'with_non_reg_noise_cov'

stc_file = op.join(experiment_dir, noise_cov, method, 'average')
contrast = create_cross_hemi_contrast(stc_file, tmin = 0, tmax = 450) # create contrast stc

# time_points = [0.260, 0.345] # Time points of interest
# time_points = [0.090, 0.100, 0.112, 0.122, 0.130, 0.140] # Time points of interest - P1
# time_points = [0.160, 0.170, 0.180, 0.185, 0.190, 0.200, 0.210] # Time points of interest - N1
# time_points = [0.230, 0.240, 0.250, 0.255, 0.260, 0.270, 0.280] # Time points of interest - N2
# time_points = [0.310, 0.320, 0.330, 0.340, 0.345, 0.350, 0.360, 0.370] # Time points of interest - P2
time_points = [0.090, 0.100, 0.112, 0.122, 0.130, 0.140, 0.160, 0.170, 0.180, 0.185, 0.190, 0.200, 0.210, 0.230, 0.240, 0.250, 0.255, 0.260, 0.270, 0.280, 0.310, 0.320, 0.330, 0.340, 0.345, 0.350, 0.360, 0.370]

mlab.options.offscreen = True

for time_point in time_points:
    time = str(int(time_point*1000))
    for view, hemi in zip(['caudal', 'ventral', 'lateral', 'lateral'],['both', 'both','lh', 'rh']):
        surfer_kwargs = dict(hemi=hemi, views=view, #clim=dict(kind='value', lims=[0.5e-12, 0.5e-12, 6.5e-12]),
                        time_unit='s', smoothing_steps=5, colorbar = True, size=(1200, 1200), time_label = None)
        file1 = 'visualizations/natural_reading_source_contrast_' + view +'_' + hemi + '_t'+time+'.png'
        fig, axs = plt.subplots(1,1, sharex=True, sharey = True, tight_layout=True, figsize = (12.8, 9.6))
        fig.suptitle('Time point: ' + time + ' ms')
        brain = contrast.plot(initial_time = time_point, **surfer_kwargs)
        img = mlab.screenshot(antialiased = True)
        mlab.close()
        axs.imshow(img)
        axs.set_axis_off()
        axs.set_title('Natural Reading ', loc='left')
        fig.savefig(file1)
