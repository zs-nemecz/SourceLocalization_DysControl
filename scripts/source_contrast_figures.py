import os.path as op
import numpy as np
import mne
from mayavi import mlab
import matplotlib.pyplot as plt

# Contrast function
def create_contrast_stc(stc1, stc2, tmin = None, tmax = None):

    # Read files if input is string
    if isinstance(stc1, str):
        stc1 = mne.read_source_estimate(stc1, 'fsaverage')
    if isinstance(stc2, str):
        stc2 = mne.read_source_estimate(stc2, 'fsaverage')
    # Crop
    stc1.crop(tmin, tmax)
    stc2.crop(tmin, tmax)
    # Create contrast
    contrast = stc1.data - stc2.data
    stc1.data = contrast
    # Return stc with contrast data
    return stc1

# Set up dirs needed for all visualizations
experiment_dir = 'fixed_gaze/'
conditions = ['phase_rand', 'armenian', 'normal']
# STC files
contrast_conditions = [conditions[1], conditions[0]]

method = 'MNE'
noise_cov = 'with_non_reg_noise_cov'
cfile0 = op.join(experiment_dir, contrast_conditions[0], noise_cov, method, contrast_conditions[0] + '_average')
cfile1 = op.join(experiment_dir, contrast_conditions[1], noise_cov, method, contrast_conditions[1] + '_average')

# time_points = [0.260, 0.345] # Time points of interest
# time_points = [0.090, 0.100, 0.112, 0.122, 0.130, 0.140] # Time points of interest - P1
time_points = [0.160, 0.170, 0.180, 0.185, 0.190, 0.200, 0.210] # Time points of interest - N1
# time_points = [0.230, 0.240, 0.250, 0.255, 0.260, 0.270, 0.280] # Time points of interest - N2
# time_points = [0.310, 0.320, 0.330, 0.340, 0.345, 0.350, 0.360, 0.370] # Time points of interest - P2

mlab.options.offscreen = True
contrast = create_contrast_stc(cfile0, cfile1) # add conditions to compare here


for time_point in time_points:
    time = str(int(time_point*1000))
    for view, hemi in zip(['caudal', 'ventral', 'lateral', 'lateral'],['both', 'both','lh', 'rh']):
        surfer_kwargs = dict(hemi=hemi, views=view, #clim=dict(kind='value', lims=[0.5e-12, 0.5e-12, 6.5e-12]),
                        time_unit='s', smoothing_steps=5, colorbar = True, size=(1200, 1200), time_label = None)
        file1 = 'visualizations/fixed_gaze_source_contrast_' + contrast_conditions[0] +'-'+ contrast_conditions[1]+ '_' + view +'_' + hemi + '_t'+time+'.png'
        fig, axs = plt.subplots(1,1, sharex=True, sharey = True, tight_layout=True, figsize = (12.8, 9.6))
        fig.suptitle('Time point: ' + time + ' ms')
        brain = contrast.plot(initial_time = time_point, **surfer_kwargs)
        img = mlab.screenshot(antialiased = True)
        mlab.close()
        axs.imshow(img)
        axs.set_axis_off()
        axs.set_title(contrast_conditions[0].capitalize() + '-' + contrast_conditions[1].capitalize(), loc='left')
        fig.savefig(file1)
