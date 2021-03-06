import os.path as op
import numpy as np
import mne
from mayavi import mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up dirs needed for all visualizations
experiment_dir = 'natural_reading/'

noise_cov = 'with_non_reg_noise_cov'
method = 'MNE'
views = ['caudal', 'lateral']
hemis = ['both', 'lh'] # looped together with views, as a tuple

stc_file = op.join(experiment_dir, noise_cov, method, 'average')
stc = mne.read_source_estimate(stc_file, 'fsaverage')

# time_points = [0.090, 0.100, 0.112, 0.122, 0.130, 0.140] # Time points of interest - P1
# time_points = [0.160, 0.170, 0.180, 0.185, 0.190, 0.200, 0.210] # Time points of interest - N1
# time_points = [0.230, 0.240, 0.250, 0.255, 0.260, 0.270, 0.280] # Time points of interest - N2
# time_points = [0.310, 0.320, 0.330, 0.340, 0.345, 0.350, 0.360, 0.370] # Time points of interest - P2
time_points = np.linspace(0.060, 0.380, num=32)
print(time_points)

mlab.options.offscreen = True
for time_point in time_points:
    time = str(int(time_point*1000))
    for view, hemi in zip(views, hemis):
        surfer_kwargs = dict(hemi=hemi, clim=dict(kind='value', lims=[6.5e-12, 7.0e-12, 1.6e-11 ]),
                             views=view, time_unit='s', smoothing_steps=5, colorbar = True, size = 700, time_label = None)
        file1 = 'visualizations/fixed_gaze_source_nr_'+view+'_t'+time+'.png'
        fig, axs = plt.subplots(1,1, sharex=True, sharey = True, tight_layout=True, figsize = (12.8, 9.6))
        fig.suptitle('Time point: ' + time + ' ms')
        brain = stc.plot(initial_time = time_point, **surfer_kwargs)
        img = mlab.screenshot(antialiased = True)
        mlab.close()
        axs.imshow(img)
        axs.set_axis_off()
        axs.set_title('Natural Reading Average', loc='left')
        fig.savefig(file1)
