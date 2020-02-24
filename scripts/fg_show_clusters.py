import os.path as op
import numpy as np
import mne
from mayavi import mlab


file = '/home/zsuzsanna/Documents/MTA/source_localization/fixed_gaze/results/merged_nc/dSPM/normal_armenian_0-450ms_p-0.05_nperm-1000'
stc = mne.read_source_estimate(file, 'fsaverage')
pos_lims=[3, 6, 40]
surfer_kwargs = dict(hemi='split', views=['lateral', 'caudal', 'ventral'], time_unit='s', smoothing_steps=5, colorbar = True, size = 800,
                # clim=dict(kind='value', pos_lims=[80.0, 100.0, 216.0]),
                time_viewer=True, time_label = '0-450 ms')
brain = stc.plot(**surfer_kwargs)

mlab.show()
mlab.close()
