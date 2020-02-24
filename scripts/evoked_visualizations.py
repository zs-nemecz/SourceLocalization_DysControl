import os.path as op
import numpy as np
import mne
from mayavi import mlab
import matplotlib.pyplot as plt

evoked_folder = '/home/zsuzsanna/Documents/MTA/source_localization/fixed_gaze/armenian/evoked/'
evoked_file =  op.join(evoked_folder, 'average_armenian-ave.fif')
average_evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False, verbose=True)

## Print out information about peaks
p1_channel, p1_time, p1_amplitude = average_evoked.get_peak(mode='pos', tmin = 0.1, tmax = 0.13, return_amplitude=True)
n1_channel, n1_time, n1_amplitude = average_evoked.get_peak(mode='neg', tmin = 0.14, tmax = 0.16, return_amplitude=True)
n2_channel, n2_time, n2_amplitude = average_evoked.get_peak(mode='neg', tmin = 0.18, tmax = 0.20, return_amplitude=True)
print('P1 \tchannel \tlatency \taplitude\n  \t{} \t\t{} \t{}'.format(p1_channel,p1_time, p1_amplitude))
print('N1 \tchannel \tlatency \taplitude\n  \t{} \t\t{} \t{}'.format(n1_channel,n1_time, n1_amplitude))
print('N2 \tchannel \tlatency \taplitude\n  \t{} \t\t{} \t{}'.format(n2_channel,n2_time, n2_amplitude))

## Create and save figures
fig1 = average_evoked.plot(time_unit='s', spatial_colors=True)
fig2 = average_evoked.plot_topomap(times=np.linspace(0.08, 0.3, 20), ch_type='eeg',
                    time_unit='s')

fig1.savefig(evoked_folder + 'average_evoked.png')
fig2.savefig(evoked_folder + 'average_topomap.png')

## Play animation from 0ms to end
times_list = average_evoked.times[average_evoked.times > -0.0001]
times = np.array(times_list)
average_evoked.animate_topomap(butterfly=True, time_unit='s', times=times, frame_rate = 24)

mlab.show()
