import os.path as op
import numpy as np
import mne
from mayavi import mlab
import matplotlib.pyplot as plt

methods = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
conditions = ['armenian', 'normal', 'phase_rand']

for condition in conditions:
    print('................................................\n{}\n................................................'.format(condition))
    evoked_folder = op.join('/home/zsuzsanna/Documents/MTA/source_localization/fixed_gaze/',condition, 'evoked' )
    evoked_file =  op.join(evoked_folder, 'average_' + condition +'-ave.fif')
    average_evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False, verbose=False)
    print('******************\nEVOKED {}\n******************'.format(condition))
    p1_channel, p1_time, p1_amplitude = average_evoked.get_peak(mode='pos', tmin = 0.10, tmax = 0.150, return_amplitude=True)
    n1_channel, n1_time, n1_amplitude = average_evoked.get_peak(mode='neg', tmin = 0.15, tmax = 0.200, return_amplitude=True)
    n2_channel, n2_time, n2_amplitude = average_evoked.get_peak(mode='neg', tmin = 0.23, tmax = 0.280, return_amplitude=True)
    p2a_channel, p2a_time, p2a_amplitude = average_evoked.get_peak(mode='pos', tmin = 0.220, tmax = 0.270, return_amplitude=True)
    p2b_channel, p2b_time, p2b_amplitude = average_evoked.get_peak(mode='pos', tmin = 0.30, tmax = 0.380, return_amplitude=True)
    print('P1 \tchannel \tlatency \taplitude\n  \t{} \t\t{} \t{}'.format(p1_channel,p1_time, p1_amplitude))
    print('N1 \tchannel \tlatency \taplitude\n  \t{} \t\t{} \t{}'.format(n1_channel,n1_time, n1_amplitude))
    print('N2 \tchannel \tlatency \taplitude\n  \t{} \t\t{} \t{}'.format(n2_channel,n2_time, n2_amplitude))
    print('P2A \tchannel \tlatency \taplitude\n  \t{} \t\t{} \t{}'.format(p2a_channel,p2a_time, p2a_amplitude))
    print('P2B \tchannel \tlatency \taplitude\n  \t{} \t\t{} \t{}'.format(p2b_channel,p2b_time, p2b_amplitude))
    # for method in methods:
    #     stc_folder = op.join('/home/zsuzsanna/Documents/MTA/source_localization/fixed_gaze/',condition, 'merged_nc', method)
    #     stc_file = op.join(stc_folder, condition + '_average')
    #     averaged_stc = mne.read_source_estimate(stc_file)
    #     ## Print out information about peaks
    #     print('******************\nSTC {}\n******************'.format(method))
    #     p1_pos, p1_latency = averaged_stc.get_peak(tmin = 0.09, tmax = 0.125)
    #     n1_pos, n1_latency = averaged_stc.get_peak(tmin = 0.14, tmax = 0.20)
    #     n2_pos, n2_latency = averaged_stc.get_peak(tmin = 0.21, tmax = 0.24)
    #     p2_pos, p2_latency = averaged_stc.get_peak(tmin = 0.24, tmax = 0.40)
    #     print('P1 \tposition \tlatency  \n\t{} \t\t{}'.format(p1_pos,p1_latency))
    #     print('N1 \tposition \tlatency  \n\t{} \t\t{}'.format(n1_pos,n1_latency))
    #     print('N2 \tposition \tlatency  \n\t{} \t\t{}'.format(n2_pos,n2_latency))
    #     print('P2 \tposition \tlatency  \n\t{} \t\t{}'.format(p2_pos,p2_latency))
