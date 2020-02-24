import os.path as op
import numpy as np
import mne
from mayavi import mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up dirs needed for all visualizations
experiment_dir = op.join('D:\\', 'Zsuzsa', 'source_localization', 'fixed_gaze')
conditions = ['phase_rand', 'normal', 'armenian']
cond_names = {'phase_rand':'Random', 'armenian':'Armenian', 'normal':'Normal'}
chs = ['PO7', 'PO8', 'PPO9h', 'PPO10h']
# Read average evoked files from all three conditions
average_evoked = []
for condition in conditions:
    evoked_dir = op.join(experiment_dir, condition, 'evoked')
    evoked_file =  op.join(evoked_dir, 'average_' + condition + '-ave.fif')
    average_evoked.append(mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False, verbose=False).crop(0.05,0.450))

red_patch = mpatches.Patch(color='red', label='normal')
black_patch = mpatches.Patch(color='black', label='phase_rand')
yellow_patch = mpatches.Patch(color='yellow', label='armenian')
channels = []
times = average_evoked[0].times
symbols = [['k-','k--', 'k-.', 'k:'],['y-', 'y--', 'y-.', 'y:'], ['r-','r--', 'r-.', 'r:']]
for i in range(len(conditions)):
    channels.append((average_evoked[i].pick_channels(chs)).data)

    for ch in range(len(chs)):
        plt.plot(times, channels[i][ch][0:] * 1000000, symbols[i][ch], alpha=0.8, label= cond_names[conditions[i]] + ' ' + chs[ch])
plt.legend(loc='lower right', fontsize = 12)
plt.grid(b=True, which='major', axis='both')
plt.xlabel('time (s)', fontsize=12)
plt.ylabel('\u03BCV', fontsize=14)
plt.show()
