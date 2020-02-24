import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set up dirs needed for all visualizations
data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
conditions = ['phase_rand', 'armenian', 'normal']
subjects = ['822816']
print('Number of participants: ', len(subjects))
conditions = ['phase_rand', 'armenian', 'normal']
task = 'fixed_gaze'

for num, subject in enumerate(subjects):
    print('\n===============================================================================\n')
    print('Subject number: ', num)
    print(subject)
    print('\n===============================================================================\n')
    if task == 'fixed_gaze':
        fig_ev, axs = plt.subplots(3,1, figsize = (12.0, 16.0), sharey=True, tight_layout=False)
        for row, condition in enumerate(conditions):
            evoked_dir = op.join(task, condition, 'evoked')
            evoked_file =  op.join(evoked_dir, subject + '_' + condition + '-ave.fif')
            evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False)
            fig = evoked.plot(show=False, spatial_colors=True, axes = axs[row])
            axs[row].set_title(condition.capitalize(), x=0.8)
            if row < 2:
                axs[row].set_xlabel('')
    else:
        evoked_dir = op.join(task, 'evoked')
        evoked_file =  op.join(evoked_dir, subject + '-ave.fif')
        fig_ev, axs = plt.subplots(1,1, figsize = (12.0, 6.0), tight_layout=False)
        evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False)
        evoked.plot(show=False, spatial_colors=True, axes = axs)

    title =  'Subject: ' + subject + '\nTask: ' + task
    fig_ev.suptitle(title, x=0.5,y=0.99,weight='demibold')
    plt.subplots_adjust(top=0.90, hspace=0.2)
    plt.show()
    file =  op.join(task,'plots', subject + '_evoked.png')
    fig_ev.savefig(file, bbox_inches='tight')
    plt.close(fig_ev)
    del fig_ev,axs
