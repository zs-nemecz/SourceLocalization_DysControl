### Modified original MNE function to plot source estimates together with evoked data in sensor space
import os.path as op
import mne
import matplotlib as plt
import mayavi as mlab

# Set up dirs needed for all visualizations
experiment_dir = 'fixed_gaze/'
conditions = ['phase_rand', 'armenian', 'normal']

average_evoked = []
for condition in conditions:
    evoked_dir = op.join(experiment_dir, condition, 'evoked')
    evoked_file =  op.join(evoked_dir, 'average_' + condition + '-ave.fif')
    average_evoked.append(mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False).crop(-0.050,None))

noise_cov = 'with_non_reg_noise_cov'
method = 'MNE'
average_stc = []
for condition in conditions:
    average_stc_file = op.join(experiment_dir, condition, noise_cov, method, condition + '_average')
    average_stc.append(mne.read_source_estimate(average_stc_file, 'fsaverage'))

fig = mne.viz.plot_evoked(average_evoked[2], stc=average_stc[2], spatial_colors=True)
