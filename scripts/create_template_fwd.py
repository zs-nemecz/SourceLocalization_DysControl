### Create forward model for source localization with template anatomy and standard EEG electrode locations
### Steps
### 1. Download/read template MRI (FSAverage)
### 2. Read EEG data
### 3. Create montage based on standard 10-5 electrode locations and ActiCap naming
### 4. Set reference and montage
### 5. Make and save forward model

import os.path as op
from mayavi import mlab
import mne
from mne.datasets import fetch_fsaverage

### 1. Download/read template MRI (FSAverage)
# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
# The files found in:
subject = 'fsaverage'
trans = op.join(fs_dir, 'bem', 'fsaverage-trans.fif')
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

### 2. Read EEG data for any random subject
data_folder = '../control-experiment/results/preproc_EEG/fixed_gaze/'
subject = '009833'
data_file = op.join(data_folder, subject, 'nat_ica', 'erp_eeg_normal_v2_avgref.set')
epochs = mne.io.read_epochs_eeglab(data_file)

### 3. Create montage based on standard 10-5 electrode locations and ActiCap naming
#Take the name 'I1' and 'I2' instead of 'O9' and 'O10'
channels = epochs.ch_names.copy()
for standard_channel, acti_channel in zip(['I1', 'I2'], ['O9', 'O10']):
    channels.remove(acti_channel)
    channels.append(standard_channel)
# Read montage
standard_montage = mne.channels.read_montage('standard_1005', ch_names = channels, transform=True)
# Name 'I1' and 'I2' back to 'O9' and 'O10', so that there is no naming inconsitency with epoch channels
ch_names = list(standard_montage.ch_names)
for standard_channel, acti_channel in zip(['I1', 'I2'], ['O9', 'O10']):
    ch_names = [x if x!=standard_channel else acti_channel for x in ch_names]
standard_montage.ch_names = tuple(ch_names)

### 4. Set montage
epochs.set_montage(standard_montage)
#Plot alignment
standard_figure = mne.viz.plot_alignment(
    epochs.info, src=src, eeg=['original', 'projected'], trans=trans, dig='fiducials')
mlab.show()

### 5. Make and save forward model
fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
mne.write_forward_solution('template-fwd.fif',fwd, overwrite=True)
