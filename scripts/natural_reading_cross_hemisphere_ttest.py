import os.path as op
from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from scipy import stats as stats
import mne
from mne.datasets import fetch_fsaverage
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

data_folder = 'natural_reading'
src_fname = '~/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-ico-5-src.fif'
noise_cov = 'with_non_reg_noise_cov'
out_folder = op.join(data_folder, 'results', noise_cov)
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

method = 'MNE'
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
n_subjects = len(subjects)

## Select preferred time window
tmin = 0.130
tmax = 0.155
tmin_ms = tmin * 1000
tmax_ms = tmax * 1000

## Output image file name based on method and time window
image_file = op.join(out_folder, method + '_left_right_' + str(int(tmin_ms)) + '_' + str(int(tmax_ms)) + '.png')

## Dummy variables for arrays built in the for loop
stc_data = None
stc_xhemi_data = None
tstep = None

src = mne.read_source_spaces(src_fname, verbose=True)
fsave_vertices = [s['vertno'] for s in src]

for subject in subjects:
    stc_file = op.join(data_folder, noise_cov, method, subject)
    stc = mne.read_source_estimate(stc_file, 'fsaverage')
    stc.crop(tmin, tmax)
    tstep = stc.tstep
    ## Morph to fsaverage_sym
    stc = mne.compute_source_morph(stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                                   warn=False,
                                   subjects_dir=subjects_dir).apply(stc)
    # Compute a morph-matrix mapping the right to the left hemisphere,
    # and vice-versa.
    morph = mne.compute_source_morph(stc, 'fsaverage_sym', 'fsaverage_sym',
                                    spacing=stc.vertices, warn=False,
                                    subjects_dir=subjects_dir, xhemi=True,
                                    verbose='error')  # creating morph map
    stc_xhemi = morph.apply(stc)
    # noinspection PyInterpreter
    if np.all(stc_data) == None:
        stc_data = stc.data
        stc_xhemi_data = stc_xhemi.data
    else:
        stc_data = np.dstack((stc_data, stc.data))
        stc_xhemi_data = np.dstack((stc_xhemi_data, stc_xhemi.data))

X = stc_data[:, :, :] - stc_xhemi_data[:,:,:]

print('Computing connectivity.')
connectivity = mne.spatial_src_connectivity(src, verbose=True)

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = np.transpose(X, [2, 1, 0])

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_permutations=100, n_jobs=1,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)

stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh = 0.05, tstep=tstep,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

brain = stc_all_cluster_vis.plot(
    hemi='lh', views=['lateral', 'caudal', 'ventral', 'medial'], subjects_dir=subjects_dir,
    title='Left - Right', size=(1200, 1200),
    smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 20]), time_label = str(tmin_ms) + ' - ' + str(tmax_ms) + ' ms')
brain.save_image(image_file)
mlab.show()
