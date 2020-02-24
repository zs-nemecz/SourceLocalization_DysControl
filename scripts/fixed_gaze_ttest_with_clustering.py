import time
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

## Set parameters of interest here##
############################################
conditions = ['normal', 'armenian']
noise_cov= 'merged_nc'
inverse_model = 'inverse_model_auto'
method = 'dSPM'
tmin = 0.00
tmax = 0.450
p_threshold = 0.05
n_permutations = 1000
###########################################

title = conditions[0].capitalize() + ' > ' + conditions[1].capitalize()
time_label = str(int(tmin * 1000)) + '-' + str(int(tmax * 1000)) + 'ms'
## subject excluded '181585'
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
n_subjects = len(subjects)
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = op.dirname(fs_dir)
data_folder = 'fixed_gaze/'
src_fname = '/home/zsuzsanna/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-ico-5-src.fif'
out_folder = op.join(data_folder, 'results', noise_cov, method)
stc_save = op.join(out_folder, conditions[0] + '_' + conditions[1] + '_' + time_label + '_p-' + str(p_threshold) + '_nperm-' + str(int(n_permutations)))

print('Number of participants included: ', n_subjects)
src = mne.read_source_spaces(src_fname, verbose=True)
fsave_vertices = [s['vertno'] for s in src]
condition0_stc_data = None
condition1_stc_data = None

tstep = None
for condition in conditions:
    stc_data = None
    for subject in subjects:
        if inverse_model == 'inverse_model_auto':
            stc_file = op.join(data_folder, condition, noise_cov, method, subject)
        else:
            stc_file = op.join(data_folder, condition, noise_cov, method, subject + '_fixed_inverse')
        stc = mne.read_source_estimate(stc_file, subject)
        stc.crop(tmin,tmax)
        tstep = stc.tstep
        if np.all(stc_data) == None:
            stc_data = stc.data
        else:
            stc_data = np.dstack((stc_data, stc.data))

    if condition == conditions[0]:
        condition0_stc_data = stc_data
    else:
        condition1_stc_data = stc_data

# normal_stc_data = np.abs(normal_stc_data)
# phase_rand_stc_data = np.abs(phase_rand_stc_data)

X = condition0_stc_data[:, :, :] - condition1_stc_data[:,:,:]
print('Computing connectivity.')
connectivity = mne.spatial_src_connectivity(src, verbose=True)

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = np.transpose(X, [2, 1, 0])

#    Now let's actually do the clustering. This can take a long time...
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
print('Clustering.')
start_time = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(start_time)))
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_permutations=n_permutations, n_jobs=1,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

print('Visualizing clusters.')

#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh = 0.05, tstep=tstep,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

end_time = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(end_time)))
elapsed_time = end_time - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
stc_all_cluster_vis.save(stc_save)
print(stc_save)

#Visualize individual clustres, one by one
for i in good_cluster_inds:
    print(i)
    cluster = T_obs, clusters[i:i+1], cluster_p_values[i:i+1], H0
    stc_cluster = summarize_clusters_stc(cluster, p_thresh = 0.05, tstep=tstep, vertices=fsave_vertices,subject='fsaverage')
    clu_name = str(i)
    fig = stc_cluster.plot(hemi='split', views=['lateral', 'caudal', 'ventral', 'medial'], subjects_dir=subjects_dir,
        title=title, size=(600, 900), time_label = time_label,
        smoothing_steps=5)
    mlab.show()


## All clusters with 'split' view
# blue blobs are for condition A < condition B, red for A > B
brain = stc_all_cluster_vis.plot(
    hemi='split', views=['lateral', 'caudal', 'ventral'], subjects_dir=subjects_dir,
    title=title, size=(600, 900),
    smoothing_steps=5, time_label = time_label)

mlab.show()
del brain

## All clusters with 'both' view
# blue blobs are for condition A < condition B, red for A > B
brain = stc_all_cluster_vis.plot(
    hemi='both', views=['lateral', 'caudal', 'ventral'], subjects_dir=subjects_dir,
    title=title, size=(600, 900),
    smoothing_steps=5, time_label = time_label)
brain.add_text(0.05, 0.85, title + '. Cluster p < 0.05. Method: ' + method, 'title', font_size = 12)
mlab.show()
