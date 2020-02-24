import os.path as op
import numpy as np
import mne
import pickle
from mne.datasets import fetch_fsaverage
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)
from scipy import stats as stats
import time
import logging

logging.basicConfig(filename='D:/Zsuzsa/source_localization/fixed_gaze/results/ch_clu_log.txt', level=logging.INFO,
                    filemode='a')

# needs to be tested (src.pop() and 'mask' as out_type)
def ttest_clustering(X, src, p_threshold, n_subjects, n_permutations, cluster_p, log=True):
    if log:
        cluster_start = time.time()
    fsave_vertices = [s['vertno'] for s in src]
    print('Computing connectivity.')
    connectivity = mne.spatial_src_connectivity(src, verbose=True)

    #    Note that X needs to be a multi-dimensional array of shape
    #    samples (subjects) x time x space, so we permute dimensions
    X = np.transpose(X, [2, 1, 0])

    #    Now let's actually do the clustering. This can take a long time...
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)

    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_permutations=n_permutations, n_jobs=1,
                                           threshold=t_threshold, buffer_size=None, verbose=True)
    #    Now select the clusters that are sig. at p < 0.05 (note that this value
    #    is multiple-comparisons corrected).
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    ok_cluster_inds = np.where(cluster_p_values < 0.1)[0]
    print('Good cluster inds:')
    print(good_cluster_inds)
    logging.info('Clusters with p<0.05: {}'.format(good_cluster_inds))
    logging.info('Clusters with p<0.1: {}'.format(ok_cluster_inds))
    stc_all_cluster_vis = None
    try:
        stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh=cluster_p,
                                                     tstep=tstep, vertices=fsave_vertices, subject='fsaverage')
    except Exception as e:
        print("Exception occurred: ", e)
    if log:
        cluster_finish = time.time()
        elapsed = cluster_finish - cluster_start
        logging.info('Clustering finished in {}'. format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    return stc_all_cluster_vis, clu


data_folder = op.join('D:\\', 'Zsuzsa', 'source_localization', 'fixed_gaze')
src_fname = "D:\\Zsuzsa\\mne_data\\MNE-fsaverage-data\\fsaverage\\bem\\fsaverage-ico-5-src.fif"
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183',
            '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103',
            '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207',
            '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
n_subjects = len(subjects)
logging.info('Number of subjects included: {}'.format(n_subjects))

## Set parameters of interest here##
############################################
conditions = ['normal', 'armenian', 'phase_rand']
contrasts = [['normal', 'phase_rand'], ['normal', 'armenian'], ['armenian', 'phase_rand']]
methods = ['MNE', 'dSPM', 'eLORETA', 'sLORETA'] # 'MNE', 'dSPM', 'eLORETA', 'sLORETA'
noise_cov = 'merged_nc'
tmins = [0.0]  # [0.102, 0.102, 0.104, 0.112, 0.170, 0.170, 0.176, 0.178, 0.240, 0.244, 0.326, 0.326, 0.328, 0.332]
tmaxs = [0.450]  # [0.122, 0.132, 0.124, 0.132, 0.190, 0.198, 0.196, 0.198, 0.260, 0.264, 0.346, 0.352, 0.348, 0.352]
inverse_model = 'inverse_model_auto'

## cluster analysis
n_permutations = 1000
src = mne.read_source_spaces(src_fname, verbose=True)
src.pop() #use only left hemisphere
p_threshold = 0.05
cluster_p = 0.05
lh_vertices = 10242

start_time = time.time()
logging.info(time.strftime("%H:%M:%S", time.gmtime(start_time)))
for tmin, tmax in zip(tmins, tmaxs):
    time_label = str(int(tmin * 1000)) + '-' + str(int(tmax * 1000)) + 'ms'
    for method in methods:
        logging.info(method)
        cross_hemi = {}
        out_folder = op.join(data_folder, 'results', 'merged_nc', 'cross_hemi', method, 'clusters_timecourse')
        for condition in conditions:
            logging.info(condition)
            stc_data = None
            file = op.join(out_folder, condition + '_' + time_label + '_p-' + str(p_threshold) + '_nperm-' + str(int(n_permutations)))
            pickle_fname = op.join(out_folder, condition + '_' + time_label + '_nperm-' + str(int(n_permutations)) +'_all_clu.pkl')

            for subject in subjects:
                stc_file = op.join(data_folder, condition, noise_cov, method, subject)
                stc = mne.read_source_estimate(stc_file, 'fsaverage')
                stc.crop(tmin, tmax)

                tstep = stc.tstep
                ## Morph to fsaverage_sym
                stc = mne.compute_source_morph(stc, subject_to='fsaverage_sym', smooth=5,
                                               warn=False,
                                               subjects_dir=subjects_dir).apply(stc)
                # Compute a morph-matrix mapping the right to the left hemisphere,
                # and vice-versa.
                morph = mne.compute_source_morph(stc, 'fsaverage_sym', 'fsaverage_sym',
                                                 spacing=stc.vertices, warn=False,
                                                 subjects_dir=subjects_dir, xhemi=True,
                                                 verbose='error')  # creating morph map
                stc_xhemi = morph.apply(stc)
                if np.all(stc_data) is None:
                    stc_data = stc.data
                    stc_xhemi_data = stc_xhemi.data
                else:
                    stc_data = np.dstack((stc_data, stc.data))
                    stc_xhemi_data = np.dstack((stc_xhemi_data, stc_xhemi.data))
            logging.info('Collected {} stc data files from {} condition'.format(stc_data.shape[2], condition))
            X = stc_data[:, :, :] - stc_xhemi_data[:, :, :]
            X = X[:lh_vertices, :, :]
            cross_hemi[condition] = X

            stc, clu = ttest_clustering(X, src, p_threshold, n_subjects, n_permutations, cluster_p)
            #pickle clu object and save stc with good clusters (p<0.05)
            with open(pickle_fname, 'wb') as f:
                pickle.dump(clu, f)
            if stc is None:
                print('No clusters within threshold.')
            else:
                stc.save(file)


        logging.info('Collected cross hemi contrasts for {} conditions'.format(len(cross_hemi)))
        logging.info([c for c in cross_hemi])

        for contrast in contrasts:
            print('Comparing {} with {} condition'.format(contrast[0], contrast[1]))
            logging.info('Contrast: {}'.format(contrast))
            file = op.join(out_folder,contrast[0] + '_' + contrast[1] + time_label + '_p-' + str(p_threshold) + '_nperm-' + str(int(n_permutations)))
            pickle_fname = op.join(out_folder,contrast[0] + '_' + contrast[1] + time_label + '_nperm-' + str(
                int(n_permutations)) + '_all_clu.pkl')
            C = cross_hemi[contrast[0]] - cross_hemi[contrast[1]]
            stc, clu = ttest_clustering(C, src, p_threshold, n_subjects, n_permutations, cluster_p)
            with open(pickle_fname, 'wb') as f:
                pickle.dump(clu, f)
            if stc is None:
                print('No clusters within threshold.')
            else:
                stc.save(file)

end_time = time.time()
logging.info(time.strftime("%H:%M:%S", time.gmtime(end_time)))
elapsed_time = end_time - start_time
logging.info('Elapsed time:')
logging.info(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
