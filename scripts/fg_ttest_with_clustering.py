##TODO save masks and clu object as is (together with t objects and p values)
## or figure out a usage with summarize_clusters_stc

import os.path as op
import time
import pickle
import mne
import numpy as np
from mne.datasets import fetch_fsaverage
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)
from scipy import stats as stats

import logging
logging.basicConfig(filename='D:/Zsuzsa/source_localization/fixed_gaze/results/clu_log.txt', level=logging.INFO, filemode='a')
logging.info('--------------------------------------------------------------------------------------------------------')
## subject excluded '181585'
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
n_subjects = len(subjects)
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = op.dirname(fs_dir)
data_folder = "D:\\Zsuzsa\\source_localization\\fixed_gaze\\"
src_fname = "D:\\Zsuzsa\\mne_data\\MNE-fsaverage-data\\fsaverage\\bem\\fsaverage-ico-5-src.fif"
print('Number of participants included: ', n_subjects)

## Set parameters of interest here##
############################################
contrasts = [['normal', 'phase_rand'], ['normal', 'armenian'], ['armenian', 'phase_rand']]
noise_cov= 'merged_nc'
inverse_model = 'inverse_model_auto'
methods = ['MNE', 'dSPM', 'eLORETA', 'sLORETA'] # 'MNE', 'dSPM', 'eLORETA', 'sLORETA'
tmins = [0.0] # [0.102, 0.102, 0.104, 0.112, 0.170, 0.170, 0.176, 0.178, 0.240, 0.244, 0.326, 0.326, 0.328, 0.332]
tmaxs = [0.450] # [0.122, 0.132, 0.124, 0.132, 0.190, 0.198, 0.196, 0.198, 0.260, 0.264, 0.346, 0.352, 0.348, 0.352]
p_threshold = 0.05
n_permutations = 1000

###########################################

for tmin, tmax in zip(tmins, tmaxs):
    time_label = str(int(tmin * 1000)) + '-' + str(int(tmax * 1000)) + 'ms'
    print(time_label)
    for method in methods:
        print('Method: ', method)
        print('**********************************************************************************************************************')
        for conditions in contrasts:
            title = conditions[0].capitalize() + ' > ' + conditions[1].capitalize()
            print(title)
            out_folder = op.join(data_folder, 'results', noise_cov, method, 'cluster_timecourse')
            good_clusters_fname = op.join(out_folder, conditions[0] + '_' + conditions[1] + '_' + time_label + '_p-' + str(p_threshold) + '_nperm-' + str(int(n_permutations)))
            pickle_fname = op.join(out_folder, conditions[0] + '_' + conditions[1] + '_' + time_label + '_nperm-' + str(int(n_permutations))+'_all_clu.pkl')
            logging.info(good_clusters_fname)
            print('Out folder: ', out_folder)

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
                                                   threshold=t_threshold, buffer_size=None, #out_type = 'mask',
                                                   verbose=True)
            #    Now select the clusters that are sig. at p < 0.05 (note that this value
            #    is multiple-comparisons corrected).
            good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
            ok_cluster_inds = np.where(cluster_p_values < 0.1)[0]
            print('Good cluster inds:')
            print(good_cluster_inds)
            logging.info('Clusters with p<0.05: {}'.format(good_cluster_inds))
            logging.info('Clusters with p<0.1: {}'.format(ok_cluster_inds))
            with open(pickle_fname, 'wb') as f:
                pickle.dump(clu, f)
            if len(good_cluster_inds) > 0:
                stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh = p_threshold
                                                             , tstep=tstep, vertices=fsave_vertices, subject='fsaverage')
                stc_all_cluster_vis.save(good_clusters_fname)
                print('Saved to ', good_clusters_fname)
            else:
                logging.info('no clusters with p<0.05 found')

            end_time = time.time()
            print(time.strftime("%H:%M:%S", time.gmtime(end_time)))
            elapsed_time = end_time - start_time
            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            print('==========================================================================================================================')
