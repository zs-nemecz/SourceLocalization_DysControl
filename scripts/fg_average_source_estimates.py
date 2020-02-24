### Read Source Estimate data arrays from each subject, stack them on each other and calculate average
### Done for each method
### Condition is asked as input in the beginning

import os.path as op
import numpy as np
import mne

conditions = ['armenian', 'normal', 'phase_rand']

inverse_solvers = ['MNE', 'dSPM', 'sLORETA', 'eLORETA']
## subject excluded: '181585'
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
print('Number of subjects included: ', len(subjects))
for condition in conditions:
    data_folder = '/home/zsuzsanna/Documents/MTA/source_localization/fixed_gaze/' + condition + '/merged_nc'
    for method in inverse_solvers:
        print('===========================================================================================================')
        print('METHOD: ', method)
        stc_data = None
        for subject in subjects:
            stc_file = op.join(data_folder, method, subject)
            ## read data
            stc = mne.read_source_estimate(stc_file, condition + '_average')
            ## stack the stc.data arrays on each other
            if np.all(stc_data) == None:
                stc_data = stc.data
            else:
                stc_data = np.dstack((stc_data, stc.data))

        print('Calculating average for method ', method)
        average = np.average(stc_data, axis = 2)
        stc.data = average
        mean_stc_file = op.join(data_folder, method, condition + '_average')
        print('Saving..')
        stc.save(mean_stc_file)
        del stc, average
