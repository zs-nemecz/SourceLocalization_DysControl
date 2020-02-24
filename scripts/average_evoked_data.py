### Average evoked data from all subjects
### Modify path based on experiment/condition as necessary
import os.path as op
import mne

evoked_folder = '/home/zsuzsanna/Documents/MTA/source_localization/natural_reading/evoked'
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169', '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
evokeds = []
for subject in subjects:
    evoked_file =  op.join(evoked_folder, subject + '-ave.fif')
    evokeds.append(mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False, verbose=True))

average_evoked = mne.grand_average(evokeds)
av_file = op.join(evoked_folder, 'average-ave.fif')
average_evoked.save(av_file)
