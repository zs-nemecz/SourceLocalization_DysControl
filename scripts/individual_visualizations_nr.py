
evoked.plot(time_unit='s', spatial_colors=True)
evoked.plot_topomap(times=np.linspace(0.05, 0.30, 15), ch_type='eeg',
                    time_unit='s')
evoked.plot_white(noise_cov, time_unit='s')
