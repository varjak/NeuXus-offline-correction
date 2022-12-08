import lib.eeglab as eeglab
import lib.correct as correct
import lib.plot as plot
import pandas as pd
from fun.find_files import find_files

dset = ''
lfol = 'data/'
sfol = 'res/'
pairs = {'set': [dset], 'fol': [lfol], 'sub': [0], 'ses': [0], 'run': [1], 'art': ['ga']}
lchunk = [50, 200, 250]
ecg_chans = ['ECG', 'EKG']
eeg_chan = 'C3'
tr = 1.260
dfs = 250
make_plot = True
save_data = False

# Find files
files = find_files(pairs)

for f, file in enumerate(files):

    print('file ', f + 1, '/', len(files), ': ', file['filename'])

    # Load file
    EEG = eeglab.tools.load(file['set'] + file['fol'] + file['filename'] + '.mat')
    # EEG = eeglab.tools.trim(EEG, event1='R128', event2='R128', timeshifts=[-10, 0])  # timeshifts=[-10, 2] |
    data, times, fs, chans, types, latencies = eeglab.tools.unpack(EEG)
    df = pd.DataFrame(data.transpose(), index=times[0, :], columns=chans)
    df_markers = pd.DataFrame(types, index=[times[0, lat] for lat in latencies], columns=['marker'])
    ecg_chan = [chan for chan in chans if chan.upper() in [ecg_chan.upper() for ecg_chan in ecg_chans]][0]
    ecg_chn = chans.index(ecg_chan)
    dr = int(fs / dfs)

    # Instantiating classes
    # corrector_ga = correct.GA(df, True, 'S  1', min_wins=7, max_wins=30)  # If using data/sub-00_ses-00_run-01_art-ga.mat
    corrector_ga = correct.GA(df, True, 'R128', min_wins=7, max_wins=30)  # minwins=7, maxwins=20 | minwins=2, maxwins=3

    if make_plot:
        plotter = plot.Plot(xdur=2, xmargin=1, slide='un', lines=[{'subplot': 'C3', 'ylim': [-10, 10], 'name': 'un', 'fs': fs, 'col': 'blue', 'label': 'Unc.'},
                                                                  {'subplot': 'C3', 'ylim': [-10, 10], 'name': 'ga', 'fs': fs, 'col': 'm', 'label': 'GA-corrected'},
                                                                  {'subplot': 'C3', 'ylim': [-10, 10], 'name': 'start_ga', 'fs': fs, 'col': 'm', 'label': 'Start of GA subtraction'}])

        plot_stride = 10
        start_plot = True

    chunk_list_stack_ga = []
    chunk_list_stack_marker_ga = []

    c = 0
    lim1 = 0
    lim2 = lchunk[0]

    while lim2 < len(df):
        chunk_list = [df.iloc[lim1:lim2].copy(deep=True)]
        chunk_list_marker = [df_markers.loc[chunk_list[0].index[0]:chunk_list[0].index[-1]]]
        chunk_list_ga, chunk_list_marker_ga = corrector_ga.update(chunk_list, chunk_list_marker)

        if make_plot:
            plotter.update('un', chunk_list, eeg_chan)
            plotter.update('ga', chunk_list_ga, eeg_chan)
            plotter.update_marker_lines('start_ga', chunk_list_marker_ga, ['Start of GA subtraction'])

            if c % plot_stride == 0:
                plotter.slide()
                plotter.draw()

        c += 1
        lim1 = lim2
        lim2 = lim2 + lchunk[c % len(lchunk)]

        chunk_list_stack_ga.extend(chunk_list_ga)
        chunk_list_stack_marker_ga.extend(chunk_list_marker_ga)

    # Make run-length dataframes
    df_ga = pd.concat(chunk_list_stack_ga)
    df_markers_ga = pd.concat(chunk_list_stack_marker_ga)

    # Add GA correction events
    start_of_ga_building_time = df_markers_ga.index[df_markers_ga['marker'] == 'Start of GA building']
    start_of_ga_subtraction_time = df_markers_ga.index[df_markers_ga['marker'] == 'Start of GA subtraction']

    start_of_ga_building_latency = [eeglab.tools.find_closest_index(df_ga.index, start_of_ga_building_time)]
    types.extend(['Start of GA building'])
    latencies.extend(start_of_ga_building_latency)

    start_of_ga_subtraction_latency = [eeglab.tools.find_closest_index(df_ga.index, start_of_ga_subtraction_time)]
    types.extend(['Start of GA subtraction'])
    latencies.extend(start_of_ga_subtraction_latency)

    # Save
    EEG = eeglab.tools.pack(file['filename'] + '_cor-gaNeXoff', df_ga.transpose().to_numpy(), df_ga.index.to_numpy(), fs, chans, types, latencies)
    eeglab.tools.save(file['set'] + sfol + file['filename'] + '_cor-gaNeXoff' + '.mat', EEG) if save_data else None
