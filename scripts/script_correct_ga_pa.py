import lib.eeglab as eeglab
import lib.correct as correct
import lib.filter as filter
import lib.plot as plot
import pandas as pd
from fun.find_files import find_files


dset = 'C:/Users/guta_/Desktop/'
lfol = 'Data Analysis/Data/NeurAugVR/Acquired/'
sfol = 'Experiment preparation/Problems/Clarify NeuXus/res/'
pairs = {'set': [dset], 'fol': [lfol], 'sub': [1], 'ses': [2], 'run': [4], 'mri': ['vary'], 'task': ['neurowMIMO']}

mfil = 'models/weights-input-500.pkl'  # win_len = 500
lchunk = [50, 200, 250]   # [50, 200, 250]
ecg_chans = ['ECG', 'EKG']
eeg_chan = 'C3'
tr = 1.260
dfs = 250
stride = 50
make_plot = True
save_data = False

# Find files
files = find_files(pairs)

for f, file in enumerate(files):

    print('file ', f + 1, '/', len(files), ': ', file['filename'])

    # Load file
    EEG = eeglab.tools.load(file['set'] + file['fol'] + file['filename'] + '.mat')
    # EEG = eeglab.tools.trim(EEG, event1='R128', event2='R128', timeshifts=[-2, 0])
    data, times, fs, chans, types, latencies = eeglab.tools.unpack(EEG)
    df = pd.DataFrame(data.transpose(), index=times[0, :], columns=chans)
    df_markers = pd.DataFrame(types, index=[times[0, lat] for lat in latencies], columns=['marker'])
    ecg_chan = [chan for chan in chans if chan.upper() in [ecg_chan.upper() for ecg_chan in ecg_chans]][0]
    ecg_chn = chans.index(ecg_chan)
    dr = int(fs / dfs)

    # Instantiating classes
    corrector_ga = correct.GA(df, True, 'R128', min_wins=7, max_wins=30)  # minwins=7, maxwins=20 | minwins=2, maxwins=3
    downsampler = filter.Downsample(fs, dfs, chans)
    # selector = filter.Select()
    # butter_filter = filter.Butter(dfs, [0.5, 30], 1, order=4)
    corrector_pa = correct.PA(df, True, dfs, mfil, start_marker='Start of GA subtraction', stride=stride, min_wins=10, max_wins=20, min_hc=0.4, max_hc=1.5, margin=0.1,numba=True, filter_ecg=True)

    if make_plot:
        # plotter = plot.Plot(fs, dfs, win_range=2)
        plot_stride = 10
        start_plot = True
        plotter = plot.Plot(xdur=2, xmargin=1, slide='fi', lines=[{'subplot': 'C3', 'ylim': [-1000, 1000], 'name': 'ga', 'fs': fs, 'col': 'm', 'label': 'GA-corrected'},
                                                                  {'subplot': 'C3', 'ylim': [-100, 100], 'name': 'ds', 'fs': dfs, 'col': 'red', 'label': 'Downsampled'},
                                                                  {'subplot': 'C3', 'ylim': [-100, 100], 'name':'pa', 'fs': dfs,'col':'orange', 'label':'PA-corrected'},
                                                                  {'subplot': 'C3', 'ylim': [-100, 100], 'name': 'start_pa', 'fs': dfs, 'col': 'orange', 'label': 'Start of PA subtraction'},
                                                                  {'subplot': 'C3', 'ylim': [-100, 100], 'name': 'det1', 'fs': dfs, 'col': 'k', 'label': 'Detection'},
                                                                  {'subplot': 'C3', 'ylim': [-100, 100], 'name': 'hold1', 'fs': dfs, 'col': 'g', 'label': 'Hold limit'},
                                                                  {'subplot': 'C3', 'ylim': [-100, 100], 'name': 'margin1', 'fs': dfs, 'col': 'y', 'label': 'Margin'},
                                                                  {'subplot': 'ECG', 'ylim': [-2000, 2000], 'name':'fi', 'fs': dfs, 'col':'red', 'label':'Downsampled'},
                                                                  {'subplot': 'ECG', 'ylim': [-2000, 2000], 'name': 'r', 'fs': dfs, 'col': 'k', 'label': 'R peaks', 'marker': 'x', 'ls': ''},
                                                                  {'subplot': 'ECG', 'ylim': [-2000, 2000], 'name': 'rm', 'fs': dfs, 'col': 'y', 'label': 'R peaks marginalized', 'marker':'o', 'ls':'', 'fillstyle':'none'},
                                                                  {'subplot': 'ECG', 'ylim': [-2000, 2000], 'name':'det2', 'fs': dfs, 'col':'k', 'label':'Detection'},
                                                                  {'subplot': 'ECG', 'ylim': [-2000, 2000], 'name':'hold2', 'fs': dfs, 'col':'g', 'label':'Hold limit'},
                                                                  {'subplot': 'ECG', 'ylim': [-2000, 2000], 'name':'margin2', 'fs': dfs, 'col':'y', 'label':'Margin'}])

        # {'subplot': 'C3', 'ylim': [-1000, 1000], 'name': 'un', 'fs': fs, 'col': 'blue', 'label': 'Unc.'},

    chunk_list_stack_pa = []
    chunk_list_stack_marker = []

    c = 0
    lim1 = 0
    lim2 = lchunk[0]

    while lim2 < len(df):
        chunk_list = [df.iloc[lim1:lim2].copy(deep=True)]
        chunk_list_marker = [df_markers.loc[chunk_list[0].index[0]:chunk_list[0].index[-1]]]
        chunk_list_ga, chunk_list_marker_ga = corrector_ga.update(chunk_list, chunk_list_marker)
        chunk_list_ds = downsampler.update(chunk_list_ga)
        # chunk_list_ecg = selector.update(chunk_list_ds, ecg_chan)
        # chunk_list_ds[0].loc[:, [ecg_chan]] = butter_filter.update(chunk_list_ecg)
        chunk_list_pa, chunk_list_marker_pa = corrector_pa.update(chunk_list_ds, chunk_list_marker_ga)

        if make_plot:
            # plotter.update('un', chunk_list, eeg_chan)
            # plotter.update('ga', chunk_list_ga, eeg_chan)
            plotter.update('fi', chunk_list_ds, ecg_chan)
            plotter.update('ds', chunk_list_ds, eeg_chan)
            plotter.update_marker_points('fi', 'rm', chunk_list_marker_pa, ['R peak marginalized'])
            plotter.update_marker_points('fi', 'r', chunk_list_marker_pa, ['R peak fixed', 'R peak'])
            # plotter.update_marker_lines('start_det', chunk_list_marker_pa, ['Start of R peak detection'])
            plotter.update_marker_lines('start_pa', chunk_list_marker_pa, ['Start of PA subtraction'])
            # plotter.update_marker_lines('start_ga', chunk_list_marker_ga, ['Start of GA subtraction'])
            plotter.update_marker_lines('det1', chunk_list_marker_pa, ['R peak detection'])
            plotter.update_marker_lines('hold1', chunk_list_marker_pa, ['Hold limit'])
            plotter.update_marker_lines('margin1', chunk_list_marker_pa, ['Margin'])
            plotter.update_marker_lines('det2', chunk_list_marker_pa, ['R peak detection'])
            plotter.update_marker_lines('hold2', chunk_list_marker_pa, ['Hold limit'])
            plotter.update_marker_lines('margin2', chunk_list_marker_pa, ['Margin'])
            plotter.update('pa', chunk_list_pa, eeg_chan)

            if c % plot_stride == 0:
                plotter.slide()
                plotter.draw()

        c += 1
        lim1 = lim2
        lim2 = lim2 + lchunk[c % len(lchunk)]

        chunk_list_stack_pa.extend(chunk_list_pa)
        chunk_list_stack_marker.extend(chunk_list_marker_ga)
        chunk_list_stack_marker.extend(chunk_list_marker_pa)

    # Make run-length dataframes
    df_pa = pd.concat(chunk_list_stack_pa)
    df_markers = pd.concat(chunk_list_stack_marker)

    # Fit original latencies to downsampled times
    latencies = eeglab.tools.fit_latencies(latencies, times[0], df_pa.index)

    # Add GA correction events
    start_of_ga_building_time = df_markers.index[df_markers['marker'] == 'Start of GA building']
    start_of_ga_subtraction_time = df_markers.index[df_markers['marker'] == 'Start of GA subtraction']

    start_of_ga_building_latency = [eeglab.tools.find_closest_index(df_pa.index, start_of_ga_building_time)]
    types.extend(['Start of GA building'])
    latencies.extend(start_of_ga_building_latency)

    start_of_ga_subtraction_latency = [eeglab.tools.find_closest_index(df_pa.index, start_of_ga_subtraction_time)]
    types.extend(['Start of GA subtraction'])
    latencies.extend(start_of_ga_subtraction_latency)

    # Add PA correction events
    start_of_pa_building_time = df_markers.index[df_markers['marker'] == 'Start of PA building']
    start_of_pa_subtraction_time = df_markers.index[df_markers['marker'] == 'Start of PA subtraction']

    start_of_ga_building_latency = [eeglab.tools.find_closest_index(df_pa.index, start_of_pa_building_time)]
    types.extend(['Start of PA building'])
    latencies.extend(start_of_ga_building_latency)

    start_of_ga_subtraction_latency = [eeglab.tools.find_closest_index(df_pa.index, start_of_pa_subtraction_time)]
    types.extend(['Start of PA subtraction'])
    latencies.extend(start_of_ga_subtraction_latency)

    # Save
    EEG = eeglab.tools.pack(file['filename'] + '_cor-paLas', df_pa.transpose().to_numpy(), df_pa.index.to_numpy(), dfs, chans, types, latencies)  # '_cor-paLas-clear'
    eeglab.tools.save(file['set'] + sfol + file['filename'] + '_cor-paNeXoff' + '.mat', EEG) if save_data else None  # '_cor-paLas-clear'
