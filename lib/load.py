import numpy as np
import scipy.io as sio


class load:
    @staticmethod
    def load_ecg(su=2, se=2, ru=6, fol='data/ecg/test/'):
        fil = 'EEGqrs_daMigraine_su{}_se{}_ru{}.mat'.format(su, se, ru)
        mat = sio.loadmat(fol + fil, struct_as_record=False)
        EEG = mat['EEGECG'][0, 0]
        ecg = EEG.data[0, :].astype(np.float64)
        fs = EEG.srate[0, 0]
        events = EEG.event[0]
        gndt = np.zeros(len(ecg), dtype=int)
        for e in range(len(events)):
            if events[e].type == 'QRSi':
                gndt[events[e].latency[0, 0] - 1] = 1
        return ecg, fs, gndt






        # def Load(fol, dset, su, se, ru):
#     mat = sio.loadmat(fol + 'EEGqrs_da{}_su{}_se{}_ru{}.mat'.format(dset, su, se, ru), struct_as_record=False)
#     EEG = mat['EEGECG'][0, 0]
#     ecg = EEG.data[0, :].astype(np.float64)
#     fs = EEG.srate[0, 0]
#     events = EEG.event[0]
#     gndt = np.zeros(len(ecg), dtype=int)
#     for e in range(len(events)):
#         if events[e].type == 'QRSi':
#             gndt[events[e].latency[0, 0] - 1] = 1
#     return ecg, fs, gndt

