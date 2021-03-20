import os
import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pathlib
from ecgdetectors import Detectors
# import biosignalsnotebooks as bsnb
from numpy import linspace, diff, zeros_like, arange, array
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

fs = 250.0
lowcut = 2
highcut = 64

data = []
test1 = pd.DataFrame()

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=1)
    ECG_filt_data = lfilter(b, a, data)
    return ECG_filt_data

directory = 'C:/Users/KK PSI/Documents/penelitian/ECG/mit-bih-arrhythmia-database-1.0.0/'

names = os.listdir(directory)

files = []
for name in names:
    fileExt = os.path.splitext(name)[-1]
    # print(name)
    # print(fileExt)
    a = name.split(".")[:-1]
    # exit()
    if '.atr' == fileExt:
        files.append(a)
        # print(a)

for filename in files:
    label = str(filename[0])
    np_label = np.array(label)
    record = wfdb.rdrecord(directory + label)
    record.__dict__['sig_name']
    # print(record.__dict__['sig_name'])
    signal_name = record.__dict__['sig_name']
    for i, c in enumerate(signal_name):
        if c == 'MLII':
            ECG_data = record.__dict__['p_signal'][0:50000, record.__dict__['sig_name'].index("MLII")]

    ECG_filt_data = butter_bandpass_filter(ECG_data, lowcut, highcut, fs, order=1)

    detectors = Detectors(fs)
    r_peaks = detectors.engzee_detector(ECG_filt_data)

    Sequence_ECG = []
    for i, c in enumerate(r_peaks[1:-1]):
        temp = np.append(ECG_filt_data[c - 60: c + 100], np_label)
        Sequence_ECG.append(temp)

    # test = pd.DataFrame(Sequence_ECG)
    test1 = test1.append(pd.DataFrame(Sequence_ECG), ignore_index=True)
# print(test1[test1.columns[22:25]].head())
# print(test1[test1.columns[22:25]].tail())
test1.to_csv('data.csv')
