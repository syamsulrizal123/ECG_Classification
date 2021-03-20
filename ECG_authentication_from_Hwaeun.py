# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:18:08 2020

@author: CML-DAUN
"""

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


# directory = 'C:/Users/KK PSI/Documents/penelitian/PPG/cu-ventricular-tachyarrhythmia-database-1.0.0/'
# directory = 'C:/Users/KK PSI/Documents/penelitian/PPG/ecg-id-database-1.0.0/'
# directory = 'C:/Users/KK PSI/Documents/penelitian/PPG/mit-bih-normal-sinus-rhythm-database-1.0.0/'

directory = 'C:/Users/KK PSI/Documents/penelitian/ECG/mit-bih-arrhythmia-database-1.0.0/'
# record = wfdb.rdrecord(directory + 'cu01')
record = wfdb.rdrecord(directory + '100')
record.__dict__['sig_name']
# print(record.__dict__)
# exit()
signal_name = record.__dict__['sig_name']
data = record.__dict__['p_signal'][0:50000,:]
# plt.figure(figsize=(20,11))
# plt.plot(data)
# plt.show()
# exit()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=3)
    ECG_filt_data = lfilter(b, a, data)
    return ECG_filt_data

for i , c in enumerate(signal_name):
    if c == 'ECG':
        ECG_data = record.__dict__['p_signal'][0:50000, record.__dict__['sig_name'].index("ECG")]
    if c == 'MLII':
        ECG_data = record.__dict__['p_signal'][0:50000, record.__dict__['sig_name'].index("MLII")]
    if c == 'ECG I filtered':
        ECG_data = record.__dict__['p_signal'][0:50000, record.__dict__['sig_name'].index("ECG I filtered")]
    if c == 'ECG I':
        ECG_data = record.__dict__['p_signal'][0:50000, record.__dict__['sig_name'].index("ECG I")]
    if c == 'ECG1':
        ECG_data = record.__dict__['p_signal'][0:50000, record.__dict__['sig_name'].index("ECG1")]

fs = 250.0
# lowcut = 2
# highcut = 65
lowcut = 2
highcut = 40

# plt.figure(figsize=(20,11))
# plt.plot(ECG_data)
# plt.plot(ECG_data1)
# plt.show()
# exit()

ECG_filt_data =  butter_bandpass_filter(ECG_data, lowcut, highcut, fs, order=1 )

#plt.figure(figsize=(20,11))
#plt.plot(ECG_data)
#plt.plot(ECG_filt_data)
#plt.show()
#exit()
# print(type(ECG_filt_data))
# exit()
detectors = Detectors(fs)
# fs = 250
# data1 = data[:,0]
# r_peaks = detectors.engzee_detector(data1)
r_peaks = detectors.engzee_detector(ECG_filt_data)
# r_peaks = detectors.pan_tompkins_detector(ECG_filt_data)
# print(r_peaks)
plt.figure(figsize=(20,11))
# # plt.plot(ECG_data)
plt.plot(ECG_filt_data)
plt.plot(r_peaks, ECG_filt_data[r_peaks], 'ro')
plt.show()
exit()
#
# print(r_peaks)
# print(r_peaks[1:30])
# print(r_peaks[1:-1])
# exit()
Sequence_ECG = []
for i, c in enumerate(r_peaks[1:-1]):
    Sequence_ECG.append(ECG_filt_data[c-60: c+100])

Sequence_ECG = np.vstack(Sequence_ECG)
data_t = [Sequence_ECG, np.ones([len(Sequence_ECG), 1])]
data_t = np.hstack(data_t)

data_t2 = [Sequence_ECG, np.ones([len(Sequence_ECG), 1])*0]
data_t2 = np.hstack(data_t2)

T_data = np.vstack([data_t, data_t2])

plt.figure(figsize=(20,11))
plt.plot(Sequence_ECG[1,:]) #pemilihan peak
# plt.plot(Sequence_ECG[2,:]) #pemilihan peak
# plt.plot(Sequence_ECG[3,:]) #pemilihan peak
# plt.plot(Sequence_ECG[4,:]) #pemilihan peak
# plt.plot(Sequence_ECG[5,:]) #pemilihan peak
# plt.plot(Sequence_ECG[6,:]) #pemilihan peak
# plt.plot(Sequence_ECG[7,:]) #pemilihan peak
# plt.plot(Sequence_ECG[8,:]) #pemilihan peak
#
plt.show()
exit()


fileList = os.listdir(directory)
input_files = []
for f in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, f)) and not f.lower().startswith('.') and f.lower().endswith('hea'):
        input_files.append(f)


# f = input_files[0]
dataset = []
for i, f in enumerate(input_files):
    record = wfdb.rdrecord(directory + f.split('.')[0])
    tmp_data = record.__dict__['p_signal'][0:10000,0]
    dataset.append(tmp_data)
   

dataset = np.vstack(dataset)
np.savetxt('C:/Users/KK PSI/Documents/penelitian/PPG/inputfile_MIT.csv', Sequence_ECG, delimiter = ',')