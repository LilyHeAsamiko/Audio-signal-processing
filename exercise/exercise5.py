
# coding: utf-8

import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import scipy
from scipy import signal
from scipy.io import wavfile
import librosa

# Q1: math exercise (1 pt in total)

## A. (0.5 pt)
# If the vocal tract length is 15cm, and you discretize it with a 48 kHz sampling rate, 
# how many discrete sampling periods it does take for a sound wave (340 m/s) to travel the vocal tract?
#0.15/340*48000 = 21.176470588235293
## B. (0.5 pt)
# What is the reflection coefficient k when a sound passes from section with area 1cm^2 to 2cm^2?
# k=1/3
# Q2: programming exercise (1 pt in total)

# read in the audio file
# fs,x = wavfile.read('rhythm_birdland.wav')
fs,x = wavfile.read('oboe59.wav')
x = signal.decimate(x,4,ftype='iir')
fs=fs/4


# normalize x so that its value is between [-1.00, 1.00] (0.1 pt)
x = x.astype('float64') / float(numpy.max(numpy.abs(x)))


## A. (0.5 pt)
# MFCCs are useful features in many speech applications.
# Follow instructions below to practice your skills in feature extraction.
# use librosa to extract 13 MFCCs
mfccs = librosa.feature.mfcc(y=x, sr=fs, S=None, n_mfcc=13)
# Visualize the MFCC series
plt.figure(figsize=(10, 4))
plt.pcolormesh(mfccs)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

## B. (0.5 pt)
# extract pitch using librosa
# set a windowsize of 30 ms
window_time = 30
fmin = 80
fmax = 350
# set an overlap of 10 ms
overlap = 10
total_samples = len(x)
# there are sample_f/1000 samples per ms
# calculate number of samples in one window
window_size = fs/1000 * window_time
hop_length = total_samples / window_size
# calculate number of windows needed
needed_nb_windows = total_samples / (window_size - overlap)
n_fft = needed_nb_windows * 2.0
# extract pitch
# th_value is sensitive in pitch tracking.
# change the th_value and check if the pitch track is what you desired.
th_value = 100 
pitches, magnitudes = librosa.core.piptrack(x, int(fs), n_fft= int(n_fft), hop_length=hop_length, fmin=fmin, fmax=fmax, threshold=th_value)
shape = numpy.shape(pitches)
nb_samples = shape[0]
nb_windows = shape[1]

# some post-processing
def extract_max(pitches,magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(numpy.max(pitches[:,i]))
        new_magnitudes.append(numpy.max(magnitudes[:,i]))
    return (new_pitches,new_magnitudes)

def smooth(x,window_len=11,window='hanning'):
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w=numpy.ones(window_len,'d')
    else:
            w=eval('numpy.'+window+'(window_len)')
    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

pitches, magnitudes = extract_max(pitches, magnitudes, shape)
pitches1 = smooth(pitches,window_len=30)

plt.figure(figsize=(20, 22))
t_vec_pitch = np.linspace(0,float(len(x))/float(fs),len(pitches1))
f,t,X = signal.spectrogram(x,fs=fs,window=scipy.signal.get_window('hann',1024))
plt.pcolormesh(t,f,20*np.log10(1e-6+np.abs(X)))
plt.xlabel('time (s)')
plt.ylabel('Frequency (Hz)')
plt.plot(t_vec_pitch,pitches1,'m.')
plt.show()




