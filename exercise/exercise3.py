
# coding: utf-8


# import Python modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal

import peakutils

from sounddevice import play
import sounddevice as sd
# to install this package, you need to use the command line tool "pip"
# pip install peakutils --user 
 
## Edit the script to synthesize audio by means of Frequency Modulation (FM)
##!! FILL IN PARTS WITH "None" (5 places, 1 pt in total)
##!! Answer the questions, Q1 - Q5, in your report (1 pt in total)

# load the audio files
fs_org,x_org = wavfile.read('gtr55.wav')
#fs_org,x_org = wavfile.read('oboe59.wav')
x = signal.decimate(x_org,4,ftype='iir')
fs = fs_org/4.0
t = np.linspace(0, len(x)/fs, len(x))

# plot the sample
plt.plot(t, x)
plt.grid('on')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.show()

# fft analysis
winlen = 1024*8

# windowing!
hann=scipy.signal.get_window('hann',winlen)

# apply windowed DFT (0.2 pt)
num_ft_pt = winlen/2.0 - 1
X = scipy.fft(hann * x[10000:10000+winlen])
X = 20*np.log10( np.abs(X[0:int(num_ft_pt)]))
frq = np.linspace(0, fs/2.0, num_ft_pt)

# plot magnitude spectrum
plt.plot(frq, X)
plt.title('Frequency domain visualization of the audio signal')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')

# find indexes of the peaks. 
# Check that the threshold value is correct! Adjust the threshold to detect peaks only (0.2 pt).
indexes = peakutils.indexes(X, thres=0.6, min_dist=np.round(50/fs*winlen))

# convert indices from interval [0,winlen/2] into frequency Hz [0,fs/2] (0.2 pt)
peak_freqs = frq[indexes]

# plot the peak locations and magnitudes.
indexes = np.array(indexes) - 1
peak_amps = X[indexes]
print('Peaks found at: ')
print('\n'.join([ '{} {}'.format(peak_freq, 'Hz') for peak_freq in peak_freqs ]))

# plot the peak locations and magnitudes.
plt.plot(peak_freqs, peak_amps,'ro')
plt.show()

# pick the first 8 peaks
[F0, F1, F2, F3, F4, F5, F6, F7] = peak_freqs[0:8]
                            
# Q1: Are correct peaks selected using thres=0.7, what happens if you set thres=0.5 (0.2 pt)?
# (hint: change the threshold value between 0 and 1, thres, and check if results are consistent)

# Q2: What is the fundamental frequency based on your plot, calculate the ratios: F1/F0, F2/F0 (0.2 pt)? 
#print("Fundamental =" + str(peak_freqs[0] + "Hz" + "F1/F0=" + str(F4/F0)+" "+ "F2/F0=" + str(F1/F0)))
# Q3: Is the audio recordings, oboe59, harmonic, why or why not? Answer the same question about the other recording,  gtr55  (0.2 pt)? 

# Try simple FM Synthesis to replicate the audio signal
# FM synthesis formula
# A: amplitude
# f_carrier: a carrier frequency
# f_mod: a modulation frequency
# ind_mod: a modulation index, which interact to generate harmonic and non-harmonic sounds
# y = A * sin( 2*pi*f_carrier + ind_mod*sin(2*pi*f_mod))

# Implement FM synthesis with modulation index of 9, set the amplitude to that of the first peak (0.6 pt)
ind_mod = 9
length = 5
f_carrier = F0
f_mod = F0-F1
A = peak_amps[0]
print('{} {}'.format('Amplitude of FM synthesized tone: ', A))

# Generates a waveform via FM synthesis
y = A*np.sin(2*np.pi*f_carrier*t + ind_mod*np.sin(2*np.pi*f_mod*t))
 
# Performs an fft and plots the result
Y = scipy.fft(hann * y[10000:10000+winlen])
Y = 20*np.log10(np.abs(Y[0:int(winlen/2.0)-1]))
plt.plot(frq, Y)
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.show()

audioname = '%dHz_carrier-%dHz_mod-%s.wav' % (f_carrier, f_mod, str(ind_mod))
wavfile.write('audio_%s' % audioname, int(fs), y/np.max(np.abs(y)))
# winsound.PlaySound(audioname, winsound.SND_FILENAME)

# Q4: How does changing the modulation index affect the sound (0.2 pt)?
# Look at the spectrogram and listen, describe the sound change.

# Q5: Does the FM synthesis version sound different from the original signal (0.2 pt).
# List possible reasons, since both have same frequency components.


play(y/np.max(np.abs(y))*0.1,fs)
sd.sleep(2000)
play(x/np.max(np.abs(x))*0.1,fs)




