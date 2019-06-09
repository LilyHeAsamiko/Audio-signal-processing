import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import tf_transform as tf
from bsseval import bss_eval_sources
import matplotlib.pyplot as plot
import matplotlib.colors as colors

eps = np.finfo(np.float32).tiny


def load_audio(_audio_filename):
    """
    Load audio file

    :param _audio_filename:  
    :return: _y: audio samples
    :return: _fs: sampling rate
    """
    _fs, _y = wav.read(_audio_filename)
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs


# Load the audio data and make the mix
audio_filename1 = 'vocals.wav'
audio_filename2 = 'music.wav'
vocals, fs = load_audio(audio_filename1)
music, fs = load_audio(audio_filename2)
mix = vocals + music
signal_length = mix.shape[0]

# Write the mixed audio to a file
wav.write('mixture.wav', fs, mix)

# stack the sources in a multi-dim array
sources = np.zeros((2,signal_length), dtype=np.float32)
sources[0, :] = vocals[0:signal_length]
sources[1, :] = music[0:signal_length]


# STFT parameters
win_size = 4096
fft_size = 4096
hop = 1024
windowing_func = signal.hamming(win_size)


# STFT of the mixture
mix_mag, mix_phase = tf.stft(mix, windowing_func, fft_size, hop)


# Compute the magnitudes of isolated sources (= Oracle masks)
vocals_mag, vocals_phase = tf.stft(vocals, windowing_func, fft_size, hop)
music_mag, music_phase = tf.stft(music, windowing_func, fft_size, hop)

# Visualize magnitude spectra of the mixture
plot.figure()
plot.subplot2grid((5, 1), (0, 0), colspan=2)
plot.imshow(mix_mag.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=mix_mag.min()+0.01, vmax=mix_mag.max()))
plot.title('Mixture'), plot.ylabel('Frequency channels')

plot.subplot2grid((5, 1), (1, 0), colspan=2)
plot.imshow(vocals_mag.T, aspect='auto', origin='lower',
            norm=colors.LogNorm(vmin=vocals_mag.min()+0.01, vmax=vocals_mag.max()))
plot.title('Singing voice'),  plot.ylabel('Frequency channels')

plot.subplot2grid((5, 1), (2, 0), colspan=2)
plot.imshow(music_mag.T, aspect='auto', origin='lower',
            norm=colors.LogNorm(vmin=music_mag.min()+0.01, vmax=music_mag.max()))
plot.title('Musical accompaniment'), plot.ylabel('Frequency channels')


# TODO: Q1. Implement the Ideal Binary Mask (IBM) - "ibm_vocals" 
# TODO: Q3 modify the threshold

normalized_vocals_mag = vocals_mag / np.max(vocals_mag)
normalized_music_mag = music_mag / np.max(music_mag)
normalized_mix_mag = mix_mag / np.max(mix_mag)
threshold_ibm = 10
ibm_vocals = normalized_vocals_mag > threshold_ibm * normalized_music_mag

# Visualize IBM
plot.subplot2grid((5, 1), (3, 0), colspan=2)
plot.imshow(ibm_vocals.T, aspect='auto', origin='lower')
plot.title('IBM'), plot.ylabel('Frequency channels')

# Apply the mask to the mixture and record the estimate
vocals_mag_ibm = ibm_vocals * mix_mag
vocals_ibm = tf.i_stft(vocals_mag_ibm, mix_phase, win_size, hop)
wav.write('vocals_ibm.wav', fs, vocals_ibm)

# Evaluate the source separation quality
sources_ibm = np.zeros((2, signal_length), dtype=np.float32)
sources_ibm[0, :] = vocals_ibm[0:signal_length]
sources_ibm[1, :] = mix - vocals_ibm[0:signal_length]
sdr_ibm, sir_ibm, sar_ibm, perm = bss_eval_sources(sources, sources_ibm)

print('IBM : SIR = {}, SAR = {}'.format(np.mean(sir_ibm), np.mean(sar_ibm)))


# TODO: Q2. Implement the Ideal Ratio Mask (IRM)- "irm_vocals". Careful with division by 0! (hint: use eps)
exponent_irm = 2
smoothness_irm = 1
irm_vocals = np.power((np.power(normalized_vocals_mag,exponent_irm)) / (np.power(normalized_mix_mag,exponent_irm) + eps), smoothness_irm)

# Visualize IRM
plot.subplot2grid((5, 1), (4, 0), colspan=2)
plot.imshow(irm_vocals.T, aspect='auto', origin='lower')
plot.title('IRM'), plot.xlabel('Time frames'), plot.ylabel('Frequency channels')

# Apply the mask to the mixture and record the estimate
vocals_mag_irm = irm_vocals * mix_mag
vocals_irm = tf.i_stft(vocals_mag_irm, mix_phase, win_size, hop)
wav.write('vocals_irm.wav', fs, vocals_irm)

# Evaluate the source separation quality
sources_irm = np.zeros((2, signal_length), dtype=np.float32)
sources_irm[0, :] = vocals_irm[0:signal_length]
sources_irm[1, :] = mix - vocals_irm[0:signal_length]
sdr_irm, sir_irm, sar_irm, perm = bss_eval_sources(sources, sources_irm)

print('IRM : SIR = {}, SAR = {}'.format(np.mean(sir_irm), np.mean(sar_irm)))

plot.show()

# Using the IBM(with threshold being 5), the separated vocal sounds clear which
# is consistent to the relatively higher SIR 23.66468 and more artificial thus
# the SAR is lower which is 8.97290.
# Using the IRM, there are more background sounds which leads to a lower SIR,
# 19.58283 but the sound is less artificial thus, the SAR is higher, which is 
# 10.39450.
# When I tune the threshold of the IBM to 0.5, the separated vocal is with more
# backsounds but sounds less artificial than using threshold as 5. The result 
# is the SIR is lower as 15.33640 while the SAR is higher as 10.39450; When I 
# tune the threshold of the IBM to 10, the effect sounds inversive. It's clearer
# again but more artificial. And it is consistence with the SIR being higher 
# as 22.25121 and SAR being lowest 6.62037. So for the IBM the higher the shreshold
# is the lower the SAR will be which means it sounds more artificial. But the SIR
# is not neccessarily the higher. If the threshold is too high the SIR is lower
# (comparing 5 to 10).
# 