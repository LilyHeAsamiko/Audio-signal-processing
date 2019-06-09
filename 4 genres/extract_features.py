# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 03:23:22 2018

@author: asamiko
"""
import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plot
import librosa
import matplotlib.colors as colors
import os
from sklearn import preprocessing
import random
random.seed(12345)


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


def extract_feature(_audio_filename, nb_mel_bands, nb_frames, nfft):
    # User set parameters
    win_len = nfft
    hop_len = win_len // 2
    window = np.hamming(win_len)
    # nb_mel_bands = 40

    # load audio
    _y, _fs = load_audio(_audio_filename)

    # audio_length = len(_y)
    # nb_frames = int(np.floor((audio_length - win_len) / float(hop_len)))

    # Precompute FFT to mel band conversion matrix
    fft_mel_bands = librosa.filters.mel(_fs, nfft, nb_mel_bands, fmin=0.0).T

    _mbe = np.zeros((nb_frames, nb_mel_bands))

    frame_cnt = 0
    for i in range(nb_frames):
        # framing and windowing
        y_win = _y[i * hop_len:i * hop_len + win_len] * window

        # calculate energy spectral density
        _fft_en = np.abs(fft(y_win)[:1 + nfft // 2]) ** 2

        # calculate mel band energy
        _mbe[frame_cnt, :] = np.dot(_fft_en, fft_mel_bands)

        frame_cnt = frame_cnt + 1
    return _mbe

# -------------------------------------------------------------------
#              Main script starts here
# -------------------------------------------------------------------


# TODO: Change the following three parameters as mentioned in the exercise
window_length = 512  # Window length in samples
nb_mel_bands = 32  # Number of Mel-bands to calculate Mel band energy feature in
nb_frames = 40  # Extracts max_nb_frames frames of features from the audio, and ignores the rest.
#  For example when max_mb_frames = 40, the script extracts features for the first 40 frames of audio
#  Where each frame is of length as specified by win_len variable in extract_feature() function


output_feat_name = 'four_genres_{}_{}_{}.npz'.format(nb_frames, nb_mel_bands, window_length)
print('output_feat_name: {}'.format(output_feat_name))

# location of data. #TODO: UPDATE ACCORDING TO YOUR SYSTEM PATH
classical_folder = 'classical'
jazz_folder = 'jazz'
metal_folder = 'metal'
pop_folder = 'pop'

classical_files = os.listdir(classical_folder)
jazz_files = os.listdir(jazz_folder)
metal_files = os.listdir(metal_folder)
pop_files = os.listdir(pop_folder)

# Generate training and testing splits
training_ratio = 0.8  # 80% files for training
nb_train_files = int(len(classical_files) * training_ratio)    # The number of files for speech and music or the same in this dataset. Hence we do this only once.
nb_test_files = len(classical_files) - nb_train_files

random.shuffle(classical_files)
classical_train_files = classical_files[:nb_train_files]
classical_test_files = classical_files[nb_train_files:]

random.shuffle(jazz_files)
jazz_train_files = jazz_files[:nb_train_files]
jazz_test_files = jazz_files[nb_train_files:]

random.shuffle(metal_files)
metal_train_files = metal_files[:nb_train_files]
metal_test_files = metal_files[nb_train_files:]

random.shuffle(pop_files)
pop_train_files = pop_files[:nb_train_files]
pop_test_files = pop_files[nb_train_files:]

# Extract training features
classical_train_data = np.zeros((nb_train_files * nb_frames, nb_mel_bands))
jazz_train_data = np.zeros((nb_train_files * nb_frames, nb_mel_bands))
metal_train_data = np.zeros((nb_train_files * nb_frames, nb_mel_bands))
pop_train_data = np.zeros((nb_train_files * nb_frames, nb_mel_bands))
for ind in range(nb_train_files):
    classical_train_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(classical_folder, classical_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    jazz_train_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(jazz_folder, jazz_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    metal_train_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(metal_folder, metal_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    pop_train_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(pop_folder, pop_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
# Extract testing features
classical_test_data = np.zeros((nb_test_files * nb_frames, nb_mel_bands))
jazz_test_data = np.zeros((nb_test_files * nb_frames, nb_mel_bands))
metal_test_data = np.zeros((nb_test_files * nb_frames, nb_mel_bands))
pop_test_data = np.zeros((nb_test_files * nb_frames, nb_mel_bands))
for ind in range(nb_test_files):
    classical_test_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(classical_folder, classical_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    jazz_test_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(jazz_folder, jazz_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    metal_test_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(metal_folder, metal_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    pop_test_data[ind * nb_frames:ind * nb_frames + nb_frames] = extract_feature(
        os.path.join(pop_folder, pop_files[ind]),
        nb_mel_bands,
        nb_frames,
        window_length
    )
    
# Plotting function to visualize training and testing data before normalization
plot.figure()
plot.subplot2grid((2, 3), (0, 0), colspan=2), plot.imshow(classical_train_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=classical_train_data.min(), vmax=classical_train_data.max()))
plot.title('TRAINING DATA')
plot.xlabel('classical - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (0, 2)), plot.plot(np.mean(classical_train_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')
plot.subplot2grid((2, 3), (1, 0), colspan=2), plot.imshow(jazz_train_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=jazz_train_data.min(), vmax=jazz_train_data.max()))
plot.xlabel('jazz - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (1, 2)), plot.plot(np.mean(jazz_train_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')

plot.figure()
plot.subplot2grid((2, 3), (0, 0), colspan=2), plot.imshow(metal_train_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=metal_train_data.min(), vmax=metal_train_data.max()))
plot.title('TRAINING DATA')
plot.xlabel('metal - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (0, 2)), plot.plot(np.mean(metal_train_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')
plot.subplot2grid((2, 3), (1, 0), colspan=2), plot.imshow(pop_train_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=pop_train_data.min(), vmax=pop_train_data.max()))
plot.xlabel('pop - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (1, 2)), plot.plot(np.mean(pop_train_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')

plot.figure()
plot.subplot2grid((2, 3), (0, 0), colspan=2), plot.imshow(classical_test_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=classical_test_data.min(), vmax=classical_test_data.max()))
plot.title('TESTING DATA')
plot.xlabel('classical - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (0, 2)), plot.plot(np.mean(classical_test_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')
plot.subplot2grid((2, 3), (1, 0), colspan=2), plot.imshow(jazz_test_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=jazz_test_data.min(), vmax=jazz_test_data.max()))
plot.xlabel('jazz - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (1, 2)), plot.plot(np.mean(jazz_test_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')

plot.figure()
plot.subplot2grid((2, 3), (0, 0), colspan=2), plot.imshow(metal_test_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=metal_test_data.min(), vmax=metal_test_data.max()))
plot.title('TESTING DATA')
plot.xlabel('metal - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (0, 2)), plot.plot(np.mean(metal_test_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')
plot.subplot2grid((2, 3), (1, 0), colspan=2), plot.imshow(pop_test_data.T, aspect='auto', origin='lower', norm=colors.LogNorm(vmin=pop_test_data.min(), vmax=pop_test_data.max()))
plot.xlabel('pop - frames')
plot.ylabel('mel-bands')
plot.subplot2grid((2, 3), (1, 2)), plot.plot(np.mean(pop_test_data, 0))
plot.xlabel('mel-bands')
plot.ylabel('mean magnitude')


# Concatenate four genres data into training and testing data
lb = preprocessing.LabelBinarizer()
train_data = np.concatenate((classical_train_data, jazz_train_data, metal_train_data, pop_train_data), 0)
test_data = np.concatenate((classical_test_data, jazz_test_data, metal_test_data, pop_test_data), 0)

# Labels for training and testing data
train_labels = np.concatenate((np.zeros(classical_train_data.shape[0]), np.ones(jazz_train_data.shape[0]), 2*np.ones(metal_train_data.shape[0]), 3*np.ones(pop_train_data.shape[0])))
train_labels = lb.fit_transform(train_labels)                        
test_labels = np.concatenate((np.zeros(classical_test_data.shape[0]), np.ones(jazz_test_data.shape[0]), 2*np.ones(metal_test_data.shape[0]), 3*np.ones(pop_test_data.shape[0])))
test_labels = lb.fit_transform(test_labels)                               
# Normalize the training data, and scale the testing data using the training data weights
scaler = preprocessing.StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Save labels and the normalized features
np.savez(output_feat_name, train_data, train_labels, test_data, test_labels)
print('output_feat_name: {}'.format(output_feat_name))
plot.show()


