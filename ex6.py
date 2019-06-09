#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import numpy as np

from scipy import signal

import tf_transform as tf

from ex6_utils import mask_calc, visualize_audio, evaluate_results, \
    print_evaluation_results, load_audio, save_audio

from ex6_models import get_model_1, get_model_2, get_model_3


def main():
    # Load the audio data and make the mix
    audio_filename1 = 'vocals.wav'
    audio_filename2 = 'music.wav'
    vocals, fs = load_audio(audio_filename1)
    music, fs = load_audio(audio_filename2)

    # Averaging amplitude
    mix = (vocals + music)/2
    signal_length = mix.shape[0]

    # Write the mixed audio to a file
    save_audio('mixture.wav', mix, fs)

    # stack the sources in a multi-dim array
    sources = np.zeros((2, signal_length), dtype=np.float32)
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

    # TODO: Q1. Implement a FNN-based network to predict the source, visualize and evaluate the result
    epochs_model_1 = 100
    model_case_1 = 'source prediction'

    model_1 = get_model_1(
        input_shape=mix_mag.shape,
        model_case=model_case_1
    )
    model_1.fit(
        x=mix_mag.reshape((1, ) + mix_mag.shape),
        y=vocals_mag.reshape((1, ) + vocals_mag.shape),
        epochs=epochs_model_1
    )
    predicted_magn_1 = model_1.predict(x=mix_mag.reshape((1, ) + mix_mag.shape))

    predicted_audio_1 = tf.i_stft(predicted_magn_1.squeeze(0), mix_phase, win_size, hop)

    sdr, sir, sar = evaluate_results(
        targeted=sources,
        predicted=predicted_audio_1,
        mixture=mix,
        signal_length=signal_length
    )

    print_evaluation_results(
        sdr=sdr, sir=sir, sar=sar, model_case=model_case_1
    )

    # Write the predicted audio to a file
    save_audio('{}.wav'.format(model_case_1), predicted_audio_1, fs)

    # TODO: Q2. Implement a FNN-based network to predict the mask, visualize and evaluate the result with respect to IRM results
    epochs_model_2 = 100
    exponent = 0.7
    smoothness = 1
    model_case_2 = 'mask prediction'

    # Estimate the mask using IRM equation
    calculated_mask = mask_calc(vocals_mag, mix_mag, exponent, smoothness)

    # Extract the vocals using mask - exercise 5 method
    vocals_mag_irm = calculated_mask * mix_mag
    vocals_irm = tf.i_stft(vocals_mag_irm, mix_phase, win_size, hop)
    save_audio('{} using irm.wav'.format(model_case_2), vocals_irm, fs)

    sdr, sir, sar = evaluate_results(
        targeted=sources,
        predicted=vocals_irm,
        mixture=mix,
        signal_length=signal_length
    )
    print_evaluation_results(
        sdr=sdr, sir=sir, sar=sar, model_case='{} using irm'.format(model_case_2)
    )

    # Estimate the mask using DNN now and extract vocals from it
    model_2 = get_model_2(
        input_shape=mix_mag.shape,
        model_case=model_case_2
    )
    model_2.fit(
        x=mix_mag.reshape((1, ) + mix_mag.shape),
        y=calculated_mask.reshape((1, ) + calculated_mask.shape),
        epochs=epochs_model_2
    )
    predicted_mask_fnn = model_2.predict(x=mix_mag.reshape((1, ) + mix_mag.shape))
    predicted_magn_2 = mix_mag * predicted_mask_fnn.squeeze(0)

    predicted_audio_2 = tf.i_stft(predicted_magn_2, mix_phase, win_size, hop)

    sdr, sir, sar = evaluate_results(
        targeted=sources,
        predicted=predicted_audio_2,
        mixture=mix,
        signal_length=signal_length
    )

    print_evaluation_results(
        sdr=sdr, sir=sir, sar=sar, model_case=model_case_2
    )

    # Write the predicted audio to a file
    save_audio('{}.wav'.format(model_case_2), predicted_audio_2, fs)

    # TODO: Q3. Modify one case of the above, using GRU RNNs and compare the results.
    epochs_model_3 = 50

    model_case_3 = 'rnn source prediction'
    model_3 = get_model_3(
        input_shape=mix_mag.shape,
        model_case=model_case_3
    )
    model_3.fit(
        x=mix_mag.reshape((1, ) + mix_mag.shape),
        y=vocals_mag.reshape((1, ) + vocals_mag.shape),
        epochs=epochs_model_3
    )
    predicted_magn_3 = model_3.predict(
        x=mix_mag.reshape((1, ) + mix_mag.shape)
    ).squeeze(0)

    # Or....
    # model_case_3 = 'mask prediction'
    #
    # model_3 = get_model_3(input_shape=mix_mag.shape[1], model_case=model_case_3)
    # model_3.fit(x=mix_mag, y=calculated_mask, epochs=epochs_model_3)
    # predicted_mask_rnn = model_3.predict(x=mix_mag)
    # predicted_magn_3 = mix_mag * predicted_mask_rnn
    #
    predicted_audio_3 = tf.i_stft(predicted_magn_3, mix_phase, win_size, hop)

    sdr, sir, sar = evaluate_results(
        targeted=sources,
        predicted=predicted_audio_3,
        mixture=mix,
        signal_length=signal_length
    )

    print_evaluation_results(
        sdr=sdr, sir=sir, sar=sar, model_case=model_case_3
    )

    # Write the predicted audio to a file
    save_audio('{}.wav'.format(model_case_3), predicted_audio_3, fs)

if __name__ == '__main__':
    main()

# EOF
