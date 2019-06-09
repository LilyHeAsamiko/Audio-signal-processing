import librosa

audio_filename = 'male.wav'
y, fs = librosa.load(audio_filename)

# STEP 1: implement pitch shift, you can comment the code after playing around with it
pitch_shift = 4  # Try -4 and 4
y = librosa.effects.pitch_shift(y, fs, n_steps=pitch_shift)  # TODO: implement pitch shift code here using librosa
librosa.output.write_wav('male_pitch_{}.wav'.format(pitch_shift), y, fs)

# # STEP 2: implement time shift, you can comment the code after playing around with it
time_stretch = 0.5 # Try 2 and 0.5
y = librosa.effects.time_stretch(y, time_stretch)  # TODO: implement time stretch code here using librosa
librosa.output.write_wav('male_stretch_{}.wav'.format(time_stretch), y, fs)

# # STEP 3: implement pitch and time shift together on the same audio with following variables
pitch_shift = 10
y = librosa.effects.pitch_shift(y, fs, n_steps=pitch_shift)  # TODO: implement pitch shift code here using librosa
time_stretch = 1.5
y = librosa.effects.time_stretch(y, time_stretch)  # TODO: implement time stretch code here using librosa
librosa.output.write_wav('male_fun_{}{}.wav'.format(pitch_shift, time_stretch), y, fs)
