import torch
import torchaudio
import numpy as np
import os
from scipy.io.wavfile import write


path = "/Users/arturvarosyan/Documents/ADL/adl-coursework/data/MagnaTagATune/samples/train/0/american_bach_soloists-j_s__bach__cantatas_volume_v-01-gleichwie_der_regen_und_schnee_vom_himmel_fallt_bwv_18_i_sinfonia-117-146.npy"
data = np.load(path)

# For writing the wave file 
# write("test.wav", rate=16000, data=data)

# print(sound, sample_rate)
spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=64)(data)
print(spectrogram)
np.save("test_spectrogram.npy", spectrogram)