import torch
import torchaudio
import numpy as np
import os
import sys


def create_spectrogram(src_path, out_path):
    """
    Args:
        src_path (str): The path to the raw audio file
        out_path (str): The path to the output spectrogram file
    """
    data = np.load(src_path)
    waveform = torch.from_numpy(data)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Convert to spectrogram
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=12000, n_fft=512, hop_length=256, n_mels=96)(waveform)

    # Add padding
    add_pad = torch.nn.ConstantPad1d(padding=37, value=0)
    pad_spectrogram = add_pad(spectrogram)

    # Convert to decibels
    transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
    spectrogram_db = transform(pad_spectrogram)

    np.save(out_path, spectrogram_db)


def parse_dataset(root_path):
    """
    Args:
        root_path (str): The absolute or relative path to MagnaTagATune directory.
    """

    path = os.path.join(root_path, "samples")
    out_root = os.path.join(root_path,"samples_spectrogram")
    if not os.path.exists(out_root):
        os.mkdir(out_root)

    for split in os.listdir(path):
        split_path = os.path.join(path, split)

        if os.path.isdir(split_path):
            for part in os.listdir(split_path):
                part_path = os.path.join(split_path, part)

                if os.path.isdir(part_path):
                    for file in os.listdir(part_path):
                        if file != ".DS_STORE":
                            file_path = os.path.join(part_path, file)
                            
                            out_path = os.path.join(out_root, split, part)
                            if not os.path.exists(out_path):
                                os.makedirs(out_path)
                            out_path = os.path.join(out_path, file)
                            
                            create_spectrogram(file_path, out_path)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
        parse_dataset(path)
    else:
        print("Incorrect usage. Provide relative or absolute path to MagnaTagATune/")