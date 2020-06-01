import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def resample(waveform: Tensor, orig_freq, int, new_freq: int) -> Tensor:
    r"""Wrapper around librosa.core.resample().
    """
    return librosa.core.resample(waveform, orig_freq, new_freq)


def wave_to_spectrogram(wave: Tensor) -> Tensor:
    r"""Wrapper around librosa.stft()
    """
    # TODO: add parameters (n_ftt, win_length, hop_length, ..)
    return torch.from_numpy(
        np.abs(librosa.stft(wave.numpy())) ** 2)  # To compute the power spectrogram


def wave_to_melspectrogram(wave: Tensor, sample_rate: int):
    r"""Wrapper around librosa.feature.melspectrogram()
    """
    # TODO: add parameters (n_ftt, win_length, hop_length, ..)
    return torch.from_numpy(librosa.feature.melspectrogram(y=wave.numpy(), sr=sample_rate))


def amplitude_to_db(spectrogram: Tensor, stype: str):
    r"""Wrapper around librosa.core.amplitude_to_db().
    """
    # stype is an argument left because of compatibility
    return torch.from_numpy(librosa.core.amplitude_to_db(S=spectrogram.numpy()))


def plot_wave(waveforms: [Tensor], sample_rate: int):
    r"""Plots the amplitude waveforms. The sample_rate must be the same for all the waveforms.
    """
    for idx, waveform in enumerate(waveforms):
        print("Waveform {}, shape: {}".format(idx, waveform.size()))
        print("Waveform {}, Sample rate: {}".format(idx, sample_rate))

    plt.figure()
    for waveform in waveforms:
        librosa.display.waveplot(waveform[0][0].numpy(), sr=sample_rate[0].numpy(), alpha=0.8)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(spectrogram: Tensor, sample_rate: int, in_dB: bool = False):
    assert len(spectrogram.size()) == 2, \
        "Dimensions of spectogram should be 2, found {}".format(len(spectrogram.size()))

    plt.figure()
    # Interesting option: y_axis can be changed to log and it displays better the lower frequencies
    librosa.display.specshow(spectrogram.numpy(), sr=sample_rate, x_axis='time', y_axis='hz')
    if in_dB:
        plt.colorbar(format='%+2.0f dB')
    else:
        plt.colorbar(format='%+2.0f')
    plt.xlabel("Time")
    plt.show()


def plot_melspectrogram(melspectrogram: Tensor, sample_rate: int, in_dB: bool = False):
    assert len(melspectrogram.size()) == 2, \
        "Dimensions of spectogram should be 2, found ".format(len(melspectrogram.size()))

    plt.figure()
    # Interesting option: y_axis can be changed to log and it displays better the lower frequencies
    librosa.display.specshow(melspectrogram.numpy(), sr=sample_rate, x_axis='time', y_axis='mel')
    if in_dB:
        plt.colorbar(format='%+2.0f dB')
    else:
        plt.colorbar(format='%+2.0f')
    plt.xlabel("Time")
    plt.show()
