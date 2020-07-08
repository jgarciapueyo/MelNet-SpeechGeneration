from typing import List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src.utils.hparams import HParams

# TODO: improve documentation in every function

def resample(waveform: Tensor, orig_freq: int, new_freq: int) -> Tensor:
    r"""Wrapper around librosa.core.resample()."""
    return librosa.core.resample(waveform, orig_freq, new_freq)


def wave_to_spectrogram(waveform: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around librosa.stft()."""
    return torch.from_numpy(np.abs(
        librosa.core.stft(y=waveform.numpy(),
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)
    ) ** hp.power)  # To compute the energy spectrogram or power spectrogram


def spectrogram_to_wave(spectrogram: Tensor, hp: HParams, n_iter: int = 32):
    r"""Wrapper around librosa.core.griffinlim()."""
    # Convert to magnitude (energy) spectrogram if we were working with power spectrogram
    S = spectrogram if hp.power == 1 else torch.sqrt(spectrogram)

    return torch.from_numpy(
        librosa.core.griffinlim(S=S.numpy(),
                                n_iter=n_iter,
                                hop_length=hp.hop_length,
                                win_length=hp.win_length))


def spectrogram_to_melspectrogram(spectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around librosa.feature.melspectrogram()."""
    return torch.from_numpy(
        librosa.feature.melspectrogram(S=spectrogram.numpy(),
                                       sr=hp.sample_rate,
                                       n_fft=hp.n_fft,
                                       hop_length=hp.hop_length,
                                       win_length=hp.win_length,
                                       n_mels=hp.mel_channels))


def melspectrogram_to_spectrogram(melspectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around librosa.feature.inverse.mel_to_stft()."""
    return torch.from_numpy(
        librosa.feature.inverse.mel_to_stft(M=melspectrogram.numpy(),
                                            sr=hp.sample_rate,
                                            n_fft=hp.n_fft,
                                            power=hp.power))


def wave_to_melspectrogram(wave: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around librosa.feature.melspectrogram()"""
    return torch.from_numpy(
        librosa.feature.melspectrogram(y=wave.numpy(),
                                       sr=hp.sample_rate,
                                       n_fft=hp.n_fft,
                                       hop_length=hp.hop_length,
                                       win_length=hp.win_length,
                                       n_mels=hp.mel_channels))


def melspectrogram_to_wave(melspectrogram: Tensor, hp: HParams, n_iter: int = 32) -> Tensor:
    r"""Wrapper around librosa.inverse.mel_to_audio()"""
    return torch.from_numpy(
        librosa.feature.inverse.mel_to_audio(M=melspectrogram.numpy(),
                                             sr=hp.sample_rate,
                                             n_fft=hp.n_fft,
                                             hop_length=hp.hop_length,
                                             win_length=hp.win_length,
                                             power=hp.power,
                                             n_iter=n_iter))

def amplitude_to_db(spectrogram: Tensor, hp: HParams):
    r"""Wrapper around librosa.core.amplitude_to_db()."""
    # Convert to power spectrogram if we were working with magnitude (energy) spectrogram
    S = spectrogram if hp.power == 2 else spectrogram**2

    return torch.from_numpy(librosa.core.power_to_db(S=S.numpy()))


def db_to_amplitude(spectrogram: Tensor, hp: HParams):
    r"""Wrapper around librosa.core.db_to_power()."""
    return torch.from_numpy(librosa.core.db_to_power(S_db=spectrogram.numpy()))


def plot_wave(waveforms: [Tensor], hp: HParams):
    r"""Plots the amplitude waveforms."""
    for idx, waveform in enumerate(waveforms):
        print("Waveform {}, shape: {}".format(idx, waveform.size()))
        print("Waveform {}, Sample rate: {}".format(idx, hp.sample_rate))

    plt.figure()
    for waveform in waveforms:
        librosa.display.waveplot(waveform.flatten().numpy(), sr=hp.sample_rate, alpha=0.8)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(spectrogram: Tensor, hp: HParams, in_db: bool = False):
    r"""Plots spectrogram without information in x-axis and y-axis"""
    assert len(spectrogram.size()) == 2, \
        "Dimensions of spectogram should be 2, found {}".format(len(spectrogram.size()))

    plt.figure()
    # Interesting option: y_axis can be changed to log and it displays better the lower frequencies
    librosa.display.specshow(spectrogram.numpy(), sr=hp.sample_rate, x_axis='time', y_axis='hz')
    if in_db:
        plt.colorbar(format='%+2.0f dB')
    else:
        plt.colorbar(format='%+2.0f')
    plt.xlabel("Time")
    plt.show()


def plot_melspectrogram(melspectrogram: Tensor, hp: HParams, in_db: bool = False):
    assert len(melspectrogram.size()) == 2, \
        "Dimensions of spectogram should be 2, found ".format(len(melspectrogram.size()))

    plt.figure()
    # Interesting option: y_axis can be changed to log and it displays better the lower frequencies
    librosa.display.specshow(melspectrogram.numpy(), sr=hp.sample_rate, x_axis='time', y_axis='mel')
    if in_db:
        plt.colorbar(format='%+2.0f dB')
    else:
        plt.colorbar(format='%+2.0f')
    plt.xlabel("Time")
    plt.show()


def save_wave(filepath: str, wave: Tensor, hp: HParams):
    r"""Wrapper around librosa.output.write_wave()."""
    return librosa.output.write_wav(filepath, wave.numpy(), hp.sample_rate)
