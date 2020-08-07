"""
This module contains transformations to work with audio in its different representations like
audio waveform, linear spectrogram, mel spectrogram.
"""
from typing import List

import matplotlib.pyplot as plt
from torch import Tensor, device
import torchaudio
from torchaudio import functional as F

from src.utils.hparams import HParams


# TODO: compare using torchaudio.transforms vs torchaudio.functional (F)


def resample(waveforms: Tensor, orig_freq: int, new_freq: int) -> Tensor:
    r"""Wrapper around torchaudio.transforms.Resample().

    Args:
        waveforms (Tensor): audio waveform. Shape: [B, 1, L] where L is the number of times the waveform
            has been sampled and B is batch size.
        orig_freq (int)
        new_freq (int)

    Returns:
        waveform (Tensor): audio waveform: Shape: [B, 1, L'] where L' is the number of times the
            waveform has been sampled (after resampling) and B is batch size.
    """
    assert len(waveforms.size()) == 3, \
        "Dimensions of waveforms should be 3: [B, 1, L], but found {}".format(len(waveforms.size()))

    r = torchaudio.transforms.Resample(orig_freq, new_freq)
    return r(waveforms)


def wave_to_spectrogram(waveforms: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.transforms.Spectrogram().

    Args:
        waveforms (Tensor): audio waveform. Shape: [B, 1, L] where L is the number of times the waveform
            has been sampled and B is batch size.
        hp (HParams): parameters. Parameters needed are n_fft, win_length, hop_length and power.

    Returns:
        spectrogram (Tensor): spectrogram corresponding to waveform. Shape: [B, FREQ, FRAMES]
            where B is batch size and FREQ and FRAMES depends on the parameters hp.
            (See: https://pytorch.org/audio/transforms.html#spectrogram)
    """
    assert len(waveforms.size()) == 3, \
        "Dimensions of waveforms should be 3: [B, 1, L], but found {}".format(len(waveforms.size()))

    stype = 2 if hp.audio.spectrogram_type == 'power' else 1
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=hp.audio.n_fft,
                                                    win_length=hp.audio.win_length,
                                                    hop_length=hp.audio.hop_length,
                                                    power=stype).to(hp.device)
    return spectrogram(waveforms).squeeze(dim=1)


def spectrogram_to_wave(spectrogram: Tensor, hp: HParams, n_iter: int = 32) -> Tensor:
    r"""Wrapper around torchaudio.transforms.GriffinLim().

    Args:
        spectrogram (Tensor): spectrogram. Shape: [B, FREQ, FRAMES] where B is batch size.
        hp (HParams): parameters. Parameters needed are n_fft, win_length, hop_length and power.
        n_iter (int): number of iteration for phase recovery process.

    Returns:
        waveform (Tensor): audio waveform. Shape: [B, 1, L] L is the number of times the waveform
            has been sampled and B is batch size.
    """
    assert len(spectrogram.size()) == 3, \
        "Dimensions of spectrogram should be 3: [B, FREQ, FRAMES], but found {}".format(
            len(spectrogram.size()))

    stype = 2 if hp.audio.spectrogram_type == 'power' else 1
    griffinlim = torchaudio.transforms.GriffinLim(n_fft=hp.audio.n_fft,
                                                  n_iter=n_iter,
                                                  win_length=hp.audio.win_length,
                                                  hop_length=hp.audio.hop_length,
                                                  power=stype).to(hp.device)
    return griffinlim(spectrogram).unsqueeze(dim=1)


def spectrogram_to_melspectrogram(spectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.trnaforms.MelScale()

    Args:
        spectrogram (Tensor): spectrogram. Shape: [B, FREQ, FRAMES] where B is batch size.
        hp (HParams): parameters. Parameters needed are mel_channels and sample_rate.

    Returns:
        melspectrogram (Tensor): melspectrogram. Shape: [B, N_MELS, FRAMES] where B is batch size
            and N_MELS is the number of mel_channels.
    """
    assert len(spectrogram.size()) == 3, \
        "Dimensions of spectrogram should be 3: [B, FREQ, FRAMES], but found {}".format(
            len(spectrogram.size()))

    # FIXME: should MelScale only be applied to power spectrogram (and not to a linear one)?
    #  Ask for an answer
    melscale = torchaudio.transforms.MelScale(n_mels=hp.audio.mel_channels,
                                              sample_rate=hp.audio.sample_rate).to(hp.device)
    return melscale(spectrogram)


def melspectrogram_to_spectrogram(melspectrogram: Tensor, hp: HParams, n_iter: int = 1000) -> Tensor:
    r"""Wrapper around torchaudio.transforms.InverseMelScale().

    Args:
        melspectrogram (Tensor): melspectrogram. Shape: [B, N_MELS, FRAMES] where B is batch size
            and N_MELS is the number of mel_channels.
        hp (HParams): parameters. Parameters needed are n_fft, mel_channels and sample_rate.
        n_iter (int): number of optimization iterations.

    Returns:
        spectrogram (Tensor): linear spectrogram. Shape: [B, FREQ, FRAMES] where B is batch size.
    """
    assert len(melspectrogram.size()) == 3, \
        "Dimensions of spectrogram should be 3: [B, N_MELS, FRAMES], but found {}".format(
            len(melspectrogram.size()))

    # n_stft = nº bins in spectrogram depending on n_fft, exactly n_fft // 2 + 1
    inversemelscale = torchaudio.transforms.InverseMelScale(n_stft=hp.audio.n_fft // 2 + 1,
                                                            n_mels=hp.audio.mel_channels,
                                                            sample_rate=hp.audio.sample_rate,
                                                            max_iter=n_iter).to(hp.device)
    return inversemelscale(melspectrogram)


def wave_to_melspectrogram(waveform: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.transforms.MelSpectrogram().

    Args:
        waveform (Tensor): audio waveform. Shape: [B, 1, L] where B is batch size
        hp (HParams): parameters. Parameters needed are sample_rate, n_fft, win_length, hop_length
            and mel_channels.

    Returns:
        melspectrogram (Tensor): melspectrogram. Shape: [B, N_MELS, FRAMES] where B is batch size.
    """
    assert len(waveform.size()) == 3, \
        "Dimensions of spectrogram should be 3: [B, 1, L], but found {}".format(
            len(waveform.size()))

    melsprectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=hp.audio.sample_rate,
                                                           n_fft=hp.audio.n_fft,
                                                           win_length=hp.audio.win_length,
                                                           hop_length=hp.audio.hop_length,
                                                           n_mels=hp.audio.mel_channels).to(hp.device)
    return melsprectrogram(waveform).squeeze(dim=1)


def melspectrogram_to_wave(melspectrogram: Tensor, hp: HParams, n_iter: int = 32) -> Tensor:
    r"""
    Composition of transforms.melspectrogram_to_spectrogram() and transforms.spectrogram_to_wave().

    Args:
        melspectrogram (Tensor): melspectrogram. Shape: [B, N_MELS, FRAMES] where B is batch size.
        hp (HParams): parameters.
        n_iter (int): number of iteration for phase recovery process.

    Returns:
        waveform (Tensor): audio waveform. Shape: [B, 1, L] where B is batch size.
    """
    assert len(melspectrogram.size()) == 3, \
        "Dimensions of spectrogram should be 3: [B, N_MELS, FRAMES], but found {}".format(
            len(melspectrogram.size()))

    spectrogram = melspectrogram_to_spectrogram(melspectrogram, hp)
    waveform = spectrogram_to_wave(spectrogram, hp, n_iter)
    return waveform


def amplitude_to_db(spectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.transforms.AmplitudeToDB().

    Args:
        spectrogram (Tensor): spectrogram in the power/amplitude scale.
            Shape: [B, FREQ, FRAMES] or [B, N_MELS, FRAMES] if it is a melspectrogram.
        hp (HParams): parameters. Parameters needed are power.

    Returns:
        spectrogram (Tensor): spectrogram in decibel scale.
            Shape: Shape: [B, FREQ, FRAMES] or [B, N_MELS, FRAMES] if it is a melspectrogram.
    """
    assert len(spectrogram.size()) == 3, \
        "Dimensions of spectrogram should be 3: [B, FREQ, FRAMES] or [B, N_MELS, FRAMES], " \
        "but found {}".format(len(spectrogram.size()))

    stype = 'power' if hp.audio.spectrogram_type == 'power' else 'magnitude'
    amplitudetodb = torchaudio.transforms.AmplitudeToDB(stype=stype).to(hp.device)
    return amplitudetodb(spectrogram)


def db_to_amplitude(spectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.functional.DB_to_amplitude().

    Args:
        spectrogram (Tensor): spectrogram in the decibel scale.
            Shape: [B, FREQ, FRAMES] or [B, N_MELS, FRAMES] if it is a melspectrogram.
        hp (HParams): parameters. Parameters needed are power.

    Returns:
        spectrogram (Tensor): spectrogram in power/amplitude scale.
            Shape: Shape: [B, FREQ, FRAMES] or [B, N_MELS, FRAMES] if it is a melspectrogram.
    """
    assert len(spectrogram.size()) == 3, \
        "Dimensions of spectrogram should be 3: [B, FREQ, FRAMES] or [B, N_MELS, FRAMES], " \
        "but found {}".format(len(spectrogram.size()))

    # power_exp calculated according to torchaudio.functional.DB_to_amplitude docs
    power_exp = 1 if hp.audio.spectrogram_type == 'power' else 0.5
    return F.DB_to_amplitude(spectrogram, ref=1, power=power_exp)


def plot_wave(waveforms: Tensor, hp: HParams) -> None:
    r"""Plots the amplitude waveforms.

    Args:
        waveforms (Tensor): list of audio waveforms. Shape: [B, 1, L] where L is the number
            of times the waveform has been sampled and B is batch size.
        hp (HParams): parameters. Parameters needed are sample_rate.
    """
    assert len(waveforms.size()) == 3, \
        "Dimensions of waveforms should be 3, found {}".format(len(waveforms.size()))

    for idx, waveform in enumerate(waveforms):
        print("Waveform {}, shape: {}".format(idx, waveform.size()))
        print("Waveform {}, Sample rate: {}".format(idx, hp.audio.sample_rate))

    n_waveforms = waveforms.shape[0]
    # In case the waveforms tensor is in the GPU
    waveforms = waveforms.detach().to('cpu')
    fig = plt.figure()
    for i in range(0, n_waveforms):
        fig.add_subplot(n_waveforms, 1, i + 1)
        plt.plot(waveforms[i].flatten(), alpha=0.8)
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(spectrogram: Tensor, hp: HParams) -> None:
    r"""Plots spectrogram without information in x-axis and y-axis.

    Args:
        spectrogram (Tensor): spectrogram. Shape: [B, FREQ, FRAMES].
        hp (HParams): parameters.

    .. Note:
        This function plots the spectrogram without axis values.
    """
    assert len(spectrogram.size()) == 3, \
        "Dimensions of spectogram should be 3, found {}".format(len(spectrogram.size()))

    n_spectrograms = spectrogram.shape[0]
    # In case the spectrogram tensor is in the GPU
    spectrogram = spectrogram.detach().to('cpu')
    fig = plt.figure()
    for i in range(0, n_spectrograms):
        fig.add_subplot(n_spectrograms, 1, i + 1)
        plt.imshow(spectrogram[i].detach().to('cpu'), origin='lower')
        plt.axis('off')
    plt.show()


def plot_melspectrogram(melspectrogram: Tensor, hp: HParams) -> None:
    r"""Plots melspectrogram without information in x-axis and y-axis.

    Args:
        melspectrogram (Tensor): melspectrogram. Shape: [B, N_MELS, FRAMES].
        hp (HParams): parameters.

    .. Note:
        This function plots the melspectrogram without axis values.
    """
    assert len(melspectrogram.size()) == 3, \
        "Dimensions of melspectogram should be 3, found {}".format(len(melspectrogram.size()))

    n_melspectrograms = melspectrogram.shape[0]
    # In case the spectrogram tensor is in the GPU
    melspectrogram = melspectrogram.detach().to('cpu')
    fig = plt.figure()
    for i in range(0, n_melspectrograms):
        fig.add_subplot(n_melspectrograms, 1, i + 1)
        plt.imshow(melspectrogram[i], origin='lower')
        plt.axis('off')
    plt.show()


def save_spectrogram(filepath: str, spectrogram: Tensor, hp: HParams) -> None:
    r"""Saves spectrogram as an image.

    Args:
        filepath (str): path where the spectrogram will be saved.
        spectrogram (Tensor): spectrogram. Shape: [B, FREQ, FRAMES].
        hp (HParams): parameters.
    """
    assert len(spectrogram.size()) == 3, \
        "Dimensions of spectogram should be 3, found {}".format(len(spectrogram.size()))

    n_spectrograms = spectrogram.shape[0]
    # In case the spectrogram tensor is in the GPU
    spectrogram = spectrogram.detach().to('cpu')
    fig = plt.figure()
    for i in range(0, n_spectrograms):
        fig.add_subplot(n_spectrograms, 1, i + 1)
        plt.imshow(spectrogram[i].detach().to('cpu'), origin='lower')
        plt.axis('off')

    fig.savefig(fname=filepath)


def save_wave(filepath: str, waveform: Tensor, hp: HParams) -> None:
    r"""Wrapper around torchaudio.save().

    Args:
        filepath (str): path where the audio will be saved.
        waveform (Tensor): audio waveform to be saved. Shape: [1, L] where L is the number
            of times the waveform has been sampled.
        hp (HParams): parameters. Parameters needed are sample_rate.
    """
    # In case the waveform is in the GPU
    waveform = waveform.detach().to('cpu')
    torchaudio.save(filepath, waveform, hp.audio.sample_rate)
