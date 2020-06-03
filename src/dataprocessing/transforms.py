import matplotlib.pyplot as plt
from src.utils.hparams import HParams
import torchaudio
from torchaudio import functional as F
from torch import Tensor


# TODO: compare using torchaudio.transforms vs torchaudio.functional (F)


def resample(waveform: Tensor, orig_freq: int, new_freq: int) -> Tensor:
    r"""Wrapper around torchaudio.transforms.Resample()."""
    r = torchaudio.transforms.Resample(orig_freq, new_freq)
    return r(waveform)


def wave_to_spectrogram(wave: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.transforms.Spectrogram()."""
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=hp.n_fft,
                                                    win_length=hp.win_length,
                                                    hop_length=hp.hop_length,
                                                    power=hp.power)
    return spectrogram(wave)


def spectrogram_to_wave(spectrogram: Tensor, hp: HParams, n_iter: int = 32) -> Tensor:
    r"""Wrapper around torchaudio.transforms.GriffinLim()."""
    griffinlim = torchaudio.transforms.GriffinLim(n_fft=hp.n_fft,
                                                  n_iter=n_iter,
                                                  win_length=hp.win_length,
                                                  hop_length=hp.hop_length,
                                                  power=hp.power)
    return griffinlim(spectrogram)


def spectrogram_to_melspectrogram(spectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.trnaforms.MelScale()"""
    # TODO: check if power spectrogram or magnitude (energy) spectrogram and modify it accordingly
    #  since melscale should be applied to power spectrogram
    melscale = torchaudio.transforms.MelScale(n_mels=hp.mel_channels,
                                              sample_rate=hp.sample_rate)
    return melscale(spectrogram)


def melspectrogram_to_spectrogram(melspectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.transforms.InverseMelScale()."""
    # n_stft = nº bins in spectrogram depending on n_fft, exactly n_fft // 2 + 1
    inversemelscale = torchaudio.transforms.InverseMelScale(n_stft=hp.n_fft // 2 + 1,
                                                            n_mels=hp.mel_channels,
                                                            sample_rate=hp.sample_rate)
    return inversemelscale(melspectrogram)


def wave_to_melspectrogram(wave: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.transforms.MelSpectrogram()."""
    melsprectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=hp.sample_rate,
                                                           n_fft=hp.n_fft,
                                                           win_length=hp.win_length,
                                                           hop_length=hp.hop_length,
                                                           n_mels=hp.mel_channels)
    return melsprectrogram(wave)


def melspectrogram_to_wave(melspectrogram: Tensor, hp: HParams, n_iter: int = 32) -> Tensor:
    r"""
    Composition of:
        transforms.melspectrogram_to_spectrogram() and transforms.spectrogram_to_wave().
    """
    spectrogram = melspectrogram_to_spectrogram(melspectrogram, hp)
    wave = spectrogram_to_wave(spectrogram, hp, n_iter)
    return wave


def amplitude_to_db(spectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.transforms.AmplitudeToDB()."""
    stype = 'power' if hp.power == 2 else 'magnitude'
    amplitudetodb = torchaudio.transforms.AmplitudeToDB(stype=stype)
    return amplitudetodb(spectrogram)


def db_to_amplitude(spectrogram: Tensor, hp: HParams) -> Tensor:
    r"""Wrapper around torchaudio.functional.DB_to_amplitude()."""
    # power_exp calculated according to torchaudio.functional.DB_to_amplitude docs
    power_exp = 1 if hp.power == 2 else 0.5
    return F.DB_to_amplitude(spectrogram, ref=1, power=power_exp)


def plot_wave(waveforms: [Tensor], hp: HParams):
    r"""Plots the amplitude waveforms."""
    for idx, waveform in enumerate(waveforms):
        print("Waveform {}, shape: {}".format(idx, waveform.size()))
        print("Waveform {}, Sample rate: {}".format(idx, hp.sample_rate))

    plt.figure()
    for waveform in waveforms:
        plt.plot(waveform.flatten(), alpha=0.8)

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(spectrogram: Tensor, hp: HParams):
    r"""Plots spectrogram without information in x-axis and y-axis"""
    # TODO: check if this function should be removed or changed to correctly display y-axis and x-axis
    assert len(spectrogram.size()) == 2, \
        "Dimensions of spectogram should be 2, found {}".format(len(spectrogram.size()))
    plt.imshow(spectrogram, origin='lower')
    plt.axis('off')
    plt.show()


def plot_melspectrogram(melspectrogram: Tensor, hp: HParams):
    r"""Plots melspectrogram without information in x-axis and y-axis"""
    # TODO: check if this function should be removed or changed to correctly display y-axis and x-axis
    assert len(melspectrogram.size()) == 2, \
        "Dimensions of spectogram should be 2, found ".format(len(melspectrogram.size()))
    plt.imshow(melspectrogram, origin='lower')
    plt.axis('off')
    plt.show()


def save_wave(filepath: str, wave: Tensor, hp: HParams):
    r"""Wrapper around torchaudio.save()."""
    return torchaudio.save(filepath, wave, hp.sample_rate)
