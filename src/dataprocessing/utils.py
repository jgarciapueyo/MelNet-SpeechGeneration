import matplotlib.pyplot as plt
import torchaudio
from torch import Tensor


def resample(waveform: Tensor, orig_freq: int, new_freq: int) -> Tensor:
    r"""Wrapper around torchaudio.transforms.Resample().
    """
    r = torchaudio.transforms.Resample(orig_freq, new_freq)
    return r(waveform)


def wave_to_spectrogram(wave: Tensor):
    r"""Wrapper around torchaudio.transforms.Spectrogram().
    """
    # TODO: add parameters (n_ftt, win_length, hop_length, power, normalized, ...)
    spectrogram = torchaudio.transforms.Spectrogram(power=2)
    return spectrogram(wave)


def wave_to_melspectrogram(wave: Tensor, sample_rate: int):
    r"""Wrapper around torchaudio.transforms.MelSpectrogram().
    """
    # TODO: add parameters (n_ftt, win_length, hop_length, f_min, f_max, n_mels, ...)
    melsprectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
    return melsprectrogram(wave)


def amplitude_to_db(spectrogram: Tensor, stype: str):
    r"""Wrapper around torchaudio.transforms.AmplitudeToDB

    Args:
        spectrogram (Tensor): ----
        stype (str): scale of the spectrogram. It can be 'power' or 'magnitude'
    """
    amplitudetodb = torchaudio.transforms.AmplitudeToDB(stype=stype)
    return amplitudetodb(spectrogram)


def plot_wave(waveforms: [Tensor], sample_rate: int):
    r"""Plots the amplitude waveforms. The sample_rate must be the same for all the waveforms.
    """
    for idx, waveform in enumerate(waveforms):
        print("Waveform {}, shape: {}".format(idx, waveform.size()))
        print("Waveform {}, Sample rate: {}".format(idx, sample_rate))

    plt.figure()
    for waveform in waveforms:
        plt.plot(waveform.flatten(), alpha=0.8)

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(spectrogram: Tensor):
    assert len(spectrogram.size()) == 2, \
        "Dimensions of spectogram should be 2, found {}".format(len(spectrogram.size()))
    plt.imshow(spectrogram, origin='lower')
    plt.show()
    # TODO: add to save plot

#TODO: plot with librosa because scale it is not correct
def plot_melspectrogram(melspectrogram: Tensor):
    assert len(melspectrogram.size()) == 2, \
        "Dimensions of spectogram should be 2, found ".format(len(melspectrogram.size()))
    plt.imshow(melspectrogram, origin='lower')
    plt.xlabel("Time (1/100 s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()
    # TODO: add to save plot
