# TODO: add file header

class HParams(dict):
    def _initialize_default_values(self):
        r"""
        Initializes the parameter values according to MelNet parameters:
        https://arxiv.org/abs/1906.01083 - Table 1
        """
        self['default_number'] = self._default_number   # default number for other parameters
        self['power'] = 2                               # use power spectrograms

        # According to MelNet parameters
        self['n_fft'] = self._default_number * 6
        self['win_length'] = self._default_number * 6
        self['hop_length'] = self._default_number
        self['mel_channels'] = self._default_number

    def __init__(self, sample_rate: int, init: dict = None, default_number: int = 256):
        r"""
        Parameters of the model regarding the audio representations.

        It will contain the values of:
            power: type of spectrogram it will represent: 1 for energy, 2 for power
            n_fft: size of the Fast Fourier Transform
            win_length: window size
            hop_length: length of hop between STFT windows
            mel_channels: number of mel channels when transforming to MelSpectrogram

        Args:
            sample_rate (int): sample rate of the sample audio
            init (int, optional): custom values of the parameters. They will overwrite the default
             values. If some default parameter is not overwritten it will remain with the default
             value. (Default: None)
            default_number: base number to construct default values (Default: 256)
        """
        super(HParams, self).__init__()
        self._default_number = default_number
        self._initialize_default_values()
        self['sample_rate'] = sample_rate

        if init is not None:
            for key in init:
                self[key] = init[key]

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
