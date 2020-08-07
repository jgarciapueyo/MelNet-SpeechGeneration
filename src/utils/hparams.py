"""
This module contains classes to load parameters from yaml files and use them later in the project.

MelNet model is controlled by a great number of hyperparameters. Depending on the dataset that we
are using to train or to do speech synthesis, this hyperparameters must be different and the best
way to save them and have them in version control is to store them in files.

In this project, the hyperparameters of the models are found in files inside
MelNet-SpeechGeneration/models/params and are divided into five categories:
* name
* audio: parameters related with the data pipeline (transforms between different audio
         representations like audio waveform, spectrogram, melspectrogram, ...).
* network: parameters related with the structure of the network (n_tiers, layers of each tier, ...).
* data: information related to the folder where the data for training is found and additional
        information depending on the dataset being used.
* training: parameters related with the training phase (optimizer, learning rate, ...)

Reference:
    * https://neptune.ai/blog/how-to-track-hyperparameters
    * https://neptune.ai/blog/how-to-track-hyperparameters
"""
import yaml

from attrdict import AttrDict
import torch


class HParams(AttrDict):
    """
    Class to load hyperparameters from yaml file and use them with dict or dot notation.

    Example::
        >> root_folder = ...
        >> hp = HParams.from_yaml(root_folder)
        >> hp.parameter_to_use
    """

    @classmethod
    def from_yaml(cls, path: str) -> AttrDict:
        with open(path) as yaml_file:
            # Read parameters from yaml file
            hparams = cls(yaml.safe_load(yaml_file))
            # Update parameters with device to do the calculations on the GPU or CPU
            hparams["device"] = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
            return hparams


class HParamsManual(dict):
    """
    Class to manually define hyperparameters (with some predefined parameters according to
    MelNet paper and use them with dict or dot notation.

    Example::
        >> sample_rate = ...
        >> init = {n_fft=.., win_length=..., other_parameter=...}
        >> hp = HParamsManual(sample_rate=sample_rate, init=init)
        >> hp.n_fft
        >> hp.other_parameter
    """

    def _initialize_default_values(self):
        r"""
        Initializes the parameter values according to MelNet parameters:
        https://arxiv.org/abs/1906.01083 - Table 1
        """
        self['default_number'] = self._default_number  # default number for other parameters
        self['power'] = 2  # use power spectrograms

        # According to MelNet parameters
        self['n_fft'] = self._default_number * 6
        self['win_length'] = self._default_number * 6
        self['hop_length'] = self._default_number
        self['mel_channels'] = self._default_number

    def __init__(self, sample_rate: int, init: dict = None, default_number: int = 256):
        r"""
        Parameters of the model regarding the audio representations and other parameters.

        By default, it will contain the values of:
            power: type of spectrogram it will represent: 1 for energy, 2 for power
            n_fft: size of the Fast Fourier Transform
            win_length: window size
            hop_length: length of hop between STFT windows
            mel_channels: number of mel channels when transforming to MelSpectrogram

        but other parameters can be added in init argument.

        Args:
            sample_rate (int)
            init (int, optional): custom values of the parameters. They will overwrite the default
             values. If some default parameter is not overwritten it will remain with the default
             value. If some parameter is in dict but is not a default parameter, it will be added.
             (Default: None)
            default_number: base number to construct default values. See table 1 (Default: 256)
        """
        super(HParamsManual, self).__init__()
        self._default_number = default_number
        self._initialize_default_values()
        self['sample_rate'] = sample_rate

        if init is not None:
            for key in init:
                self[key] = init[key]

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
