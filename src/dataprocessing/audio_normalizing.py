"""
This module contains functions to normalize the spectrograms and feed them as input to the MelNet
model when training.
This is needed because if not, the values of the spectrograms are very different and with a
distribution skewed towards 0. The intention of this functions is to modify the distribution of the
values of the spectrograms closer to a normal distribution and with values between 0 and 1.

For more information, see notebooks/09_normalizing.ipynb
# TODO: change name and number of notebooks

Reference:
    * https://github.com/keithito/tacotron/blob/master/util/audio.py
"""

from torch import Tensor
import torch

import src.dataprocessing.transforms as T
from src.utils.hparams import HParams


def preprocessing(spectrogram: Tensor, hp: HParams) -> Tensor:
    """
    Preprocess spectrogram (in power) representation so that the values are distributed close to
    a normal distribution and in the range [0., 1.].

    Args:
        spectrogram (Tensor): in power representation. Shape: [B, FREQ, FRAMES]
        hp (HParams): parameters for the audio transformations.

    Returns:
        normalized_spectrogram (Tensor): Shape: [B, FREQ, FRAMES]
    """
    spectrogram = T.amplitude_to_db(spectrogram, hp)
    spectrogram = spectrogram - hp.audio.ref_level_db
    return normalize(spectrogram, hp)


def postprocessing(spectrogram: Tensor, hp: HParams) -> Tensor:
    """
    Postprocess spectrogram (output from the MelNet model) to return it to a power spectrogram.

    Args:
        spectrogram (Tensor): in power representation. Shape: [B, FREQ, FRAMES]
        hp (HParams): parameters for the audio transformations.

    Returns:
        power_spectrogram (Tensor): Shape: [B, FREQ, FRAMES]
    """
    spectrogram = denormalize(spectrogram, hp)
    spectrogram = spectrogram + hp.audio.ref_level_db
    return T.db_to_amplitude(spectrogram, hp)


def normalize(spectrogram: Tensor, hp: HParams) -> Tensor:
    return torch.clamp((spectrogram - hp.audio.min_level_db) / - hp.audio.min_level_db,
                       min=0.0, max=1.0)


def denormalize(spectrogram: Tensor, hp: HParams) -> Tensor:
    return (torch.clamp(spectrogram, min=0.0, max=1.0) * (- hp.audio.min_level_db)) \
           + hp.audio.min_level_db
