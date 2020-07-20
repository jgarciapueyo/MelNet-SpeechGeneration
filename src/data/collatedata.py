"""
This module contains classes to allow for loading audio datasets in batches.

The problem is that audio datasets samples are of various lengths and
torch.utils.datasets.DataLoader() require that every sample of the batch has the same shape. In
order to do that, we have to pad the samples with 0 to collate (concatenate) them into a single
tensor with shape [B, FREQ, FRAMES].
"""
from typing import List, Tuple

from torch import Tensor
import torch


class AudioCollatePodcast():
    """
    It allows to collate (concatenate) samples of a batch of the Podcast dataset with various
    lengths by padding with zeros so that all samples of the batch have the same shape.

    Example::
        >> dataset = podcast.Podcast(root=..., ...)
        >> dataloader = torch.utils.datasets.DataLoader(dataset=dataset,
                                                    batch_size=batch_size,
                                                    collate_fn=AudioCollatePodcast())
    """
    def __init__(self):
        return

    def __call__(self, batch_samples: List[Tensor]) -> Tuple[Tensor, List[str]]:
        """
        Collates the samples of the batch.

        Args:
            batch_samples (List[Tensor]): list of the samples of the batch. Each sample with shape:
                            Tuple[waveform, sample_rate, text, season_id, episode_id, utterance_id]
                            where waveform is of shape [1, L]

        Returns:
            waveforms (Tensor): contains the waveforms of all the samples with shape: [B, 1, L]
            texts (List[str]): contains the texts corresponding to the waveforms.
        """
        waveforms = torch.nn.utils.rnn.pad_sequence(
            [sample[0].T for sample in batch_samples],
            batch_first=True
        ).transpose(1, 2)

        texts = [sample[2] for sample in batch_samples]
        del batch_samples
        return waveforms, texts
