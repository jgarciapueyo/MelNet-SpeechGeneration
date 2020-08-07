""""
This module contains functions to download the LJSPEECH dataset and load it to be used in an easy
way, as an iterable over the dataset (torch.utils.data.DataLoader), following PyTorch guidelines.

If the module is invoked as the main program, it will download the LJSPEECH dataset.
Run [python librispeech.py -h] for more information about the program arguments and its usage.

If the module is imported, it can be used as following.
Example::
    >> root = ...
    >> ljspeech = ljspeech.download_data(root)
    >> dataloader_ljspeech = torch.utils.data.DataLoader(ljspeech)
    >> dataiter = iter(dataloader_ljspeech)
    >> waveform, sample_rate, transcript, normalized_transcript = dataiter.next()
"""
import argparse

import torch.utils.data
import torchaudio


def download_data(root: str) -> torch.utils.data.Dataset:
    r"""Wrapper to download LJSPEECH dataset. Each item is a tuple of the form:
        waveform, sample_rate, transcript, normalized_transcript

        Args:
            root (str): root directory to download the dataset

        Returns:
            dataset (Dataset): LJSPEECH dataset (according to PyTorch specification of a dataset)
        """
    ljspeech_data = torchaudio.datasets.LJSPEECH(root, download=True)
    return ljspeech_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=False, default='../../datasets/ljspeech',
                        help="root directory where to download the dataset")
    args = parser.parse_args()
    download_data(args.root)
