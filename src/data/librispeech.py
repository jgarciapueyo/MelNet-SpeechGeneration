"""
This module contains functions to download the LIBRISPEECH dataset and load it to be used in an easy
way, as an iterable over the dataset (torch.utils.data.DataLoader), following PyTorch guidelines.

If the module is invoked as the main program, it will download the LIBRISPEECH dataset.
Run [python librispeech.py -h] for more information about the program arguments and its usage.

If the module is imported, it can be used as following.
Example::
    >> root = ...
    >> librispeech = librispeech.download_data(root, 'dev-clean')
    >> dataloader_librispeech = torch.utils.data.DataLoader(librispeech)
    >> dataiter = iter(dataloader_librispeech)
    >> waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = dataiter.next()
"""
import argparse

from torch.utils.data import Dataset
import torchaudio


def download_data(root: str, url: str) -> Dataset:
    r"""Wrapper to download LIBRISPEECH dataset. Each item is a tuple of the form:
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id

    Args:
        root (str): root directory to download the dataset
        url (str): type of set of LIBRISPEECH to download (e.g. "dev-clean", "test-clean", ...)
            See: http://www.openslr.org/12/ for all the options

    Returns:
        dataset (Dataset): LIBRISPEECH dataset (according to PyTorch specification of a dataset)
    """
    librispeech_data = torchaudio.datasets.LIBRISPEECH(root, url, download=True)
    return librispeech_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=False, default='../../datasets',
                        help="root directory where to download the dataset")
    parser.add_argument('-u', '--url', type=str, required=False, default='dev-clean',
                        help="version of the dataset to download. \
                        See: http://www.openslr.org/12/ for all the options")
    args = parser.parse_args()
    download_data(args.root, args.url)
