import argparse
import torch
import torchaudio


def download_data(root: str, url: str):
    r"""Wrapper to download LIBRISPEECH dataset. Each item is a tuple of the form:
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id

    Args:
        root (str): root directory to download the dataset
        url (str): type of set of LIBRISPEECH to download (e.g. "dev-clean", "test-clean", ...)
            See: http://www.openslr.org/12/ for all the options
    """
    librispeech_data = torchaudio.datasets.LIBRISPEECH(root, url, download=True)
    return librispeech_data


def create_dataloader(root: str, url: str):
    r"""Wrapper to load LIBRISPEECH dataset.

    Args:
        root (str): root directory where the dataset is stored. If it is not there,
            it will download it
        url (str): type of set of LIBRISPEECH to download (e.g. "dev-clean", "test-clean", ...)
    """
    librispeech_data = download_data(root, url)
    return torch.utils.data.DataLoader(librispeech_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=False, default='../../data',
                        help="root directory where to download the dataset")
    parser.add_argument('-u', '--url', type=str, required=False, default='dev-clean',
                        help="version of the dataset to download")
    args = parser.parse_args()
    download_data(args.root, args.url)

