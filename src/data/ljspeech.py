import argparse

from torch.utils.data import Dataset
import torchaudio


def download_data(root: str) -> Dataset:
    ljspeech_data = torchaudio.datasets.LJSPEECH(root, download=True)
    return ljspeech_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=False, default='../../datasets/ljspeech',
                        help="root directory where to download the dataset")
    args = parser.parse_args()
    download_data(args.root)
