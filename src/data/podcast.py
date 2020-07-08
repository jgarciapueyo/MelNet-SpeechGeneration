"""
This module contains functions to load the PODCAST dataset and load it to be used in an easy
way, as an iterable over the dataset (torch.utils.data.DataLoader), following PyTorch guidelines.

The PODCAST dataset is composed of audio from a dialogue-based podcast.
At this moment, the PODCAST dataset is private and can not be downloaded.

Example::
    >> root = ...
    >> podcast = podcast.PODCAST(root=root, audio_folder="corpus", text_file="metadata_TCC.csv")
    >> dataloader_podcast = torch.utils.data.DataLoader(podcast)
    >> dataiter_podcast = iter(dataloader_podcast)
    >> waveform, sample_rate, text, season_id, episode_id, utterance_id = dataiter_podcast.next()
"""
import csv
import os
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset
import torchaudio
from torchaudio.datasets.utils import unicode_csv_reader, walk_files


def load_podcast_item(fileid: str, root: str, audio_folder: str, ext_audio: str,
                      text: Tuple[str, str]) -> Tuple[Tensor, int, Tuple[str, str], str, str, str]:
    """
    Load one item of the PODCAST dataset. Each item is a tuple of the form:
    waveform, sample_rate, [grapheme_text, phoneme_text], season_id, episode_id, utterance_id

    Args:
        fileid (str): name of the file to load (without extension).
        root (str): root folder of the dataset.
        audio_folder (str): folder with the audio files inside root folder.
        ext_audio (str): extension type of the audiofiles.
        text (Tuple[str, str]): text transcription of audio file in grapheme_text and in phoneme_text.

    Returns:
        waveform (Tensor): audio waveform of the audio. Shape: [1, L] where L is the number of times
            the waveform has been sampled.
        sample_rate (int): sample rate of the waveform
        text (Tuple[str, str]): grapheme text (characters) and phoneme text (how it sounds)
        season_id (str): podcast season of the item
        episode_id (str): episode of the item
        utterance_id (str): utterance string of the item
    """
    # Read audio
    file_audio = os.path.join(root, audio_folder, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    season_id, episode_id, utterance_id = fileid.split("_")

    return waveform, sample_rate, text, season_id, episode_id, utterance_id


class PODCAST(Dataset):
    """
    Creates a Dataset for Podcast. This dataset contains audio from a dialogue-based podcast.
    Each item is a tuple of the form:
    waveform, sample_rate, [grapheme_text, phoneme_text], season_id, episode_id, utterance_id

    Args:
            root (str): root folder of the dataset
            audio_folder (str): folder with the audio files inside root folder
            text_file (str): path to the file with the text transcriptions of the audio files inside
                root folder
    """

    _ext_text = ".csv"
    _ext_audio = ".wav"

    def __init__(self, root: str, audio_folder: str, text_file: str):
        self._root = root
        self._audio_folder = audio_folder
        walker = walk_files(root, suffix=self._ext_audio, prefix=False, remove_suffix=True)
        self._walker = list(walker)

        text_path = os.path.join(root, text_file)
        with open(text_path, "r") as text_file:
            text = unicode_csv_reader(text_file, delimiter="|", quoting=csv.QUOTE_NONE)
            self._text = list(text)
            # Delete first row of csv with the information about the columns
            self._text.pop(0)

        assert len(self._walker) == len(self._text), \
            "Number of audiofiles is different from number of texts"

    def __getitem__(self, n: int):
        """
        Load nth item of the PODCAST dataset. Each item is a tuple of the form:
        waveform, sample_rate, [grapheme_text, phoneme_text], season_id, episode_id, utterance_id

        Args:
            n (int): number of the item to load

        Returns:
            waveform (Tensor): audio waveform of the audio. Shape: [1, L] where L is the number
                of times the waveform has been sampled. Note that if you use
                torch.utils.data.DataLoader(dataset) the shape when iterating through the dataset
                will be [B, 1, L] where B is the batch size.
            sample_rate (int): sample rate of the waveform
            text (Tuple[str, str]): grapheme text (characters) and phoneme text (how it sounds)
            season_id (str): podcast season of the item
            episode_id (str): episode of the item
            utterance_id (str): utterance string of the item
        """
        fileid, grapheme, g2p = self._text[n]
        return load_podcast_item(fileid, self._root, self._audio_folder, self._ext_audio,
                                 (grapheme, g2p))

    def __len__(self):
        return len(self._walker)
