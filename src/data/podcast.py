import csv
import os
from typing import Tuple

import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import unicode_csv_reader, walk_files


def load_podcast_item(fileid: str, root: str, audio_folder: str, ext_audio: str, text: Tuple[str, str]):
    """
    Load one item of the dataset.

    Args:
        fileid (str): name of the file to load (without extension)
        root (str): root folder of the dataset
        audio_folder (str): folder with the audio files inside root folder
        ext_audio (str): extension type of the audiofiles
        text (Tuple[str, str]): text transcription of audio file in grapheme_text and in phoneme_text
    """
    # Read audio
    file_audio = os.path.join(root, audio_folder, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    season_id, episode_id, utterance_id = fileid.split("_")

    return waveform, sample_rate, text, season_id, episode_id, utterance_id


class PODCAST(Dataset):
    """
    Create a Dataset for Podcast. This dataset contains audio from a dialogue-based podcast.
    Each item is a tuple of the form:
    waveform, sample_rate, [grapheme_text, phoneme_text], season_id, episode_id, utterance_id
    """

    _ext_text = ".csv"
    _ext_audio = ".wav"

    def __init__(self, root: str, audio_folder: str, text_file: str):
        """
        Args:
            root (str): root folder of the dataset
            audio_folder (str): folder with the audio files inside root folder
            text_file (str): path to the file with the text transcriptions of the audio files inside
                root folder
        """
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
        fileid, grapheme, g2p = self._text[n]
        return load_podcast_item(fileid, self._root, self._audio_folder, self._ext_audio,
                                 (grapheme, g2p))

    def __len__(self):
        return len(self._walker)
