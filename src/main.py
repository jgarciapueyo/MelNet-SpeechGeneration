from data import librispeech, podcast
from dataprocessing import utils, utils_librosa
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    #dataloader = librispeech.create_dataloader('../data', 'dev-clean')
    #dataiter = iter(dataloader)
    #waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = dataiter.next()
    #waveform2, sample_rate2, utterance2, speaker_id2, chapter_id2, utterance_id2 = dataiter.next()
    #utils.plot_wave([waveform, utils.resample(waveform, sample_rate, 44100)], sample_rate)

    podcast = podcast.PODCAST(root="../data/Podcast", audio_folder="corpus", text_file="metadata_TCC.csv")
    dataloader_podcast = torch.utils.data.DataLoader(podcast)
    dataiter_podcast = iter(dataloader_podcast)

    waveform1, sample_rate1, text1, season_id1, episode_id1, utterance_id1 = dataiter_podcast.next()
    utils_librosa.plot_wave([waveform1], sample_rate1)

