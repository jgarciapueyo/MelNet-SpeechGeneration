import torch.utils.data
import torch

from src.data import librispeech, podcast, loaddata
from src.dataprocessing import transforms as T
from src.dataprocessing.audio_normalizing import preprocessing
from src.model.GMM import GMMLoss
from src.model.MelNet import MelNet
from src.utils.hparams import HParams


def train():
    # Read parameters from file
    hp = HParams.from_yaml('models/params/dummymodel_podcast.yml')
    # check if GPU available
    hp["training"]["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(hp.training.device)

    # Setup training dataset
    dataset = podcast.PODCAST(root=hp.data.path,
                              audio_folder=hp.data.audio_folder,
                              text_file=hp.data.text_file)

    # https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=hp.training.batch_size,
                                             shuffle=False,
                                             num_workers=hp.training.num_workers,
                                             collate_fn=loaddata.AudioCollatePodcast(),
                                             pin_memory=True)

    # Setup model
    melnet = MelNet(n_tiers=hp.network.n_tiers,
                    layers=hp.network.layers,
                    hidden_size=hp.network.hidden_size,
                    gmm_size=hp.network.gmm_size,
                    freq=hp.audio.mel_channels)
    melnet = melnet.to(hp.training.device)
    # melnet.train()

    # Setup optimizer
    criterion = GMMLoss()
    optimizer = torch.optim.RMSprop(params=melnet.parameters(),
                                    lr=hp.training.lr,
                                    momentum=hp.training.momentum)

    # Train the network
    for epoch in range(1):

        # running_loss = 0.0
        for i, (waveform, utterance) in enumerate(dataloader):

            # transform waveform input to melspectrogram
            waveform = waveform.to(device=hp.training.device, non_blocking=True)
            spectrogram = T.wave_to_melspectrogram(waveform, hp)
            spectrogram = preprocessing(spectrogram, hp)

            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            mu_hat, std_hat, pi_hat = melnet(tier_idx=0, spectrogram=spectrogram)
            # calculate the loss
            loss = criterion(mu=mu_hat, std=std_hat, pi=pi_hat, target=spectrogram)
            # credit assignment
            loss.backward()  # divide to control loss
            # update model weights
            optimizer.step()
            # print loss
            print(f"Epoch {epoch} - Iteration {i} - Loss {loss}")

            if i == 3:
                break


if __name__ == '__main__':
    train()
