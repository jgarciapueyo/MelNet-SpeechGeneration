import argparse
from argparse import Namespace
import logging
# Modify sys.path adding directory from where this file has been called so that different packages
# and modules of the project can be imported from Jupyter Notebooks as well as from python files
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.getcwd())

import torch.utils.data
import torch

from src.data import librispeech, podcast, loaddata
from src.dataprocessing import transforms as T
from src.dataprocessing.audio_normalizing import preprocessing
from src.model.GMM import GMMLoss
from src.model.MelNet import MelNet
from src.utils.hparams import HParams
from src.utils.logging import TensorboardWriter


def get_dataset(hp: HParams):
    if hp.data.dataset == "podcast":
        return podcast.PODCAST(root=hp.data.path,
                               audio_folder=hp.data.audio_folder,
                               text_file=hp.data.text_file)
    elif hp.data.dataset == "librispeech":
        Path(hp.data.path).mkdir(parents=True, exist_ok=True)
        return librispeech.download_data(root=hp.data.path, url=hp.data.url)
    else:
        raise Exception(f"Dataset {hp.data.dataset} does not exist")


def train(hp: HParams, args: Namespace, extension_run: str, tensorboardwriter: TensorboardWriter,
          logger: logging.Logger):
    # Setup training dataset
    dataset = get_dataset(hp)

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

    # to see the parameters of the MelNet model
    # for name, param in melnet.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.shape)

    # Setup loss criterion and optimizer
    criterion = GMMLoss()
    optimizer = torch.optim.RMSprop(params=melnet.parameters(),
                                    lr=hp.training.lr,
                                    momentum=hp.training.momentum)

    if args.checkpoint_path is not None:
        logger.info(f"Resuming training with weights from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        hp_chkpt = checkpoint["hp"]
        epoch_chkpt = checkpoint["epoch"]
        iterations_chkpt = checkpoint["iterations"]
        total_iterations_chkpt = checkpoint["total_iterations"]
        melnet.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if hp_chkpt != hp:
            logger.info("New params are different from checkpoint. It will use new params")
    else:
        logger.info(f"Starting new training")

    # Train the network
    melnet = melnet.train()
    total_iterations = 0
    for epoch in range(hp.training.epochs):
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

            # Check if loss has exploded
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("Loss exploded at Epoch: {epoch} - Iteration: {i}")
                raise Exception("loss exploded")

            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            # Logging
            total_iterations += 1
            if total_iterations % hp.logging.log_iterations == 0:
                # print loss
                tensorboardwriter.log_training(loss, epoch)
                logger.info(f"Epoch: {epoch} - Iteration: {i} - Loss: {loss}")
            # Save model
            if total_iterations % hp.training.save_iterations == 0:
                torch.save(
                    obj={
                        'dataset': hp.data.dataset,
                        'hp': hp,
                        'epoch': epoch,
                        'iterations': i,
                        'total_iterations': total_iterations,
                        'model': melnet.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    f=hp.training.dir_chkpt + extension_run + ".pt")
                break


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-config", type=str, required=True,
                        help="path to yaml configuration file")
    parser.add_argument("--checkpoint-path", type=str, required=False, default=None,
                        help="path to model weights to resume training")
    args = parser.parse_args()

    # read parameters from file
    hp = HParams.from_yaml(args.file_config)
    # check if GPU available and add it to parameters
    hp["training"]["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create directories for logging and model weights if they do not exist
    # create model weights directory
    Path(hp.training.dir_chkpt).mkdir(parents=True, exist_ok=True)
    # create general log directory
    Path(hp.logging.dir_log).mkdir(parents=True, exist_ok=True)
    # create extension for this run
    # format: d(dataset)_t(n_tiers)_l(n_layers)_hd(hidden_size)_gmm(gmm_size)-time.
    extension_run = f"d{hp.data.dataset}_t{hp.network.n_tiers}_" \
                    f"l{'.'.join(map(str, hp.network.layers))}_" \
                    f"hd{hp.network.hidden_size}_gmm{hp.network.gmm_size}_" \
                    f"{time.time()}"

    # create tensorboard logs directory
    # tensorboard requires a different folder for each run of the model so we add the extension for
    # this run
    hp["logging"]["dir_log_tensorboard"] = hp.logging.dir_log_tensorboard + extension_run
    Path(hp.logging.dir_log_tensorboard).mkdir(parents=True, exist_ok=True)

    # setup general logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=hp.logging.dir_log + extension_run + ".log"),
            # handler to save the log to a file
            logging.StreamHandler()  # handler to output the log to the terminal
        ])
    logger = logging.getLogger()
    logger.info(f"Device for training: {hp.training.device}")

    # setup tensorboard logging
    tensorboardwriter = TensorboardWriter(hp)

    train(hp, args, extension_run, tensorboardwriter, logger)
