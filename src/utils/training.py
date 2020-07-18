"""
This module contains the functions needed to train one or several models of MelNet.

It only needs the information where the hyperparameters yaml file is (HParams). It will train the
model, save it and log the training, all depending on the values specified in the hyperparameters
file.

Example::
    >> args = ... (dict with the path to the hyperparameters yaml file and, optionally the path
                   to the weights to resume training from that point)
    >> setup_training(args)
"""

from datetime import datetime
import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader
import torch.utils.data
import torch

from src.data import librispeech, podcast, collatedata
from src.dataprocessing import transforms as T
from src.dataprocessing.audio_normalizing import preprocessing
from src.model.GMM import GMMLoss
from src.model.MelNet import MelNet
from src.utils.hparams import HParams
from src.utils.logging import TensorboardWriter


def get_dataloader(hp: HParams) -> DataLoader:
    """
    Select dataaset according to the parameters file and
    Args:
        hp (HParams): parameters.

    Returns:
        dataloader (Dataloader): dataset to be consumed directly following PyTorch guidelines.
    """
    if hp.data.dataset == "podcast":
        dataset = podcast.PODCAST(root=hp.data.path,
                                  audio_folder=hp.data.audio_folder,
                                  text_file=hp.data.text_file)

        # https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=hp.training.batch_size,
                                           shuffle=False,
                                           num_workers=hp.training.num_workers,
                                           collate_fn=collatedata.AudioCollatePodcast(),
                                           pin_memory=True)
    elif hp.data.dataset == "librispeech":
        Path(hp.data.path).mkdir(parents=True, exist_ok=True)
        dataset = librispeech.download_data(root=hp.data.path, url=hp.data.url)
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=hp.training.batch_size,
                                           shuffle=False,
                                           num_workers=hp.training.num_workers,
                                           collate_fn=collatedata.AudioCollatePodcast(),
                                           pin_memory=True)
    else:
        raise Exception(f"Dataset {hp.data.dataset} does not exist")


def train(hp: HParams, args: argparse.Namespace, extension_run: str,
          tensorboardwriter: TensorboardWriter,
          logger: logging.Logger) -> None:
    """
    Trains one model of MelNet.

    Args:
        hp (HParams): hyperparameters for the model and other parameters (training, dataset, ...)
        args (argparse.Namespace): input parameters (path to HParams file, ...)
        extension_run (str): information about the run (training) of this model to identify the logs
                             and weights of the model
        tensorboardwriter (TensorboardWriter): interface to save logs to tensorboard
        logger (logging.Logger): log general information about the training of MelNet
    """
    # Setup training dataset and dataloader
    dataloader = get_dataloader(hp)

    # Setup model
    melnet = MelNet(n_tiers=hp.network.n_tiers,
                    layers=hp.network.layers,
                    hidden_size=hp.network.hidden_size,
                    gmm_size=hp.network.gmm_size,
                    freq=hp.audio.mel_channels)
    melnet = melnet.to(hp.training.device)

    # Setup loss criterion and optimizer
    criterion = GMMLoss()
    optimizer = torch.optim.RMSprop(params=melnet.parameters(),
                                    lr=hp.training.lr,
                                    momentum=hp.training.momentum)

    # Check if training has to be resumed from previous checkpoint
    if args.checkpoint_path is not None:
        if not Path(args.checkpoint_path).exists():
            logger.error(f"Path for resuming training {args.checkpoint_path} does not exist.")
            raise Exception(f"Path for resuming training {args.checkpoint_path} does not exist.")

        logger.info(f"Resuming training with weights from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        hp_chkpt = checkpoint["hp"]

        if hp_chkpt.audio != hp.audio:
            logger.warning("New params for audio are different from checkpoint. "
                           "It will use new params.")
        if hp_chkpt.network != hp.network:
            logger.error("New params for network structure are different from checkpoint.")
            raise Exception("New params for network structure are different from checkpoint.")

        if hp_chkpt.data != hp.data:
            logger.warning("New params for dataset are different from checkpoint. "
                           "It will use new params.")

        if hp_chkpt.training != hp.training:
            logger.warning("New params for training are different from checkpoint. "
                           "It will use new params.")

        # epoch_chkpt = checkpoint["epoch"]
        # iterations_chkpt = checkpoint["iterations"]
        # total_iterations_chkpt = checkpoint["total_iterations"]
        melnet.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    else:
        logger.info(
            f"Starting new training on dataset {hp.data.dataset} with configuration file "
            f"name {hp.name}")

    # Train the network
    melnet = melnet.train()
    total_iterations = 0
    loss_global = 0
    loss_logging = 0
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
            del spectrogram
            del mu_hat, std_hat, pi_hat

            # Check if loss has exploded
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Loss exploded at Epoch: {epoch} - Iteration: {i}")
                raise Exception(f"Loss exploded at Epoch: {epoch} - Iteration: {i}")

            loss.backward()
            optimizer.step()

            # Logging and saving model
            total_iterations += 1
            loss_cpu = loss.item()
            loss_global += loss_cpu
            loss_logging += loss_cpu

            # Save model
            if total_iterations % hp.training.save_iterations == 1:
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
                loss_model = round(
                    loss_logging / (hp.logging.log_iterations * hp.training.batch_size), 2)
                logger.info(f"Model saved to: {hp.training.dir_chkpt + str(loss_model) + '.pt'}")

            # Logging
            if total_iterations % hp.logging.log_iterations == 0:
                # print loss of one sample of one batch
                tensorboardwriter.log_training(hp, loss, total_iterations)
                logger.info(
                    f"Epoch: {epoch} - Iteration: {i} - "
                    f"Loss: {loss_logging / (hp.logging.log_iterations * hp.training.batch_size)}"
                )
                loss_logging = 0

    # After finishing training: save model, hyperparameters and total loss
    torch.save(
        obj={
            'dataset': hp.data.dataset,
            'hp': hp,
            'epoch': -1,
            'iterations': -1,
            'total_iterations': total_iterations,
            'model': melnet.state_dict(),
            'optimizer': optimizer.state_dict()
        },
        f=hp.training.dir_chkpt + str(loss_global) + '-final.pt')
    tensorboardwriter.log_end_training(hp, loss_global)
    logger.info(f"Model saved to: {hp.training.dir_chkpt + str(loss_global) + '-final.pt'}")
    logger.info("Finished training")


def setup_training(args: argparse.Namespace) -> None:
    """
    Sets up and trains a model with the parameters specified in args.

    Args:
        args (argparse.Namespace): parameters to set up the training
    """
    # read hyperparameters from file
    hp = HParams.from_yaml(args.path_config)
    # check if GPU available and add it to parameters
    hp["training"]["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create extension for this run
    # format: f(params_file)_t(n_tiers)_l(n_layers)_hd(hidden_size)_gmm(gmm_size)-time.
    extension_run = f"d{hp.name}_t{hp.network.n_tiers}_" \
                    f"l{'.'.join(map(str, hp.network.layers))}_" \
                    f"hd{hp.network.hidden_size}_gmm{hp.network.gmm_size}_" \
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # create directories for logging and model weights if they do not exist
    # create model weights directory
    hp["training"]["dir_chkpt"] = hp.training.dir_chkpt + extension_run + "/"
    Path(hp.training.dir_chkpt).mkdir(parents=True, exist_ok=True)
    # create general log directory
    Path(hp.logging.dir_log).mkdir(parents=True, exist_ok=True)
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

    tensorboardwriter.close()
