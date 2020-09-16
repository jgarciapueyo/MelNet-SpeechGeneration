"""This module contains the functions needed to train a complete MelNet model or a single tier.

It only needs the information where the hyperparameters yaml file is (HParams). It will train the
model, save it and log the training, all depending on the values specified in the hyperparameters
file.

In MelNet paper (Table 1), they explain the batch size for training. However, this batch size is too
big depending on the memory of the GPU. To be able to simulate this, we can use gradient
accumulation which consists on splitting the batch of samples into several mini-batches of samples
that will be run sequentially and then update the model (after having run all the mini-batches).
To learn more see:
https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

Example::
    >> args = ... (dict with the path to the hyperparameters yaml file and, optionally the path
                   to the weights to resume training from that point)
    >> setup_training(args)
"""
import argparse
from datetime import datetime
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.optim
import torch.utils.data

from src.data import collatedata, librispeech, ljspeech, podcast
from src.dataprocessing import transforms
from src.dataprocessing import audio_normalizing
from src.model.GMM import GMMLoss
# this module implements the basic functionality of tiers. (The TierCheckpoint module should be
# favored over this one)
# from src.model.Tier import Tier1, Tier
# this module implements the functionality of tiers adding PyTorch checkpointing allowing for tiers
# with bigger hidden size
from src.model.TierCheckpoint import Tier1, Tier
from src.utils import tierutil
# from src.utils import gpumemory
from src.utils.eval import evaluation
from src.utils.hparams import HParams
from src.utils.logging import TensorboardWriter


def gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item()
    return total_norm ** (1. / 2)

def get_dataloader(hp: HParams) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Select dataaset according to the parameters file.

    Args:
        hp (HParams): hyperparameters for the model and other parameters (dataset, ...).

    Returns:
        dataloader (Dataloader): dataset to be consumed directly following PyTorch guidelines.
        num_samples (int): number of samples of the dataset
    """
    if hp.data.dataset == "podcast":
        dataset = podcast.PODCAST(root=hp.data.path,
                                  audio_folder=hp.data.audio_folder,
                                  text_file=hp.data.text_file)
        length = len(dataset)
        train_length = int(0.95 * length)
        train_data, test_data = torch.utils.data.random_split(dataset, [train_length,
                                                                        length - train_length])

        # https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
        train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                       batch_size=hp.training.batch_size,
                                                       shuffle=False,
                                                       num_workers=hp.training.num_workers,
                                                       collate_fn=collatedata.AudioCollatePodcast(),
                                                       pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                      batch_size=hp.training.batch_size,
                                                      shuffle=False,
                                                      num_workers=hp.training.num_workers,
                                                      collate_fn=collatedata.AudioCollatePodcast(),
                                                      pin_memory=True)
        return train_dataloader, test_dataloader, int(0.95 * length)

    elif hp.data.dataset == "librispeech":
        Path(hp.data.path).mkdir(parents=True, exist_ok=True)
        dataset = librispeech.download_data(root=hp.data.path, url=hp.data.url)
        length = len(dataset)
        train_length = int(0.95 * length)
        train_data, test_data = torch.utils.data.random_split(dataset, [train_length,
                                                                        length - train_length])

        # https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
        train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                       batch_size=hp.training.batch_size,
                                                       shuffle=False,
                                                       num_workers=hp.training.num_workers,
                                                       collate_fn=collatedata.AudioCollatePodcast(),
                                                       pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                      batch_size=hp.training.batch_size,
                                                      shuffle=False,
                                                      num_workers=hp.training.num_workers,
                                                      collate_fn=collatedata.AudioCollatePodcast(),
                                                      pin_memory=True)
        return train_dataloader, test_dataloader, int(0.95 * length)

    elif hp.data.dataset == "ljspeech":
        Path(hp.data.path).mkdir(parents=True, exist_ok=True)
        dataset = ljspeech.download_data(root=hp.data.path)
        length = len(dataset)
        train_length = int(0.95 * length)
        train_data, test_data = torch.utils.data.random_split(dataset, [train_length,
                                                                        length - train_length])

        # https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
        train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                       batch_size=hp.training.batch_size,
                                                       shuffle=False,
                                                       num_workers=hp.training.num_workers,
                                                       collate_fn=collatedata.AudioCollatePodcast(),
                                                       pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                      batch_size=hp.training.batch_size,
                                                      shuffle=False,
                                                      num_workers=hp.training.num_workers,
                                                      collate_fn=collatedata.AudioCollatePodcast(),
                                                      pin_memory=True)
        return train_dataloader, test_dataloader, int(0.95 * length)

    else:
        raise Exception(f"Dataset {hp.data.dataset} does not exist")


def resume_training(args: argparse.Namespace, hp: HParams, tier: int, model: Tier,
                    optimizer: torch.optim.Optimizer, logger: logging.Logger) \
        -> Tuple[Tier, torch.optim.Optimizer]:
    """
    Loads the model specified in args.checkpoint_path to resume training from that point.

    Args:
        args (argparse.Namespace): parameters to set up the training. At least, args must contain:
                                   args = {"path_config": ...,
                                           "tier": ...,
                                           "checkpoint_path": ...}
        hp (HParams): hyperparameters for the model and other parameters (training, dataset, ...)
        tier (int): number of the tier to load.
        model (Tier): model where the weights will be loaded.
        optimizer (torch.optim.Optimizer): optimizer where the information will be loaded.
        logger (logging.Logger): to log general information about resuming the training.

    Returns:
        model (Tier) and optimizer (torch.optim.Optimizer)
    """
    if not Path(args.checkpoint_path).exists():
        logger.error(f"Path for resuming training {args.checkpoint_path} does not exist.")
        raise Exception(f"Path for resuming training {args.checkpoint_path} does not exist.")

    logger.info(f"Resuming training with weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path)
    hp_chkpt = checkpoint["hp"]

    # Check if current hyperparameters and the ones from saved model are the same
    if hp_chkpt.audio != hp.audio:
        logger.warning("New params for audio are different from checkpoint. "
                       "It will use new params.")

    if hp_chkpt.network != hp.network:
        logger.error("New params for network structure are different from checkpoint.")
        # raise Exception("New params for network structure are different from checkpoint.")

    if checkpoint["tier_idx"] != tier:
        logger.error(
            f"New tier to train ({tier}) is different from checkpoint ({checkpoint['tier']}).")
        raise Exception(
            f"New tier to train ({tier}) is different from checkpoint ({checkpoint['tier']}).")

    if hp_chkpt.data != hp.data:
        logger.warning("New params for dataset are different from checkpoint. "
                       "It will use new params.")

    if hp_chkpt.training != hp.training:
        logger.warning("New params for training are different from checkpoint. "
                       "It will use new params.")

    # epoch_chkpt = checkpoint["epoch"]
    # iterations_chkpt = checkpoint["iterations"]
    # total_iterations_chkpt = checkpoint["total_iterations"]
    model.load_state_dict(checkpoint["tier"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


def train_tier(args: argparse.Namespace, hp: HParams, tier: int, extension_architecture: str,
               timestamp: str, tensorboardwriter: TensorboardWriter,
               logger: logging.Logger) -> None:
    """
    Trains one tier of MelNet.

    Args:
        args (argparse.Namespace): parameters to set up the training. At least, args must contain:
                                   args = {"path_config": ...,
                                           "tier": ...,
                                           "checkpoint_path": ...}
        hp (HParams): hyperparameters for the model and other parameters (training, dataset, ...)
        tier (int): number of the tier to train.
        extension_architecture (str): information about the network's architecture of this run
                                      (training) to identify the logs and weights of the model.
        timestamp (str): information that identifies completely this run (training).
        tensorboardwriter (TensorboardWriter): to log information about training to tensorboard.
        logger (logging.Logger): to log general information about the training of the model.
    """
    logger.info(f"Start training of tier {tier}/{hp.network.n_tiers}")

    # Setup the data ready to be consumed
    train_dataloader, test_dataloader, num_samples = get_dataloader(hp)

    # Setup tier
    # Calculate size of FREQ dimension for this tier
    tier_freq = tierutil.get_size_freqdim_of_tier(n_mels=hp.audio.mel_channels,
                                                  n_tiers=hp.network.n_tiers,
                                                  tier=tier)

    if tier == 1:
        model = Tier1(tier=tier,
                      n_layers=hp.network.layers[tier - 1],
                      hidden_size=hp.network.hidden_size,
                      gmm_size=hp.network.gmm_size,
                      freq=tier_freq)
    else:
        model = Tier(tier=tier,
                     n_layers=hp.network.layers[tier - 1],
                     hidden_size=hp.network.hidden_size,
                     gmm_size=hp.network.gmm_size,
                     freq=tier_freq)
    model = model.to(hp.device)
    model.train()
    parameters = model.parameters()

    # Setup loss criterion and optimizer
    criterion = GMMLoss()
    optimizer = torch.optim.RMSprop(params=parameters,
                                    lr=hp.training.lr,
                                    momentum=hp.training.momentum)

    # Check if training has to be resumed from previous checkpoint
    if args.checkpoint_path is not None:
        model, optimizer = resume_training(args, hp, tier, model, optimizer, logger)
    else:
        logger.info(f"Starting new training on dataset {hp.data.dataset} with configuration file "
                    f"name {hp.name}")

    # Train the tier
    total_iterations = 0
    loss_logging = 0  # accumulated loss between logging iterations
    loss_save = 0  # accumulated loss between saving iterations
    prev_loss_onesample = 1e8  # used to compare between saving iterations and decide whether or not
    # to save the model

    gradients = []

    for epoch in range(hp.training.epochs):
        logger.info(f"Epoch: {epoch}/{hp.training.epochs} - Starting")
        for i, (waveform, utterance) in enumerate(train_dataloader):

            # 1.1 Transform waveform input to melspectrogram and apply preprocessing to normalize
            waveform = waveform.to(device=hp.device, non_blocking=True)
            spectrogram = transforms.wave_to_melspectrogram(waveform, hp)
            spectrogram = audio_normalizing.preprocessing(spectrogram, hp)
            # 1.2 Get input and output from the original spectrogram for this tier
            input_spectrogram, output_spectrogram = tierutil.split(spectrogram=spectrogram,
                                                                   tier=tier,
                                                                   n_tiers=hp.network.n_tiers)
            length_spectrogram = input_spectrogram.size(2)
            # if item is too long, we jump to the next one
            if length_spectrogram > 1000:
                continue

            # 2. Compute the model output
            if tier == 1:
                # generation is unconditional so there is only one input
                mu_hat, std_hat, pi_hat = model(spectrogram=input_spectrogram)
            else:
                # generation is conditional on the spectrogram generated by previous tiers
                mu_hat, std_hat, pi_hat = model(spectrogram=output_spectrogram,
                                                spectrogram_prev_tier=input_spectrogram)
            # gpumemory.stat_cuda("Forward")
            # 3. Calculate the loss
            loss = criterion(mu=mu_hat, std=std_hat, pi=pi_hat, target=output_spectrogram)
            # gpumemory.stat_cuda("Loss")
            del spectrogram
            del mu_hat, std_hat, pi_hat

            # 3.1 Check if loss has exploded
            if torch.isnan(loss) or torch.isinf(loss):
                error_msg = f"Loss exploded at Epoch: {epoch}/{hp.training.epochs} - " \
                            f"Iteration: {i * hp.training.batch_size}/{num_samples}"
                logger.error(error_msg)
                raise Exception(error_msg)

            # 4. Compute gradients
            loss_cpu = loss.item()
            loss = loss / hp.training.accumulation_steps
            loss.backward()

            # 5. Perform backpropagation (using gradient accumulation so efective batch size is the
            # same as in the paper)
            if (total_iterations + 1) % (
                    hp.training.accumulation_steps / hp.training.batch_size) == 0:

                gradients.append(gradient_norm(model))
                avg_gradient = sum(gradients) / len(gradients)
                logger.info(f"Gradient norm: {gradients[-1]} - "
                            f"Avg gradient: {avg_gradient}")
                torch.nn.utils.clip_grad_norm_(parameters, 2200)
                optimizer.step()
                model.zero_grad()

            # 6. Logging and saving model
            loss_oneframe = loss_cpu / (length_spectrogram * hp.training.batch_size)
            loss_logging += loss_oneframe  # accumulated loss between logging iterations
            loss_save += loss_oneframe  # accumulated loss between saving iterations

            # 6.1 Save model (if is better than previous tier)
            if (total_iterations + 1) % hp.training.save_iterations == 0:
                # Calculate average loss of one sample of a batch
                loss_onesample = loss_save / hp.training.save_iterations
                # if loss_onesample of these iterations is lower, the tier is better and we save it
                if loss_onesample <= prev_loss_onesample:
                    path = f"{hp.training.dir_chkpt}/tier{tier}_{timestamp}_loss{loss_onesample:.2f}.pt"
                    torch.save(obj={'dataset': hp.data.dataset,
                                    'tier_idx': tier,
                                    'hp': hp,
                                    'epoch': epoch,
                                    'iterations': i,
                                    'total_iterations': total_iterations,
                                    'tier': model.state_dict(),
                                    'optimizer': optimizer.state_dict()}, f=path)
                    logger.info(f"Model saved to: {path}")
                    prev_loss_onesample = loss_onesample
                loss_save = 0

            # 6.2 Logging
            if (total_iterations + 1) % hp.logging.log_iterations == 0:
                # Calculate average loss of one sample of a batch
                loss_onesample = loss_logging / hp.logging.log_iterations
                tensorboardwriter.log_training(hp, loss_onesample, total_iterations)
                logger.info(f"Epoch: {epoch}/{hp.training.epochs} - "
                            f"Iteration: {i * hp.training.batch_size}/{num_samples} - "
                            f"Loss: {loss_onesample:.4f}")
                loss_logging = 0

            # 6.3 Evaluate
            if (total_iterations + 1) % hp.training.evaluation_iterations == 0:
                evaluation(hp, tier, test_dataloader, model, criterion, logger)
            total_iterations += 1

        # After finishing training: save model, hyperparameters and total loss
        path = f"{hp.training.dir_chkpt}/tier{tier}_{timestamp}_epoch{epoch}_final.pt"
        torch.save(obj={'dataset': hp.data.dataset,
                        'tier_idx': tier,
                        'hp': hp,
                        'epoch': epoch,
                        'iterations': evaluation(hp, tier, test_dataloader, model, criterion,
                                                 logger),
                        'total_iterations': total_iterations,
                        'tier': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, f=path)
        logger.info(f"Model saved to: {path}")
        tensorboardwriter.log_end_training(hp=hp, loss=-1)
        logger.info("Finished training")


def train_model(args: argparse.Namespace, hp: HParams, extension_architecture: str, timestamp: str,
                logger: logging.Logger) -> None:
    """
    Sets up tensorboard writer (one for every tier to train) and calls to train_tier(...).
    Tiers are trained independently as explained in Section 6.1 of MelNet paper.

    Args:
        args (argparse.Namespace): parameters to set up the training. At least, args must contain:
                                   args = {"path_config": ...,
                                           "tier": ...,
                                           "checkpoint_path": ...}
        hp (HParams): hyperparameters for the model and other parameters (training, dataset, ...)
        extension_architecture (str): information about the network's architecture of this run
                                      (training) to identify the logs and weights of the model.
        timestamp (str): information that identifies completely this run (training).
        logger (logging.Logger): to log general information about the training of the model.
    """
    # 1. Check if we have to train a single tier or a complete model (with several tiers)
    if args.tier is not None:
        # 1.1 Argument tier was defined. Only that tier will be trained.
        logging.info(f"Training single tier of the model: Tier {args.tier}")

        # 2. Setup tensorboard logging
        # 2.1 Create tensorboard logs directory (tensorboard requires a different folder for each
        # run of the model, in this case every run to train a tier) so we add the extension of the
        # network's architecture of this run and the timestamp to identify it completely
        tensorboard_dir = f"{hp.logging.dir_log_tensorboard}{extension_architecture}" \
                          f"_{timestamp}_tier{args.tier}"
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
        # 2.2 Create tensorboard writer
        tensorboardwriter = TensorboardWriter(hp, tensorboard_dir)

        # 3. Start training of the tier
        train_tier(args, hp, args.tier, extension_architecture, timestamp, tensorboardwriter,
                   logger)

        tensorboardwriter.close()

    else:
        # 1.2 Argument tier was not defined. Train all tiers of the model.
        logging.info("Training all tiers of the model")

        for tier in range(1, hp.network.n_tiers + 1):
            # 2. Setup tensorboard logging (one for every tier)
            # 2.1 Create tensorboard logs directory (tensorboard requires a different folder for
            # each run of the model, in this case every run to train a tier) so we add the extension
            # of the network's architecture of this run and the timestamp to identify it completely
            tensorboard_dir = hp.logging.dir_log_tensorboard + extension_architecture \
                              + f"_{timestamp}_tier{tier}"
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            # 2.2 Create tensorboard writer
            tensorboardwriter = TensorboardWriter(hp, tensorboard_dir)

            # 3. Start training of the tier
            train_tier(args, hp, tier, extension_architecture, timestamp, tensorboardwriter, logger)

            tensorboardwriter.close()
            del tensorboardwriter


def setup_training(args: argparse.Namespace) -> None:
    """
    Sets up directories (logging and tensorboard), sets up the general logger and calls
    train_model(...) with the parameters specified in args for this run.
    The model to train can be a complete model with all the tiers (if args.tier == None) or a
    single tier (the one specified in args.tier).

    Args:
        args (argparse.Namespace): parameters to set up the training. At least, args must contain:
                                   args = {"path_config": ...,
                                           "tier": ...,
                                           "checkpoint_path": ...}
    """
    # 1. Read hyperparameters from file
    hp = HParams.from_yaml(args.path_config)
    # check if GPU available and add it to parameters
    hp["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 2. Create extension of the architecture of the model and timestamp for this run (use to
    # identify folders and files created for this run)
    # format: f(params_file)_t(n_tiers)_l(n_layers)_hd(hidden_size)_gmm(gmm_size).
    extension_architecture = f"d{hp.name}_t{hp.network.n_tiers}_" \
                             f"l{'.'.join(map(str, hp.network.layers))}_" \
                             f"hd{hp.network.hidden_size}_gmm{hp.network.gmm_size}"
    timestamp = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 3 Create directories for saving logs and model weights if they do not exist
    # 3.1 Create model weights directory for this run (the same directory will be used for different
    #     runs of a model with same architecture and the difference will be in the file stored)
    hp["training"]["dir_chkpt"] = hp.training.dir_chkpt + extension_architecture
    Path(hp.training.dir_chkpt).mkdir(parents=True, exist_ok=True)
    # 3.2 Create general log directory for this run (the same directory will be used for different
    #     runs of a model with same architecture and the difference will be in the file stored)
    hp["logging"]["dir_log"] = hp.logging.dir_log + extension_architecture
    Path(hp.logging.dir_log).mkdir(parents=True, exist_ok=True)

    # 4. Setup general logging (it will use the folder previously created and the filename will be:
    tier = str(args.tier) if args.tier is not None else 'ALL'
    filename = f"{hp.logging.dir_log}/tier{tier}_{timestamp}"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=filename),  # handler to save the log to a file
            logging.StreamHandler()  # handler to output the log to the terminal
        ])
    logger = logging.getLogger()

    # 5. Show device that will be used for training: CPU or GPU
    logger.info(f"Device for training: {hp.device}")

    # 6. Start training of the model (or a single tier, depending on args)
    train_model(args, hp, extension_architecture, timestamp, logger)
