"""Evaluation for one tier.

Allows to evaluate a model (in this case this means an individual tier). The result is the loss of
the model as defined by the criterion.
"""
import logging
from typing import Union

import torch
import torch.utils.data

from src.dataprocessing import audio_normalizing
from src.dataprocessing import transforms
from src.model.GMM import GMMLoss
from src.model.Tier import Tier1, Tier
# from src.model.TierCheckpoint import Tier1, Tier
from src.utils import tierutil
from src.utils.hparams import HParams


def evaluation(hp: HParams, tier: int, dataloader: torch.utils.data.DataLoader,
               model: Union[Tier1, Tier],
               criterion: GMMLoss, logger: logging.Logger) -> int:
    """
    Evaluates the model (tier) with respect to the data according to the criterion and logs it.

    Args:
        hp (HParams): hyperparameters for the model and other parameters (training, dataset, ...).
        tier (int): number of the tier (the model).
        dataloader (torch.utils.data.DataLoader): dataset enclosed as a DataLoader following PyTorch
                                                  guidelines.
        model (Tier): individual tier that will be evaluated.
        criterion (GMMLoss): function to compute the loss of the model.
        logger (logging.Logger): to log general information about the evaluation.

    Returns:
        avg_loss_of_frame (int): the average loss of a frame of a spectrogram.
    """
    model.eval()
    loss_of_sample = []
    length_of_sample = []

    for waveform, utterance in dataloader:
        # 1.1 Transform waveform input to melspectrogram and apply preprocessing to normalize
        waveform = waveform.to(device=hp.device, non_blocking=True)
        spectrogram = transforms.wave_to_melspectrogram(waveform, hp)
        spectrogram = audio_normalizing.preprocessing(spectrogram, hp)
        # 1.2 Get input and output from the original spectrogram for this tier
        input_spectrogram, output_spectrogram = tierutil.split(spectrogram=spectrogram,
                                                               tier=tier,
                                                               n_tiers=hp.network.n_tiers)

        with torch.no_grad():
            # 2. Compute the model output
            if tier == 1:
                # generation is unconditional so there is only one input
                mu_hat, std_hat, pi_hat = model(spectrogram=input_spectrogram)
            else:
                # generation is conditional on the spectrogram generated by previous tiers
                mu_hat, std_hat, pi_hat = model(spectrogram=output_spectrogram,
                                                spectrogram_prev_tier=input_spectrogram)

            # 3. Calculate the loss
            loss = criterion(mu=mu_hat, std=std_hat, pi=pi_hat, target=output_spectrogram)
        loss_of_sample.append(loss.item())
        length_of_sample.append(input_spectrogram.size(2))  # get FRAMES of input

    total_loss = sum(loss_of_sample)
    total_length = sum(length_of_sample)
    avg_loss_of_frame_sample = [loss / float(length) for loss, length in
                                zip(loss_of_sample, length_of_sample)]
    avg_loss_of_frame = sum(avg_loss_of_frame_sample) / len(avg_loss_of_frame_sample)
    logger.info(f"Evaluation - Total loss: {total_loss} Total length: {total_length} "
                f"Avg loss of frame: {avg_loss_of_frame}")
    model.train()
    return avg_loss_of_frame
