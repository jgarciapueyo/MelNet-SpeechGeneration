"""
MelNet complete architecture

Implementation of the complete MelNet model, composed of Tiers for the multiscale modelling.

For more information, see notebooks/08_MelNet.ipynb
# TODO: finish explanation in notebooks/08_MelNet.ipynb

.. Note:
    We expect the spectrogram to have shape: [B, FREQ, FRAMES] where B is batch size

    Explanation of axis of one of the spectrogram in the batch, which has shape [FREQ, FRAMES]:

    high frequencies  M +------------------------+
                        |     | |                |
         FREQ = j       |     | | <- a frame     |
                        |     | |                |
    low frequencies   0 +------------------------+
                        0                        T
                   start time    FRAMES = i    final time

    As the paper describes:
    - row major ordering
    - a frame is x_(i,*)
    - which proceeds through each frame from low to high frequency
"""
import logging
from pathlib import Path
from typing import List, Tuple

from torch import Tensor
import torch
import torch.nn as nn

from src.model.Tier import Tier1, Tier
from src.model.GMM import sample_gmm_batch
from src.utils.hparams import HParams
import src.utils.tierutil as tierutil


class MelNet(nn.Module):
    """MelNet model

    MelNet generates high-resolution spectrograms. Capturing global structure in such
    high-dimensional distributions is challenging. For that, they generate the spectrogram in a
    coarse-to-fine order partitioning the spectrogram into n_tiers. Each tier is modelled by a
    separate network phi_g.

    For this reason, each tier can be trained independently by partitioning the original spectrogram
    into the corresponding tiers, as explained in Section 6.1. For training, the method
    `forward(self, tier_idx: int, spectrogram) -> Tuple[Tensor, Tensor, Tensor]` should be used,
    which calls a single tier and generates the corresponding parameters of the GMM to, later,
    calculate the negative log-likelihood of the generated parameters with respect to the real data
    (spectrogram) using src.model.GMM.GMMLoss.

    Sampling, as explained in Section 6.2, should be done recursively. First, by sampling
    unconditionally from the first tier p(x_1; phi_1), and subsequently sampling conditionally on
    previous tiers p(x_g | x_<g ; phi_g). For sampling, the method
    `def sample(self, n_samples: int, length: int) -> Tensor` should be used, which will generate
    n_samples of audio of the given length by following the previous instructions.

    .. Note:
        MelNet is a model composed of tiers. Each tier is trained individually during the training
        phase so this class is only used for synthesis calling the method sample(). For this reason
        there is no method forward() because this class is not intended for training.
    """

    def __init__(self, n_tiers: int, layers: List[int], hidden_size: int, gmm_size: int, freq: int):
        """
        Args:
            n_tiers (int): number of tiers the model is composed of
            layers (List[int]): list with the layers of every tier
            hidden_size (int): parameter for the hidden_state of the Delayed Stack Layers and other
                               and other sizes
            gmm_size (int): number of mixture components of the GMM
            freq (int): size of the frequency axis of the spectrogram to generate. See note in the
                        documentation of the file.
        """
        super(MelNet, self).__init__()

        self.n_tiers = n_tiers
        self.layers = layers
        self.hidden_size = hidden_size
        self.gmm_size = gmm_size

        assert freq >= 2 ** (self.n_tiers / 2), "Size of frequency axis is too small for " \
                                                "being generated with the number of tiers " \
                                                "of this model"
        self.freq = freq

        self.tiers = nn.ModuleList(
            [Tier1(tier=1,
                   n_layers=layers[0],
                   hidden_size=hidden_size,
                   gmm_size=gmm_size,
                   # Calculate size of FREQ dimension for this tier
                   freq=tierutil.get_size_freqdim_of_tier(n_mels=self.freq,
                                                          n_tiers=self.n_tiers,
                                                          tier=1))]
            +
            [Tier(tier=tier_idx,
                  n_layers=layers[tier_idx],
                  hidden_size=hidden_size,
                  gmm_size=gmm_size,
                  # Calculate size of FREQ dimension for this tier
                  freq=tierutil.get_size_freqdim_of_tier(n_mels=self.freq,
                                                         n_tiers=self.n_tiers,
                                                         tier=tier_idx + 1))
             for tier_idx in range(1, n_tiers)]
        )

    def sample(self, hp: HParams, synthesisp: HParams, timestamp: str, logger: logging.Logger,
               n_samples: int, length: int) -> Tensor:
        """
        Generates n_samples of audio of the given length.

        Args:
            hp (HParams): parameters. Parameters needed are hp.device
            synthesisp (HParams): parameters for performing the synthesis. Parameters needed are
                                  synthesisp.output_path to save the spectrogram generated at
                                  each tier.
            timestamp (str): information that identifies completely this run (synthesis).
            logger (logging.Logger):
            n_samples (int): amount of samples to generate.
            length (int): length of the samples to generate (in timesteps).

        Returns:
            spectrograms (Tensor): samples of audio in spectrogram representation.
                                   Shape: [B=n_samples, FREQ=self.freq, FRAMES=length].
        """
        assert length >= 2 ** (
                self.n_tiers / 2), "Length is too short for being generated with the " \
                                   "number of tiers of this model."

        # Initially, the spectrogram (x) to generate it does not exist.
        x = None

        # --- TIER 1 ----
        # The spectrogram is generated autoregressively, frame (length, or timestep) by frame.
        logger.info(f"Starting Tier 1/{self.n_tiers}")
        freq_of_tier1 = tierutil.get_size_freqdim_of_tier(n_mels=self.freq, n_tiers=self.n_tiers,
                                                          tier=1)
        length_of_tier1 = tierutil.get_size_timedim_of_tier(timesteps=length, n_tiers=self.n_tiers,
                                                            tier=1)
        for i in range(0, length_of_tier1):
            logger.info(f"Tier 1/{self.n_tiers} - Frame {i}/{length_of_tier1}")
            if x is None:
                # If the spectrogram has not been initialized, we initialized to an initial frame
                # of all zeros
                x = torch.zeros((n_samples, freq_of_tier1, 1), device=hp.device)
            else:
                # If the spectrogram has already been initialized, we have already computed some
                # frames. We concatenate a new frame initialized to all zeros which will be replaced
                # pixel by pixel by the new values
                # We change the shape from [B, FREQ, FRAMES] to [B, FREQ, FRAMES+1] by adding a new
                # frame
                x = torch.cat(
                    [x, torch.zeros((n_samples, freq_of_tier1, 1), device=hp.device)], dim=-1)

            # Inside a frame, the spectrogram is generated autoregressively, freq by freq
            for j in range(0, freq_of_tier1):
                # we generate the parameters for all the spectrogram (across all samples)
                mu_hat, std_hat, pi_hat = self.tiers[0](x)
                # with the parameters we generate the values of the next spectrogram
                # (across all samples)
                new_spectrogram = sample_gmm_batch(mu_hat, std_hat, pi_hat)
                # but only use the value of the new pixel that we are generating
                # (across all samples) since the spectrogram is generated autoregressively
                x[:, j, i] = new_spectrogram[:, j, i]

        # Save spectrogram generated at tier1
        torch.save(x, f"{synthesisp.output_path}/{timestamp}_tier1.pt")

        # --- TIER >1 ---
        for tier_idx in range(2, self.n_tiers + 1):
            temp_x = None  # temporary spectrogram that will be generated by this tier
            # The spectrogram is generated autoregressively, frame (length, or timestep) by frame.
            logger.info(f"Starting Tier {tier_idx}/{self.n_tiers}")
            freq_of_tierX = tierutil.get_size_freqdim_of_tier(n_mels=self.freq,
                                                              n_tiers=self.n_tiers,
                                                              tier=tier_idx)
            length_of_tierX = tierutil.get_size_timedim_of_tier(timesteps=length,
                                                                n_tiers=self.n_tiers,
                                                                tier=tier_idx)
            for i in range(0, length_of_tierX):
                logger.info(f"Tier {tier_idx}/{self.n_tiers} - Frame {i}/{length_of_tierX}")
                if temp_x is None:
                    # If the spectrogram of this tier has not been initialized, we initialized to an
                    # initial frame of all zeros
                    temp_x = torch.zeros((n_samples, freq_of_tierX, 1), device=hp.device)
                else:
                    # If the spectrogram of this tier has already been initialized, we have already
                    # computed some frames. We concatenate a new frame initialized to all zeros
                    # which will be replaced pixel by pixel by the new values
                    # We change shape from [B, FREQ, FRAMES] to [B, FREQ, FRAMES+1] by adding a new
                    # frame
                    temp_x = torch.cat(
                        [temp_x, torch.zeros((n_samples, freq_of_tierX, 1), device=hp.device)],
                        dim=-1)

                # Inside a frame, the spectrogram is generated autoregressively, freq by freq
                for j in range(0, freq_of_tierX):
                    # we generate the parameters for all the spectrogram (across all samples)
                    mu_hat, std_hat, pi_hat = self.tiers[tier_idx - 1](temp_x, x)
                    # with the parameters we generate the values of the next spectrogram
                    # (across all samples)
                    new_spectrogram = sample_gmm_batch(mu_hat, std_hat, pi_hat)
                    # but only use the value of the new pixel that we are generating
                    # (across all samples) since the spectrogram is generated autoregressively
                    temp_x[:, j, i] = new_spectrogram[:, j, i]

            # After generating the spectrogram of this tier, we interleave it to put it together
            # with the spectrogram generated by previous tiers. In the next iteration, this will
            # be the input to condition the next tier
            x = tierutil.interleave(temp_x, x, tier_idx)
            x = x.to(hp.device)
            # Save spectrogram generated at tier1
            torch.save(temp_x, f"{synthesisp.output_path}/{timestamp}_tier{tier_idx}.pt")
            torch.save(x, f"{synthesisp.output_path}/{timestamp}_tier1-tier{tier_idx}.pt")

        return x

    def load_tiers(self, checkpoints_path: List[str], logger: logging.Logger) -> None:
        """
        Loads the weights of the trained tiers into MelNet.

        Args:
            checkpoints_path (List[str]): path to the weights of the tiers.
            logger (logging.Logger):
        """
        if len(checkpoints_path) != self.n_tiers:
            logger.error(f"Number of checkpoints tiers ({len(checkpoints_path)}) is different from "
                         f"the number of tiers of current model ({self.n_tiers})")
            raise Exception(
                f"Number of checkpoints tiers ({len(checkpoints_path)}) is different from "
                f"the number of tiers of current model ({self.n_tiers})")

        for tier_idx, checkpoint_path in enumerate(checkpoints_path):
            # Load weights from previously trained tier
            if not Path(checkpoint_path).exists():
                logger.error(
                    f"Path for tier {tier_idx} with weigths {checkpoint_path} does not exist.")
                raise Exception(
                    f"Path for tier {tier_idx} with weigths {checkpoint_path} does not exist.")

            logger.info(f"Loading tier {tier_idx} with weights {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)

            self.tiers[tier_idx].load_state_dict(checkpoint["tier"])
