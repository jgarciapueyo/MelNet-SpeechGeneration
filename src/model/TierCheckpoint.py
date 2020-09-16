"""Implementation of the Tier using checkpointing.

In MelNet, the Tier is the basic unit in the multiscale modelling as explained in Section 6 of the
MelNet paper. A Tier is composed of Delayed Stack Layers and, at the final layer, a linear
transformation to the output (hidden state) of the frequency-delayed stack to produce the
unconstrained parameters: mu_hat, std_hat, pi_hat.

This module uses checkpoint technique to allow for a bigger model as explained in the module
src.model.ModuleWrapper. To learn more about checkpoint in PyTorch see:
https://pytorch.org/docs/stable/checkpoint.html.

For more information, see notebooks/07_TierDimensionsAndParameters.ipynb
# TODO: finish explanation in notebooks/07_TierDimensionsAndParameters.ipynb

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
from typing import Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from src.model.DelayedStack import DelayedStackLayer0, DelayedStackLayer
from src.model.FeatureExtraction import FeatureExtractionLayer
from src.model.ModuleWrapper import ModuleWrapperDelayedStackLayer
from src.model.ModuleWrapper import ModuleWrapperDelayedStackLayer0
from src.model.ModuleWrapper import ModuleWrapperFeatureExtraction


class Tier1(nn.Module):
    """First tier of MelNet (multiscale modelling)

    This tier contains a list of delayed stack layers as explained in Section 6.

    Examples::

        >> layers = [5, 6, ..., 4]
        >> hidden_size = ...
        >> gmm_size = ...
        >> freq = ...
        >> tier1 = Tier1(tier=1,
                         n_layers=layers[0],
                         hidden_size=hidden_size,
                         gmm_size=gmm_size,
                         freq=freq)
        >> mu_hat, std_hat, pi_hat = tier1(spectrogram)
    """

    def __init__(self, tier: int, n_layers: int, hidden_size: int, gmm_size: int, freq: int):
        """
        Args:
            tier (int): the tier that this module represents.
            n_layers (int): number of layers this tier is composed of.
            hidden_size (int): parameter for the hidden_state of the Delayed Stack Layers
            gmm_size (int): number of mixture components of the GMM
            freq (int): size of the frequency axis of the spectrogram to generate. See note in the
                        documentation of the file.
        """
        super(Tier1, self).__init__()

        self.tier = tier
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.k = gmm_size

        # Only the initial tier uses a centralized stack according to MelNet paper (Table 1)
        self.has_central_stack = True

        # Define layers of the tier
        self.layers = nn.ModuleList(
            [ModuleWrapperDelayedStackLayer0(
                DelayedStackLayer0(tier=tier,
                                   hidden_size=hidden_size,
                                   has_central_stack=self.has_central_stack,
                                   freq=freq)
            )]
            +
            [ModuleWrapperDelayedStackLayer(
                DelayedStackLayer(tier=tier,
                                  layer=layer_idx,
                                  hidden_size=hidden_size,
                                  has_central_stack=self.has_central_stack,
                                  freq=freq)
            ) for layer_idx in range(1, n_layers)]
        )

        # Define dummy tensor to trick checkpointing
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # Linear transformation from final layer of the frequency-delayed stack to produce
        # unconstrained parameters
        self.W_theta = nn.Linear(in_features=hidden_size, out_features=3 * self.k)

    def forward(self, spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the unconstrained parameters of the GMM of the first tier.
        If it is the initial tier, the parameters are generated unconditionally.
        If it is other tier, the parameters are generated conditionally on the the spectrogram of
        previous tiers.

        Args:
            spectrogram (torch.Tensor): input spectrogram.
                                  It will be constructed autoregressively, so in the beginning it
                                  will be artificial values (all 0, random, etc.). Later, the
                                  spectrogram will be built 'pixel' by 'pixel' adding to the initial
                                  spectrogram by feeding the increasing spectrogram to this
                                  module (tier).
                                  Shape: [B, FREQ, FRAMES]

        Returns:
            mu_hat (torch.Tensor): means of GMM with k components. Shape: [B, FREQ, FRAMES, K]
            std_hat (torch.Tensor): std of GMM with k components. Shape: [B, FREQ, FRAMES, K]
            pi_hat (torch.Tensor): pi of GMM with k components. Shape: [B, FREQ, FRAMES, K]
        """
        # Hidden states of layer zero for time-delayed, frequency-delayed and centralized stack
        # h_t, h_f, h_c = self.layers[0](spectrogram)  #-> call layer0 without checkpointing
        h_t, h_f, h_c = torch.utils.checkpoint.checkpoint(self.layers[0], self.dummy_tensor,
                                                          spectrogram, None)

        # Hidden states of every layer for time-delayed, frequency-delayed and centralized stack.
        # Shapes of hidden states are always:
        # - Time-delayed stack and Frequency-delayed stack: [B, FREQ, FRAMES, HIDDEN_SIZE]
        # - Centralized stack: [B, 1, FRAMES, HIDDEN_SIZE]
        for layer_idx in range(1, self.n_layers):
            # h_t, h_f, h_c = self.layers[layer_idx](h_t, h_f, h_c)  #-> call layerX
            #                                                            without checkpointing
            h_t, h_f, h_c = torch.utils.checkpoint.checkpoint(self.layers[layer_idx],
                                                              self.dummy_tensor, h_t, h_f, h_c)

        # At final layer, a linear transformation is applied to the output of the frequency-delayed
        # stack to produce the unconstrained parameters according to MelNet formula (10)
        theta_hat = self.W_theta(h_f)

        # Split theta_hat into the unconstrained parameters from
        # shape [B, FREQ, FRAMES, 3K] to three tensors of shape [B, FREQ, FRAMES, K]
        mu_hat, std_hat, pi_hat = theta_hat.split(split_size=self.k, dim=-1)

        return mu_hat, std_hat, pi_hat


class Tier(nn.Module):
    """
    Tier of MelNet (multiscale modelling)

    This tier contains a list of delayed stack layers as explained in Section 6.

    .. Note:
        This module is valid for the tiers greater than 1.

    Examples::

        >> layers = [5, 6, ..., 4]
        >> hidden_size = ...
        >> gmm_size = ...
        >> freq = ...
        >> tiers = nn.ModuleList([Tier(tier=tier_idx+1,
                                       n_layers=layers[tier_idx],
                                       hidden_size=hidden_size,
                                       gmm_size=gmm_size,
                                       freq=freq)
                                 for tier_idx in range(1, n_tiers)])
    """

    def __init__(self, tier: int, n_layers: int, hidden_size: int, gmm_size: int, freq: int):
        """
        Args:
            tier (int): the tier that this module represents.
            n_layers (int): number of layers this tier is composed of.
            hidden_size (int): parameter for the hidden_state of the Delayed Stack Layers
            gmm_size (int): number of mixture components of the GMM
            freq (int): size of the frequency axis of the spectrogram to generate. See note in the
                        documentation of the file.
        """
        super(Tier, self).__init__()

        self.tier = tier
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.k = gmm_size
        self.freq = freq

        # Only the initial tier uses a centralized stack according to MelNet paper (Table 1)
        self.has_central_stack = False

        # Define layers of the tier
        self.layers = nn.ModuleList(
            [ModuleWrapperDelayedStackLayer0(
                DelayedStackLayer0(tier=tier,
                                   hidden_size=hidden_size,
                                   has_central_stack=self.has_central_stack,
                                   freq=freq,
                                   is_conditioned=True,
                                   hidden_size_condition=hidden_size * 4)
            )]
            +
            [ModuleWrapperDelayedStackLayer(
                DelayedStackLayer(tier=tier,
                                  layer=layer_idx,
                                  hidden_size=hidden_size,
                                  has_central_stack=self.has_central_stack,
                                  freq=freq)
            ) for layer_idx in range(1, n_layers)]
        )
        # The Layer 0 of this tier (greater than first tier) is conditioned on the output of the
        # feature extraction network. These conditioning features are the concatenation of the
        # hidden state of 4 one dimensional RNN, that's why the hidden size of the condition in
        # DelayedStackLayer0 has * 4

        # Define feature extraction network
        self.feature_extraction = ModuleWrapperFeatureExtraction(
            FeatureExtractionLayer(hidden_size))

        # Define dummy tensor to trick checkpointing
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # Linear transformation from final layer of the frequency-delayed stack to produce
        # unconstrained parameters
        self.W_theta = nn.Linear(in_features=hidden_size, out_features=3 * self.k)

    def forward(self, spectrogram: torch.Tensor, spectrogram_prev_tier: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the unconstrained parameters of the GMM of a tier.
        If it is the initial tier, the parameters are generated unconditionally.
        If it is other tier, the parameters are generated conditionally on the spectrogram of
        previous tiers.

        Args:
            spectrogram (torch.Tensor): input spectrogram.
                              It will be constructed autoregressively, so in the beginning it
                              will be artificial values (all 0, random, etc.). Later, the
                              spectrogram will be built 'pixel' by 'pixel' adding to the initial
                              spectrogram by feeding the increasing spectrogram to this
                              module (tier).
                              Shape: [B, FREQ, FRAMES]
            spectrogram_prev_tier (torch.Tensor): spectrogram generated by the previous tiers.
                              It is the spectrogram used to condition the unconstrained
                              parameters of the GMM of this tier. In the paper, the spectrogram
                              generated by previous tiers is named x^<g.
                              Shape: [B, FREQ, FRAMES]

        Returns:
        mu_hat (torch.Tensor): means of GMM with k components. Shape: [B, FREQ, FRAMES, K]
        std_hat (torch.Tensor): std of GMM with k components. Shape: [B, FREQ, FRAMES, K]
        pi_hat (torch.Tensor): pi of GMM with k components. Shape: [B, FREQ, FRAMES, K]
        """
        B, FREQ, FRAMES = spectrogram.size()
        # Calculate conditioning features from the spectrograms generated by the previous tiers.
        # In the paper, these conditioning features are named as z (Section 4.4)
        # conditioning = self.feature_extraction(spectrogram_prev_tier) #-> call feature extraction
        #                                                                   without checkpointing
        conditioning = torch.utils.checkpoint.checkpoint(self.feature_extraction, self.dummy_tensor,
                                                         spectrogram_prev_tier)
        # We only take the condition features until the current frame (there is an error later in
        # the forward pass of DelayedStack0 that broadcast along the time dimension)
        conditioning = conditioning[:, :, :FRAMES]

        # Hidden states of layer zero for time-delayed and frequency-delayed stack
        # h_t, h_f, _ = self.layers[0](spectrogram, conditioning)  #-> call layer0
        #                                                              without checkpointing
        h_t, h_f = torch.utils.checkpoint.checkpoint(self.layers[0], self.dummy_tensor,
                                                     spectrogram, conditioning)

        # Hidden states of every layer for time-delayed and frequency-delayed.
        # Shapes of hidden states are always:
        # - Time-delayed stack and Frequency-delayed stack: [B, FREQ, FRAMES, HIDDEN_SIZE]
        for layer_idx in range(1, self.n_layers):
            # h_t, h_f, _ = self.layers[layer_idx](h_t, h_f, None)  #-> call layerX
            #                                                           without checkpointing
            h_t, h_f = torch.utils.checkpoint.checkpoint(self.layers[layer_idx],
                                                         self.dummy_tensor, h_t, h_f)

        # At final layer, a linear transformation is applied to the output of the frequency-delayed
        # stack to produce the unconstrained parameters according to MelNet formula (10)
        theta_hat = self.W_theta(h_f)

        # Split theta_hat into the unconstrained parameters from
        # shape [B, FREQ, FRAMES, 3K] to three tensors of shape [B, FREQ, FRAMES, K]
        mu_hat, std_hat, pi_hat = theta_hat.split(split_size=self.k, dim=-1)

        return mu_hat, std_hat, pi_hat
