"""
Tier

Implementation of the Tier, which is the basic unit in the multiscale modelling as explained in
Section 6 of the MelNet paper.

A Tier is composed of Delayed Stack Layers and, at the final layer, a linear transformation to the
output (hidden state) of the frequency-delayed stack to produce the unconstrained parameters:
mu_hat, std_hat, pi_hat.

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

from torch import Tensor
import torch.nn as nn

from src.model.DelayedStack import DelayedStackLayer0, DelayedStackLayer


# For now, the implementation of Tier is only useful for initial tier for unconditional
# speech generation
class Tier(nn.Module):
    """Tier of the multiscale modelling

    This tier contains a list of delayed stack layers as explained in Section 6.

    Examples::

        >> layers = [5, 6, ..., 4]
        >> hidden_size = ...
        >> gmm_size = ...
        >> freq = ...
        >> tiers = nn.ModuleList([Tier(tier=tier_idx,
                                       n_layers=layers[tier_idx],
                                       hidden_size=hidden_size,
                                       gmm_size=gmm_size,
                                       freq=freq)
                                 for tier_idx in range(n_tiers)])
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

        # Only the initial tier uses a centralized stack according to MelNet paper (Table 1)
        self.has_central_stack = True if tier == 1 else False

        # Define layers of the tier
        self.layers = nn.ModuleList(
                                    [DelayedStackLayer0(hidden_size=hidden_size,
                                                        has_central_stack=self.has_central_stack,
                                                        freq=freq)]
                                    +
                                    [DelayedStackLayer(layer=layer_idx,
                                                       hidden_size=hidden_size,
                                                       has_central_stack=self.has_central_stack)
                                     for layer_idx in range(1, n_layers)]
                                    )

        # Linear transformation from final layer of the frequency-delayed stack to produce
        # unconstrained parameters
        self.W_theta = nn.Linear(in_features=hidden_size, out_features=3 * self.k)

    def forward(self, spectrogram: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the unconstrained parameters of the GMM of a tier.
        If it is the initial tier, the parameters are generated unconditionally.
        If it is other tier, the parameters are generated conditionally on the output of previous
        tiers.

        Args:
            spectrogram (Tensor): input spectrogram.
                                  It will be constructed autoregressively, so in the beginning it
                                  will be artificial values (all 0, random, etc.). Later, the
                                  spectrogram will be built 'pixel' by 'pixel' adding to the initial
                                  spectrogram by feeding the increasing spectrogram to this
                                  module (tier).
                                  Shape: [B, FREQ, FRAMES]

        Returns:
            mu_hat (Tensor): means of GMM with k components. Shape: [B, FREQ, FRAMES, K]
            std_hat (Tensor): std of GMM with k components. Shape: [B, FREQ, FRAMES, K]
            pi_hat (Tensor): pi of GMM with k components. Shape: [B, FREQ, FRAMES, K]
        """
        # Hidden states of layer zero for time-delayed, frequency-delayed and centralized stack
        h_t, h_f, h_c = self.layers[0](spectrogram)

        # Hidden states of every layer for time-delayed, frequency-delayed and centralized stack.
        # Shapes of hidden states are always:
        # - Time-delayed stack and Frequency-delayed stack: [B, FREQ, FRAMES, HIDDEN_SIZE]
        # - Centralized stack: [B, 1, FRAMES, HIDDEN_SIZE]
        for layer_idx in range(1, self.n_layers):
            h_t, h_f, h_c = self.layers[layer_idx](h_t, h_f, h_c)

        # At final layer, a linear transformation is applied to the output of the frequency-delayed
        # stack to produce the unconstrained parameters according to MelNet formula (10)
        theta_hat = self.W_theta(h_f)

        # Split theta_hat into the unconstrained parameters from
        # shape [B, FREQ, FRAMES, 3K] to three tensors of shape [B, FREQ, FRAMES, K]
        mu_hat, std_hat, pi_hat = theta_hat.split(split_size=self.k, dim=-1)

        return mu_hat, std_hat, pi_hat
