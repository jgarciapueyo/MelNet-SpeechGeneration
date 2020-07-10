# TODO: add header to file
from typing import Tuple

from torch import Tensor
import torch.nn as nn

from model.DelayedStack import DelayedStackLayer0, DelayedStackLayer


# For now, the implementation of Tier is only useful for initial tier for unconditional
# speech generation
class Tier(nn.Module):

    # TODO: change n_layers, hidden_size, FREQ with hpparams or other structure that contains all
    #  the information together
    def __init__(self, tier: int, n_layers: int, hidden_size: int, FREQ: int, K: int):
        super(Tier, self).__init__()

        self.tier = tier
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.K = K

        # Only the initial tier uses a centralized stack according to MelNet paper (Table 1)
        self.has_central_stack = True if tier == 1 else False

        # Define layers of the tier
        self.layer0 = DelayedStackLayer0(hidden_size=hidden_size,
                                         has_central_stack=self.has_central_stack,
                                         FREQ=FREQ)
        self.layers = nn.ModuleList([DelayedStackLayer(layer=layer_idx,
                                                       hidden_size=hidden_size,
                                                       has_central_stack=self.has_central_stack)
                                     for layer_idx in range(1, n_layers)])

        # Linear transformation from final layer of the frequency-delayed stack
        self.W_theta = nn.Linear(in_features=hidden_size, out_features=3 * K)

    def forward(self, spectrogram: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the unconstrained parameters of the GMM of a tier.
        If it is the initial tier, the parameters are generated unconditionally.
        If it is other tier, the parameters are generated conditionally on the output of previous
        tiers.

        Args:
            # FIXME: improve this description of the input
            spectrogram (Tensor): input spectrogram that will be constructed autoregressively.
                                  Shape: [B, FREQ, FRAMES]

        Returns:
            mu_hat (Tensor): means of GMM with k components. Shape: [B, FREQ, FRAMES, K]
            std_hat (Tensor): std of GMM with k components. Shape: [B, FREQ, FRAMES, K]
            pi_hat (Tensor): pi of GMM with k components. Shape: [B, FREQ, FRAMES, K]
        """
        # Hidden states of layer zero for time-delayed, frequency-delayed and centralized stack
        h_t, h_f, h_c = self.layer0(spectrogram)

        # Hidden states of every layer for time-delayed, frequency-delayed and centralized stack.
        # Shapes of hidden states are always:
        # - Time-delayed stack and Frequency-delayed stack: [B, FREQ, FRAMES, HIDDEN_SIZE]
        # - Centralized stack: [B, 1, FRAMES, HIDDEN_SIZE]
        for layer in self.layers:
            h_t, h_f, h_c = layer(h_t, h_f, h_c)

        # At final layer, a linear transformation is applied to the output of the frequency-delayed
        # stack to produce the unconstrained parameters according to MelNet formula (10)
        theta_hat = self.W_theta(h_f)

        # Split theta_hat into the unconstrained parameters from
        # shape [B, FREQ, FRAMES, 3K] to three tensors of shape [B, FREQ, FRAMES, K]
        mu_hat, std_hat, pi_hat = theta_hat.split(split_size=self.K, dim=-1)

        return mu_hat, std_hat, pi_hat
