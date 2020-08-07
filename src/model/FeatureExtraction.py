"""Feature Extraction Network

Implementation of the Feature Extraction network for multiscale modelling to compute the features
from spectrograms created by previous tiers and condition the generation of the spectrogram of the
current tier as explained in Section 6.1
"""
import torch.nn as nn
import torch
from torch import Tensor


class FeatureExtractionLayer(nn.Module):
    """
    Feature Extraction Network Layer

    This module contains the Feature Extraction Network to compute features from spectrograms
    generated from previous tiers. Every tier contains one layer of this Feature Extraction Network.

    Examples (inside one tier)::

        >> hidden_size = ...
        >> feature_extraction = FeatureExtractionLayer(hidden_size=hidden_size)
        >> condition = feature_extraction(spectrogram_prev_tier)
        >> h_t, h_f, _ = layer0(spectrogram, condition)
    """

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size (int): parameter for size the hidden_state of the RNN
        """
        super(FeatureExtractionLayer, self).__init__()

        self.hidden_size = hidden_size

        # Four one dimensional RNN which run bidirectionally along slices of both axes
        # My interpretation is that there are four one dimensional RNN (one for each direction:
        # up, down, left, right) or, similarly, two bidirectional RNN (one for each dimension:
        # along time and frequency axis)

        self.rnn_frequency = nn.LSTM(input_size=1,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     batch_first=True,
                                     bidirectional=True)

        self.rnn_time = nn.LSTM(input_size=1,
                                hidden_size=hidden_size,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

    def forward(self, prev_spectrogram: Tensor) -> Tensor:
        """
        Calculates the conditioning features (z in the formulae 13 and 14 of MelNet Paper) from the
        information (spectrogram) from the preceding tiers (x^<g) as explained in Section 6.1

        Args:
            prev_spectrogram (Tensor): spectrogram generated at previous tiers. In the paper, this
                                       described as x^<g.

        Returns:
            h_fextr (Tensor): hidden state of the feature extraction network. In the paper, this is
                              described as z.
        """
        B, FREQ, FRAMES = prev_spectrogram.size()
        INITIAL_HIDDEN_SIZE = 1

        # As explained in Section 6.1, a layer of the feature extraction network is similar to a
        # layer of the time-delayed stack, so we can use the Figure 2 to understand it better.

        # Figure 2(a)-1: Every arrow is the same LSTM applied to different sequences of pixels.
        # We could see this as if every arrow computation was one example of the batch. The only
        # difference is that now the LSTM are bidirectional.
        # Add 4 dimension (hidden size): change shape from [B, FREQ, FRAMES] to [B, FREQ, FRAMES, 1]
        prev_spectrogram_time = prev_spectrogram.unsqueeze(-1)
        # Change shape from [B, FREQ, FRAMES, 1] to [B*FREQ, FRAMES, 1]
        # to accommodate for every arrow computation as an example of the batch
        prev_spectrogram_time = prev_spectrogram_time.contiguous().view(-1, FRAMES, INITIAL_HIDDEN_SIZE)
        # Calculate the feature_extraction hidden states
        h_fextr_time, _ = self.rnn_time(prev_spectrogram_time)
        h_fextr_time = h_fextr_time.view(B, FREQ, FRAMES, self.hidden_size * 2)
        # (*2 because bidirectional)

        # Figure 2(a)-2,3: Every arrow is the same LSTM applied to different sequences of pixels.
        # We could see this as if every arrow computation was one example of the batch. The only
        # difference is that now the LSTM are bidirectional.
        # Add 4 dimension (hidden size): change shape from [B, FREQ, FRAMES] to [B, FREQ, FRAMES, 1]
        prev_spectrogram_freq = prev_spectrogram.unsqueeze(-1)
        # Change shape from [B, FREQ, FRAMES, 1] to [B, FRAMES, FREQ, 1]
        prev_spectrogram_freq = prev_spectrogram_freq.transpose(1, 2)
        # Change shape from [B, FRAMES, FREQ, 1] to [B*FRAMES, FREQ, 1]
        # to accomodate for every arrow computation as an example of the batch
        prev_spectrogram_freq = prev_spectrogram_freq.contiguous().view(-1, FREQ,
                                                                        INITIAL_HIDDEN_SIZE)
        h_fextr_freq, _ = self.rnn_frequency(prev_spectrogram_freq)
        h_fextr_freq = h_fextr_freq.contiguous() \
            .view(B, FRAMES, FREQ, self.hidden_size * 2) \
            .transpose(1, 2)
        # (*2 because bidirectional)

        # Similar to time-delayed stack layer, the output of each layer of the feature extraction
        # network is the concatenation of the four one dimensional RNN (or two bidirectional RNN)
        h_fextr = torch.cat((h_fextr_time, h_fextr_freq), dim=3)
        # Hidden states of feature extraction has shape: [B, FREQ, FRAMES, self.hidden_size * 4]
        return h_fextr
