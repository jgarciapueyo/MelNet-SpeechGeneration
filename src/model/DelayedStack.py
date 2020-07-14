"""
Delayed Stack

Implementation of the layers that compose the Delayed Stack as explained in Section 4 of the MelNet
paper.

A Delayed Stack Layer is composed of three stacks:
- Time-delayed stack: composed of three RNN (forward in time, forward in frequency and backward in
                      frequency)
- Frequency-delayed stack: composed of one RNN (forward in frequency)
- Centralized stack: composed of one RNN (forward in time)

For more information, see: notebooks/06_DelayedStackDimensions.ipynb
# TODO: finish explanation in notebooks/06_DelayedStackDimensions.ipynb

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
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class DelayedStackLayer(nn.Module):
    """Layer of the Delayed Stack

    This module contains a layer of the time-delayed stack, frequency-delayed stack and
    central-delayed stack as explained in Section 4.

    .. Note:
        This module is valid for the layers greater than 0.

    Examples::

        >> B, FREQ, FRAMES = spectrogram.shape
        >> hidden_size = ...
        >> has_central_stack =
        >> layers = nn.ModuleList([DelayedStackLayer(layer=l,
                                                     hidden_size=hidden_size,
                                                     has_central_stack=has_central_stack)
                                   for l in range(num_layers)])
        >> for layer in layers:
        >>     h_t, h_f, h_c = layer(h_t, h_f, h_c)
    """

    def __init__(self, layer: int, hidden_size: int, has_central_stack: bool):
        """
        Args:
            layer (int): the layer that this module represents.
            hidden_size (int): hidden_size parameter for the hidden_state shape of the RNN.
            has_central_stack (bool): true if the layer has a central stack.
        """
        super(DelayedStackLayer, self).__init__()

        self.layer = layer
        self.has_central_stack = has_central_stack

        # Layer of Time-Delayed Stack composed of a multidimensional RNN.
        # Each multidimensional RNN is composed of three one-dimensional RNN (Section 4.1)
        self.rnn_t_l_forwardtime = nn.LSTM(input_size=hidden_size,
                                           hidden_size=hidden_size,
                                           num_layers=1,
                                           batch_first=True)

        self.rnn_t_l_forwardfreq = nn.LSTM(input_size=hidden_size,
                                           hidden_size=hidden_size,
                                           num_layers=1,
                                           batch_first=True)

        self.rnn_t_l_backwardfreq = nn.LSTM(input_size=hidden_size,
                                            hidden_size=hidden_size,
                                            num_layers=1,
                                            batch_first=True)

        # in_features=hidden_size*3 because in_features is the concatenation of the
        # three RNN hidden states
        # NOTE: should bias=False?
        self.W_t_l = nn.Linear(in_features=hidden_size * 3, out_features=hidden_size)

        # Layer of Frequency-Delayed Stack composed of a one-dimensional RNN (Section 4.2)
        self.rnn_f_l = nn.LSTM(input_size=hidden_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               batch_first=True)

        self.W_f_l = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        # Layer of Centralized Stack composed of a RNN (Section 4.3)
        self.rnn_c_l = nn.LSTM(input_size=hidden_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               batch_first=True)

        self.W_c_l = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(self, h_t_prev: Tensor, h_f_prev: Tensor, h_c_prev: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the hidden state for the time-delayed, frequency-delayed and central stack of
        the current layer using the previous hidden states.

        Args:
            h_t_prev (Tensor): time-delayed hidden state of previous layer.
                               Shape: [B, FREQ, FRAMES, HIDDEN_SIZE].
            h_f_prev (Tensor): frequency-delayed hidden state of previous layer.
                               Shape: [B, FREQ, FRAMES, HIDDEN_SIZE]
            h_c_prev (Tensor): central stack hidden state of previous layer.
                               Shape: [B, 1, FRAMES, HIDDEN_SIZE]

        Returns:
            h_t (Tensor): time-delayed hidden state of current layer.
                          Shape: [B, FREQ, FRAMES, HIDDEN_SIZE].
            h_f (Tensor): frequency-delayed hidden state of current layer.
                          Shape: [B, FREQ, FRAMES, HIDDEN_SIZE]
            h_c (Tensor): central stack hidden state of current layer.
                          Shape: [B, 1, FRAMES, HIDDEN_SIZE]
        """
        B, FREQ, FRAMES, HIDDEN_SIZE = h_t_prev.size()

        # ---- TIME-DELAYED STACK COMPUTATION ----

        # Figure 2(a)-1.
        # Every arrow is the same LSTM applied to different sequences of pixels.
        # We could see this as if every arrow computation was one example of the batch.
        # Change shape from [B, FREQ, FRAMES, HIDDEN_SIZE] to [B*FREQ, FRAMES, HIDDEN_SIZE]
        # to accommodate for every arrow computation as an example of the batch
        h_t_prev_forwardtime = h_t_prev.view(-1, FRAMES, HIDDEN_SIZE)
        h_t_forwardtime, _ = self.rnn_t_l_forwardtime(h_t_prev_forwardtime)
        h_t_forwardtime = h_t_forwardtime.view(B, FREQ, FRAMES, HIDDEN_SIZE)

        # Figure 2(a)-2.
        # Every arrow is the same LSTM applied to different sequences of pixels.
        # We could see this as if every arrow computation was one example of the batch.
        # Change shape from [B, FREQ, FRAMES, HIDDEN_SIZE] to [B, FRAMES, FREQ, HIDDEN_SIZE]
        h_t_prev_forwardfreq = h_t_prev.transpose(1, 2)
        # Change shape from [B, FRAMES, FREQ, HIDDEN_SIZE] to [B*FRAMES, FREQ, HIDDEN_SIZE]
        # to accommodate for every arrow computation as an example of the batch
        h_t_prev_forwardfreq = h_t_prev_forwardfreq.contiguous().view(-1, FREQ, HIDDEN_SIZE)
        h_t_forwardfreq, _ = self.rnn_t_l_forwardfreq(h_t_prev_forwardfreq)
        h_t_forwardfreq = h_t_forwardfreq.contiguous() \
            .view(B, FRAMES, FREQ, HIDDEN_SIZE) \
            .transpose(1, 2)

        # Figure 2(a)-3.
        # Every arrow is the same LSTM applied to different sequences of pixels.
        # We could see this as if every arrow computation was one example of the batch.
        # Change shape from [B, FREQ, FRAMES, HIDDEN_SIZE] to [B, FRAMES, FREQ, HIDDEN_SIZE]
        h_t_prev_backwardfreq = h_t_prev.transpose(1, 2)
        # Change shape from [B, FRAMES, FREQ, HIDDEN_SIZE] to [B*FRAMES, FREQ, HIDDEN_SIZE]
        # to accommodate for every arrow computation as an example of the batch
        h_t_prev_backwardfreq = h_t_prev_backwardfreq.contiguous().view(-1, FREQ, HIDDEN_SIZE)
        # Because the computation has to be backward with respect to the frequency, we reverse it
        h_t_prev_backwardfreq = h_t_prev_backwardfreq.flip(1)
        h_t_backwardfreq, _ = self.rnn_t_l_backwardfreq(h_t_prev_backwardfreq)
        h_t_backwardfreq = h_t_backwardfreq.contiguous() \
            .view(B, FRAMES, FREQ, HIDDEN_SIZE) \
            .transpose(1, 2)  # [B, FREQ, FRAMES, HIDDEN_SIZE]

        # Output of each layer of the time-delayed stack is the concatenation of the
        # three RNN hidden states across the last dimension (HIDDEN_SIZE)
        h_t = torch.cat((h_t_forwardtime, h_t_forwardfreq, h_t_backwardfreq), dim=3)
        # New time-delayed hidden state according to MelNet formula (6)
        h_t = self.W_t_l(h_t) + h_t_prev

        # ---- CENTRALIZED STACK COMPUTATION ----
        # At each time step, it takes an entire frame as input and outputs a single vector
        # consisting of the RNN hidden state
        h_c = None
        if self.has_central_stack:
            # Change from [B, 1, FRAMES, HIDDEN_SIZE] to [B*1, FRAMES, HIDDEN_SIZE]
            h_c_prev = h_c_prev.view(-1, FRAMES, HIDDEN_SIZE)
            h_c, _ = self.rnn_c_l(h_c_prev)
            # Change from [B*1, FRAMES, HIDDEN_SIZE] to [B, 1, FRAMES, HIDDEN_SIZE]
            h_c = h_c.contiguous().view(B, 1, FRAMES, HIDDEN_SIZE)

            # New central stack hidden state according to MelNet formula (11)
            h_c = self.W_c_l(h_c) + h_c_prev

        # ---- FREQUENCY-DELAYED STACK COMPUTATION ----
        h_f_in = None
        if self.has_central_stack:
            # Frequency-delayed stack takes three inputs which are summed and used as
            # input to the RNN
            h_f_in = h_f_prev + h_t + h_c
        else:
            # Frequency-delayed stack takes two inputs which are summed and used as input to the RNN
            h_f_in = h_f_prev + h_t

        # Frequency-delayed stack runs forward in frequency
        # Change shape from [B, FREQ, FRAMES, HIDDEN_SIZE] to [B, FRAMES, FREQ, HIDDEN_SIZE]
        h_f_in = h_f_in.transpose(1, 2)
        # Change shape from [B, FRAMES, FREQ, HIDDEN_SIZE] to [B*FRAMES, FREQ, HIDDEN_SIZE]
        # to accommodate for every arrow computation as an example of the batch
        h_f_in = h_f_in.contiguous().view(-1, FREQ, HIDDEN_SIZE)
        h_f, _ = self.rnn_f_l(h_f_in)
        h_f = h_f.contiguous() \
            .view(B, FRAMES, FREQ, HIDDEN_SIZE) \
            .transpose(1, 2)  # [B, FREQ, FRAMES, HIDDEN_SIZE]

        # New frequency-delayed hidden state according to to MelNet formula (8)
        # (modified to accommodate for central stack if necessary)
        h_f = self.W_f_l(h_f) + h_f_prev

        return h_t, h_f, h_c


class DelayedStackLayer0(nn.Module):
    """Layer zero of the Delayed Stack

    This module contains the layer zero of the time-delayed stack, frequency-delayed stack and
    central-delayed stack as explained in Section 4.

    Examples::

        >> B, FREQ, FRAMES = spectrogram.shape
        >> hidden_size = ...
        >> has_central_stack =
        >> l_0 = DelayedStackLayer0(hidden_size=hidden_size,
                                    has_central_stack=has_central_stack,
                                    FREQ=FREQ)
        >> h_t, h_f, h_c = l_0(spectrogram)
    """

    def __init__(self, hidden_size: int, has_central_stack: bool, freq: int):
        """
        Args:
            hidden_size (int): hidden_size parameter for the hidden_state shape of the RNN.
            has_central_stack (bool): true if the layer has a central stack.
            FREQ (int): size of the second dimension of the spectrogram.
                B, FREQ, FRAMES = spectrogram.shape
        """
        super(DelayedStackLayer0, self).__init__()

        self.hidden_size = hidden_size
        self.has_central_stack = has_central_stack
        self.freq = freq

        # Layer zero of Time-Delayed Stack composed of a linear transformation (Section 4.1)
        self.W_t_0 = nn.Linear(in_features=1,
                               out_features=hidden_size)

        # Layer zero of Frequency-Delayed Stack composed of a linear transformation (Section 4.2)
        self.W_f_0 = nn.Linear(in_features=1,
                               out_features=hidden_size)

        # Layer zero of Centralized Stack composed of a linear transformation (Section 4.3)
        self.W_c_0 = nn.Linear(in_features=freq,
                               out_features=hidden_size)

    def forward(self, spectrogram: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the hidden state for the time-delayed, frequency-delayed and central stack of
        the layer 0 using the spectrogram.

        Args:
            spectrogram (Tensor): spectrogram cuurently being generated which is used as input
                                  to the model. Shape: [B, FREQ, FRAMES]

        Returns:
            h_t_0 (Tensor): time-delayed hidden state of zero layer.
                          Shape: [B, FREQ, FRAMES, HIDDEN_SIZE].
            h_f_0 (Tensor): frequency-delayed hidden state of zero layer.
                          Shape: [B, FREQ, FRAMES, HIDDEN_SIZE]
            h_c_0 (Tensor): central stack hidden state of zero layer.
                          Shape: [B, 1, FRAMES, HIDDEN_SIZE]
        """
        B, FREQ, FRAMES = spectrogram.size()

        assert FREQ == self.freq, \
            "Current second dimension of spectrogram (FREQ) and previously declared FREQ " \
            "do not match"

        # ---- TIME-DELAYED STACK COMPUTATION ----

        # To ensure output h^t_ij[l] is only a function of frames which lie in the context x_<ij,
        # the inputs to the time-delayed stack are shifted backwards one step in time, so we
        # "invent" the first frame. In this case, we will assume that the first frame is all 0
        x_t_pad = F.pad(spectrogram, [1, -1])  # we put -1 to maintain the number of FRAMES equal
        # Change shape from [B, FREQ, FRAMES] to [B, FREQ, FRAMES, 1]
        x_t_pad = x_t_pad.unsqueeze(-1)
        # Time-delayed hidden-state of the layer 0 according to MelNet formula (7)
        h_t_0 = self.W_t_0(x_t_pad)

        # ---- FREQUENCY-DELAYED STACK COMPUTATION
        # To ensure output h^f_ij[l] is only a function of frames which lie in the context x_<ij,
        # the inputs to the frequency-delayed stack are shifted backwards one step along the
        # frequency axis, so we "invent" the "first (lowest)" frequency for all frames.
        # In this case, we are going to assume that the "first (lowest)" frequency is 0.
        x_f_pad = F.pad(spectrogram, [0, 0, 1, -1])  # we put -1 to maintain the number of FREQ
        # Change shape from [B, FREQ, FRAMES] to [B, FREQ, FRAMES, 1]
        x_f_pad = x_f_pad.unsqueeze(-1)
        # Frequency-delayed hidden-state of the layer 0 according to MelNet formula (9)
        h_f_0 = self.W_f_0(x_f_pad)

        # ---- CENTRALIZED STACK COMPUTATION ----
        h_c_0 = None
        if self.has_central_stack:
            # To ensure output h^c_i[l] is only a function of frames which lie in the context x_<ij,
            # the inputs to the centralized stack are shifted backwards one step along the time
            # axis, so we "invent" the "first (lowest)" frequency for all frames.
            # In this case, we are going to assume that the frame is all 0.
            x_c_pad = F.pad(spectrogram, [1, -1])  # we put -1 to maintain the number of FRAMES
            # The central stack, at each timestep, it takes an entire frame as input and outputs a
            # single vector consisting of the RNN hidden state, so we manipulate the tensor.
            # Change shape from [B, FREQ, FRAMES] to [B, FRAMES, FREQ]
            x_c_pad = x_c_pad.transpose(1, 2)
            # Centralized hidden-state of the layer 0 according to MelNet formula (11)
            h_c_0 = self.W_c_0(x_c_pad)
            # Change shape from [B, FRAMES, HIDDEN_SIZE] to [B, 1, FRAMES, HIDDEN_SIZE]
            h_c_0 = h_c_0.unsqueeze(dim=1)

        return h_t_0, h_f_0, h_c_0
