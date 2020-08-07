"""Wrapper around Delayed Stack Layer and Feature Extracion layer to allow for checkpointing.

MelNet size is controlled by several parameters: n_tiers, n_layers for each tier and hidden_size of
the LSTM that compose the layers. If the parameters of the tier are used, the model is very big
and impossible to train in a consumer grade GPU. The initial model fits but when performing forward
and backward passes, because PyTorch stores the gradient for every parameter, the memory usage
explodes and the model does not fit. (i.e. In a NVIDIA GeForce RTX 2080 with 8GB of VRAM, the
for a MelNet model of 6 tiers with layers [12, 6, 5, 4, 2, 2] the maximum hidden_size was 16).

PyTorch offers a way to reduce GPU memory by only saving the gradients of specific tensors and
recomputing them in the backward pass. This obviously increases training time. To learn more see:
https://pytorch.org/docs/stable/checkpoint.html.

When using checkpoints, it requires that at least one of the inputs to the checkpoint function
has requires_grad=True to perform backpropagation in the module that has been checkpointed. Because
the way a MelNet Tier is constructed, layers are a perfect point to do checkpoint. However, the
input to one layer are tensors that do not have requires_grad=True, so when performing checkpoint,
backpropagation is None.

To fix this, we can pass a dummy input which requires grad and not use it in the computation of the
layer. This is the purpose of this module, to wrap the real module that we want to run and pass a
dummy input which requires grad to trick checkpointing and be able to perform backpropagation.

The results of using checkpointing is that with the same architecture explained before, now the
maximum hidden_size is 200 (and maybe could be increased a little bit more).
"""
import torch.nn as nn


class ModuleWrapperDelayedStackLayer0(nn.Module):
    """Module Wrapper around Delayed Stack Layer 0 to allow for checkpointing.

    Input and output are the same (except for the dummy input).
    """

    def __init__(self, module):
        super(ModuleWrapperDelayedStackLayer0, self).__init__()
        self.module = module

    def forward(self, dummy_arg, spectrogram, condition):
        assert dummy_arg is not None
        return self.module(spectrogram, condition)


class ModuleWrapperDelayedStackLayer(nn.Module):
    """Module Wrapper around Delayed Stack Layer to allow for checkpointing.

        Input and output are the same (except for the dummy input).
    """

    def __init__(self, module):
        super(ModuleWrapperDelayedStackLayer, self).__init__()
        self.module = module

    def forward(self, dummy_arg, h_t_prev, h_f_prev, h_c_prev=None):
        assert dummy_arg is not None
        return self.module(h_t_prev, h_f_prev, h_c_prev)


class ModuleWrapperFeatureExtraction(nn.Module):
    """Module Wrapper around Feature Extraction Layer to allow for checkpointing.

        Input and output are the same (except for the dummy input).
    """

    def __init__(self, module):
        super(ModuleWrapperFeatureExtraction, self).__init__()
        self.module = module

    def forward(self, dummy_arg, prev_spectrogram):
        assert dummy_arg is not None
        return self.module(prev_spectrogram)
