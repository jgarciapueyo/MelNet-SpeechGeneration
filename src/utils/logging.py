"""
This module contains classes to log the training process as well as save the logs in a Tensorboard
format.
"""
import io

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from PIL import Image
import numpy as np

from src.dataprocessing import transforms as T
from src.utils.hparams import HParams


class CustomSummaryWriter(SummaryWriter):
    """
    Custom SummaryWriter that does not create a subfolder inside the folder for a given run when
    calling add_hparams(...). (Default SummaryWriter creates a subfolder in the run folder every
    time that the method add_hparams(...) is called)
    """

    def add_hparams(self, hparam_dict, metric_dict):
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict (dict): Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value.
            metric_dict (dict): Each key-value pair in the dictionary is the
              name of the metric and it's corresponding value. Note that the key used
              here should be unique in the tensorboard record. Otherwise the value
              you added by ``add_scalar`` will be displayed in hparam plugin. In most
              cases, this is unwanted.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        # ---- Previously, add_hparams() added a subfolder inside each run ----
        # logdir = os.path.join(
        #     self._get_file_writer().get_logdir(),
        #     str(time.time())
        # )

        # ---- Now, it does not add any subfolder ----
        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


class TensorboardWriter:
    """
    Writes logs to be visualized in Tensorboard.
    """

    def __init__(self, hp: HParams, run_dir: str):
        """
        Args:
            hp (HParams): parameters for logging. Parameters used are hp.logging.dir_log_tensorboard
            run_dir (str): directory to save logs of the current run.
        """
        self.hp = hp
        self.writer = CustomSummaryWriter(log_dir=run_dir)

    def log_training(self, hp: HParams, loss: int, epoch: int):
        """
        Logs information related to training.

        Args:
            loss (int): loss during training
        """
        tag = f"{hp.data.dataset}/train/loss"
        self.writer.add_scalar(tag=tag, scalar_value=loss, global_step=epoch)

    def log_end_training(self, hp: HParams, loss: int):
        """
        Logs parameters of the training and final loss.

        Args:
            hp (HParams): parameters of the training of this run.
            loss (int): final loss of the model.
        """
        params = get_important_hparams(hp)
        loss_tag = f"{hp.data.dataset}/train/loss_global"
        metrics = {loss_tag: loss}
        self.writer.add_hparams(hparam_dict=params, metric_dict=metrics)

    def log_synthesis(self, spectrogram: Tensor):
        """
        Logs the spectrogram produced in synthesis

        Args:
            spectrogram (Tensor):
        """
        # use a buffer to save the image
        buf = io.BytesIO()
        T.save_spectrogram(buf, spectrogram, self.hp)
        buf.seek(0)
        # transform image in buffer to torch tensor
        spectrogram = np.array(Image.open(buf))
        # remove fourth channel (transparency)
        spectrogram = spectrogram[..., :3]
        # add image
        self.writer.add_image("synthesize", spectrogram, dataformats="HWC")

    def close(self):
        self.writer.close()


def get_important_hparams(hp: HParams) -> dict:
    """
    Get the important params for training to store them in tensorboard logs
    Args:
        hp (HParams): parameters

    Returns:
        important_params (dict)
    """
    important_params = dict()
    # dataset
    important_params["data.dataset"] = hp.data.dataset

    # audio
    important_params["audio.sample_rate"] = hp.audio.sample_rate
    important_params["audio.spectrogram_type"] = hp.audio.spectrogram_type
    important_params["audio.n_fft"] = hp.audio.n_fft
    important_params["audio.mel_channels"] = hp.audio.mel_channels
    important_params["audio.hop_length"] = hp.audio.hop_length
    important_params["audio.win_length"] = hp.audio.win_length
    important_params["audio.ref_level_db"] = hp.audio.ref_level_db
    important_params["audio.min_level_db"] = hp.audio.min_level_db

    # network
    important_params["network.n_tiers"] = hp.network.n_tiers
    important_params["network.layers"] = '_'.join(map(str, hp.network.layers))
    important_params["network.hidden_size"] = hp.network.hidden_size
    important_params["network.gmm_size"] = hp.network.gmm_size

    # training
    important_params["training.optimizer"] = hp.training.optimizer
    important_params["training.epochs"] = hp.training.epochs
    important_params["training.batch_size"] = hp.training.batch_size
    important_params["training.lr"] = hp.training.lr
    important_params["training.momentum"] = hp.training.momentum

    return important_params
