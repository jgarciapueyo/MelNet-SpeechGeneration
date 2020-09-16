"""
This module contains the functions needed to create (synthesize) one sample.

It only needs the information where the hyperparameters yaml file is (HParams) and the path to the
weights of the model.

Example::
    >> args = ... (dict with the path to the hyperparameters yaml file, the path to the weights to
                   the model, the timesteps to generate and the output path)
    >> setup_synthesize(args)
"""
from datetime import datetime
import argparse
import logging
from pathlib import Path

import torch.utils.data
import torch

from src.dataprocessing import transforms as T
from src.dataprocessing.audio_normalizing import preprocessing, postprocessing
# this module implements the normal MelNet generation procedure
from src.model.MelNet import MelNet
# this module implements the MelNet generation procedure where the output of the first tier is
# a low-resolution spectrogram from the dataset (to isolate and test the effect of Upsampling Layers
# only). It should be used only in that case.
# from src.model.MelNetUpsampling import MelNet
from src.utils.hparams import HParams
from src.utils.logging import TensorboardWriter


def synthesize(args: argparse.Namespace, hp: HParams, synthesisp: HParams,
               extension_architecture: str, timestamp: str, tensorboardwriter: TensorboardWriter,
               logger: logging.Logger) -> None:
    """
    Synthesizes one or several samples.

    Args:
        args (argparse.Namespace): parameters to set up the training. At least, args must contain:
                                   args = {"path_config": ...,
                                           "tier": ...,
                                           "checkpoint_path": ...}
        hp (HParams): hyperparameters for the model and other parameters (training, dataset, ...)
        synthesisp (HParams): parameters for doing synthesis. It contains the path to the weights of
                              the trained tiers.
        extension_architecture (str): information about the network's architecture of this run
                                      (synthesis) to identify the logs and output of the model.
        timestamp (str): information that identifies completely this run (synthesis).
        tensorboardwriter (TensorboardWriter): to log information about training to tensorboard.
        logger (logging.Logger): to log general information about the training of the model.
    """
    # Setup model
    melnet = MelNet(n_tiers=hp.network.n_tiers,
                    layers=hp.network.layers,
                    hidden_size=hp.network.hidden_size,
                    gmm_size=hp.network.gmm_size,
                    freq=hp.audio.mel_channels)
    melnet = melnet.to(hp.device)
    # Load weights from previously trained tiers
    melnet.load_tiers(synthesisp.checkpoints_path, logger)
    melnet.eval()

    # Perform inference
    logger.info("Starting synthesis")
    with torch.no_grad():
        spectrogram = melnet.sample(hp=hp,
                                    synthesisp=synthesisp,
                                    timestamp=timestamp,
                                    logger=logger,
                                    n_samples=1,
                                    length=args.timesteps)
    logger.info("Synthesis finished")

    # Compute spectrogram
    spectrogram = postprocessing(spectrogram, hp)
    logger.info("Spectrogram post processed")
    T.save_spectrogram(synthesisp.output_path + "/" + timestamp, spectrogram, hp)
    logger.info("Spectrogram saved as image")
    torch.save(spectrogram, synthesisp.output_path + "/" + timestamp + ".pt")
    logger.info("Spectrogram saved as tensor")
    tensorboardwriter.log_synthesis(spectrogram)
    logger.info("Spectrogram saved in Tensorboard")


def setup_synthesize(args: argparse.Namespace):
    """
    Sets up synthesis with the parameters specified in args and the path to the weights of the model

    Args:
        args (argparse.Namespace): parameters to set up the synthesis.
    """
    # 1. Read hyperparameters from file
    hp = HParams.from_yaml(args.path_config)
    synthesisp = HParams.from_yaml(args.path_synthesis)
    # check if GPU available and add it to parameters
    hp["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 2. Create extension of the architecture of the model and timestamp for this run (use to
    # identify folders and files created for this run)
    # format: f(params_file)_t(n_tiers)_l(n_layers)_hd(hidden_size)_gmm(gmm_size).
    extension_architecture = f"d{hp.name}_t{hp.network.n_tiers}_" \
                             f"l{'.'.join(map(str, hp.network.layers))}_" \
                             f"hd{hp.network.hidden_size}_gmm{hp.network.gmm_size}"
    timestamp = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 3 Create directories for saving logs and output if they do not exist
    # 3.1 Create general log directory for this run (the same directory will be used for different
    #     runs of a model with same architecture and the difference will be in the file stored)
    hp["logging"]["dir_log"] = hp.logging.dir_log + extension_architecture
    Path(hp.logging.dir_log).mkdir(parents=True, exist_ok=True)
    # 3.2 Create directory for the outputs of this run (the same directory will be used for
    #     different runs of a model with same architecture and the difference will be in the weights
    #     of the model)
    synthesisp.output_path = synthesisp.output_path + extension_architecture
    Path(synthesisp.output_path).mkdir(parents=True, exist_ok=True)

    # 4. Setup general logging (it will use the folder previously created and the filename will be:
    filename = f"{hp.logging.dir_log}/synthesis_{timestamp}"
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

    # 6. Setup tensorboard logging
    # 6.1 Create tensorboard logs directory (tensorboard requires a different folder for
    # each run of the model, in this case every run to train a tier) so we add the extension
    # of the network's architecture of this run and the timestamp to identify it completely
    tensorboard_dir = hp.logging.dir_log_tensorboard + extension_architecture \
                      + f"synthesis_{timestamp}"
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    # 2.2 Create tensorboard writer
    tensorboardwriter = TensorboardWriter(hp, tensorboard_dir)

    synthesize(args, hp, synthesisp, extension_architecture, timestamp, tensorboardwriter, logger)

    tensorboardwriter.close()
