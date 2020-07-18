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
from argparse import Namespace
import logging
from pathlib import Path

import torch.utils.data
import torch

from src.dataprocessing import transforms as T
from src.dataprocessing.audio_normalizing import preprocessing, postprocessing
from src.model.MelNet import MelNet
from src.utils.hparams import HParams
from src.utils.logging import TensorboardWriter


def synthesize(hp: HParams, args: Namespace, extension_run: str,
               tensorboardwriter: TensorboardWriter, logger: logging.Logger):
    """
    Synthesizes one or several samples.

    Args:
        hp (HParams): hyperparameters for the model and other parameters (audio, network
                      architecture, ...)
        args (argparse.Namespace): input parameters (path to HParams file, ...)
        extension_run (str): information about the run (synthesis) to identify the logs
                             and the output
        tensorboardwriter (TensorboardWriter): interface to save logs to tensorboard
        logger (logging.Logger): log general information about the synthesis

    """
    # Setup model
    melnet = MelNet(n_tiers=hp.network.n_tiers,
                    layers=hp.network.layers,
                    hidden_size=hp.network.hidden_size,
                    gmm_size=hp.network.gmm_size,
                    freq=hp.audio.mel_channels)
    melnet = melnet.to(hp.training.device)

    # Load weights from previously trained model
    if not Path(args.checkpoint_path).exists():
        logger.error(f"Path for model weigths {args.checkpoint_path} does not exist.")
        raise Exception(f"Path for model weigths {args.checkpoint_path} does not exist.")

    logger.info(f"Synthesis with weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path)
    hp_chkpt = checkpoint["hp"]

    if hp_chkpt.audio != hp.audio:
        logger.warning("New params for audio are different from checkpoint. "
                       "It will use new params.")

    if hp_chkpt.network != hp.network:
        logger.error("New params for network structure are different from checkpoint.")
        raise Exception("New params for network structure are different from checkpoint.")

    melnet.load_state_dict(checkpoint["model"])

    # Perform inference
    logger.info("Starting synthesis")
    spectrogram = melnet.sample(hp=hp, logger=logger, n_samples=1, length=args.timesteps)
    logger.info("Synthesis finished")

    # Compute spectrogram
    spectrogram = postprocessing(spectrogram, hp)
    logger.info("Spectrogram post processed")
    T.save_spectrogram(args.output_path+extension_run, spectrogram, hp)
    logger.info("Spectrogram saved as image")
    torch.save(spectrogram, args.output_path+extension_run+".pt")
    logger.info("Spectrogram saves as tensor")
    tensorboardwriter.log_synthesis(spectrogram)
    logger.info("Spectrogram saved in Tensorboard")


def setup_synthesize(args: Namespace):
    """
    Sets up synthesis with the parameters specified in args and the path to the weights of the model

    Args:
        args (argparse.Namespace): parameters to set up the synthesis.
    """
    # read parameters from file
    hp = HParams.from_yaml(args.path_config)
    # check if GPU available and add it to parameters
    hp["training"]["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create extension for this run
    # format: f(params_file)_t(n_tiers)_l(n_layers)_hd(hidden_size)_gmm(gmm_size)-time.
    extension_run = f"d{hp.name}_t{hp.network.n_tiers}_" \
                    f"l{'.'.join(map(str, hp.network.layers))}_" \
                    f"hd{hp.network.hidden_size}_gmm{hp.network.gmm_size}_" \
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # create directories for logging if they do not exist
    # create general log directory
    Path(hp.logging.dir_log).mkdir(parents=True, exist_ok=True)
    # create tensorboard logs directory
    # tensorboard requires a different folder for each run of the model so we add the extension for
    # this run
    hp["logging"]["dir_log_tensorboard"] = hp.logging.dir_log_tensorboard + extension_run
    Path(hp.logging.dir_log_tensorboard).mkdir(parents=True, exist_ok=True)
    # create directory for output
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # setup general logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=hp.logging.dir_log + extension_run + ".log"),
            # handler to save the log to a file
            logging.StreamHandler()  # handler to output the log to the terminal
        ])
    logger = logging.getLogger()
    logger.info(f"Device for synthesis: {hp.training.device}")

    # setup tensorboard logging
    tensorboardwriter = TensorboardWriter(hp)

    synthesize(hp, args, extension_run, tensorboardwriter, logger)

    tensorboardwriter.close()
