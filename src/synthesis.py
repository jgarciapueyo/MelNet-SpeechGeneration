"""
Main entry point to synthesize a melspectrogram

Usage:
    python synthesis.py [-h] -p PATH_CONFIG -s PATH_SYNTHESIS [-t TIMESTEPS]

This module generates spectrograms.
See README.md and models/params/README.md for more information about the configurations file.
PATH_CONFIG must be a single file defining a single model and PATH_SYNTHESIS must be a single file
defining the path to the weights of the model being used do synthesis. Both files must correspond to
a model with the same architecture (number of tiers, number of layers in each tier, hidden_size,
...).

TIMESTEPS is the number of frames of the spectrogram to generate. It can also be seen as the length
of the spectrogram.
"""

import argparse
from pathlib import Path
import os
import sys

sys.path.insert(0, os.getcwd())

from src.utils.synthesize import setup_synthesize

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path-config", type=str, required=True,
                        help="Path to yaml configuration file. It must be a file and not a "
                             "directory")
    parser.add_argument("-s", "--path-synthesis", type=str, required=True,
                        help="Path to yaml synthesis file. It must be a file and not a directory. "
                             "The filemodel must contain the path to the weights of the tiers")
    parser.add_argument("-t", "--timesteps", type=int, required=False, default=300,
                        help="Number of frames of spectrogram to synthesize")
    args = parser.parse_args()

    if Path(args.path_config).is_dir():
        raise Exception("Path to yaml configuration can not be a directory")
    else:
        setup_synthesize(args)
