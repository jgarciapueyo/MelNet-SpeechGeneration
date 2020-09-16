"""
Main entry point to train a model

Usage:
   python train.py [-h] -p PATH_CONFIG [--tier TIER] [--checkpoint-path CHECKPOINT_PATH]

This module trains a model or several models, and every model is composed of tiers.
See README.md and models/params/README.md for more information about the configuration file.
If PATH_CONFIG is a single file, it defines a single model and a single model will be trained.
If PATH_CONFIG is a folder, it can contain several files and various models will be trained (one for
    every file in the folder PATH_CONFIG).

If you want to train only one tier of the model, the flag --tier can be used.
If PATH_CONFIG is a single file, then the specified tier of that model will be trained.
If PATH_CONFIG is a folder, then the specified tier for various models will be trained.

Only the training of a single tier of a single model can be RESUMED. This means that if
CHECKPOINT_PATH is a file (weights of a tier), then PATH_CONFIG must point to a single file
(the parameters of the weight file and the configuration file must be the same) and TIER must define
the tier for which training is going to be resumed.
"""

import argparse
from pathlib import Path
import os
import sys

sys.path.insert(0, os.getcwd())

# this module implements the basic training. (The training_batch module should be favored over this
# one)
# from src.utils.training import setup_training
# this module implements the training using gradient accumulation to train in bigger batches
from src.utils.training_batch import setup_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path-config", type=str, required=True,
                        help="Path to yaml configuration file or folder with files. If path to "
                             "configuration is a folder with files, it trains a model for every "
                             "configuration file.")
    parser.add_argument("--tier", type=int, required=False, default=None,
                        help="Training has to be done in different tiers. If this argument is "
                             "defined only this tier of the model will be trained. If the argument "
                             "is not defined, all the tiers of the model will be trained "
                             "consecutively.")
    parser.add_argument("--checkpoint-path", type=str, required=False, default=None,
                        help="Path to model weights to resume training. Only if path-config is a "
                             "file and not a folder. Only the training of a single tier can be "
                             "resumed.")
    args = parser.parse_args()

    if Path(args.path_config).is_dir():
        if args.checkpoint_path is not None:
            raise Exception("Can not use a path to a folder with configuration files and declare a"
                            " path to model weights checkpoint.")
        else:
            # if path to configuration is a folder with files, it trains a model for every
            # configuration file
            all_files_recursively = sorted(Path(args.path_config).glob('**/*.yml'))
            for file in all_files_recursively:
                args.path_config = file
                setup_training(args)
    else:
        # if path to configuration is a file, train the model
        setup_training(args)
