import argparse
from pathlib import Path
import os
import sys

sys.path.insert(0, os.getcwd())

from src.utils.training import setup_training

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path-config", type=str, required=True,
                        help="Path to yaml configuration file or folder with files. If path to "
                             "configuration is a folder with files, it trains a model for every " 
                             "configuration file.")
    parser.add_argument("--checkpoint-path", type=str, required=False, default=None,
                        help="Path to model weights to resume training. Only if path-config is a "
                             "file and not a folder")
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
