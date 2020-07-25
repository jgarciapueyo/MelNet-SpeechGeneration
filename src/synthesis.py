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
    # TODO: add output_path to synthesis_dataset_vX.yml instead of a command line option
    parser.add_argument("-o", "--output-path", type=str, required=False, default="out/")
    args = parser.parse_args()

    if Path(args.path_config).is_dir():
        raise Exception("Path to yaml configuration can not be a directory")
    else:
        setup_synthesize(args)
