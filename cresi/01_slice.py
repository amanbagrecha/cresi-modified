import os
import json
import argparse

from utils.utils import update_config
from configs.config import Config


###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()

    # get config
    with open(args.config_path, "r") as f:
        cfg = json.load(f)
    config = Config(**cfg)
    
    # set images folder (depending on if we are slicing or not)
    if (len(config.eight_bit_dir) > 0) and (config.slice_x > 0):
        print("Executing tile_im.py..")

        cmd = "python " + config.path_src + "/data_prep/tile_im.py " + args.config_path
        print("slice command:", cmd)
        os.system(cmd)

    else:
        print("Invalid configuration or no files in sliced_dir")
