import time

import os

# import shutil
import torch

import json
import glob
import argparse

############
# need the following to avoid the following error:
#  TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481
from tqdm import tqdm

tqdm.monitor_interval = 0
############

from net.dataset.reading_image_provider import ReadingImageProvider
from net.dataset.raw_image import RawImageType
from net.pytorch_utils.concrete_eval import FullImageEvaluator
from configs.config import Config


###############################################################################
class RawImageTypePad(RawImageType):
    global config

    def finalyze(self, data):
        # border reflection of 22 yields a field size of 1344 for 1300 pix inputs
        return self.reflect_border(data, config.padding)  # 22)


###############################################################################
def eval_cresi(
    config,
    paths,
    fn_mapping,
    image_suffix,
    save_dir,
    num_channels=3,
    weight_dir="",
    nfolds=4,
):

    # no grad needed for test, and uses less memory?
    with torch.no_grad():
        # if 2 > 1:
        # t0 = time.time()
        ds = ReadingImageProvider(
            RawImageTypePad,
            paths,
            fn_mapping,
            image_suffix=image_suffix,
            num_channels=num_channels,
        )

        folds = [([], list(range(len(ds)))) for i in range(nfolds)]
        if torch.cuda.is_available():
            num_workers = 0 if os.name == "nt" else 2
        else:
            # get connection error if more than 0 workers and cpu:
            #   https://discuss.pytorch.org/t/data-loader-crashes-during-training-something-to-do-with-multiprocessing-in-docker/4379/5
            num_workers = 0

        print("num_workers:", num_workers)
        keval = FullImageEvaluator(
            config,
            ds,
            save_dir=save_dir,
            flips=3,
            num_workers=num_workers,
            border=config.padding,
        )
        for fold, (t, e) in enumerate(folds):
            print("fold:", fold)
            if args.fold is not None and int(args.fold) != fold:
                print("ummmm....")
                continue
            keval.predict(fold, e, weight_dir, verbose=False)

    return folds


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

    # check image files
    exts = (
        "*.tif",
        "*.tiff",
        "*.jpg",
        "*.JPEG",
        "*.JPG",
        "*.png",
    )  # the tuple of file types
    files_grabbed = []
    for ext in exts:
        files_grabbed.extend(glob.glob(os.path.join(config.sliced_dir, ext)))

    if len(files_grabbed) == 0:
        print("02_eval.py: No valid image files to process, returning...")

        exit(1)

    paths = {"masks": "", "images": config.sliced_dir}

    # set weights_dir
    weight_dir = config.path_weights

    # make sure output folders exist
    save_dir = os.path.join(config.results_dir, config.folds_save_dir)

    print("save_dir:", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    fn_mapping = {"masks": lambda name: os.path.splitext(name)[0] + ".tif"}  #'.png'
    image_suffix = ""  #'img'

    # set folds
    skip_folds = []
    if args.fold is not None:
        skip_folds = [i for i in range(4) if i != int(args.fold)]

    print("paths:", paths)
    print("fn_mapping:", fn_mapping)
    print("image_suffix:", image_suffix)
    ###################

    # execute
    t0 = time.time()

    folds = eval_cresi(
        config,
        paths,
        fn_mapping,
        image_suffix,
        save_dir,
        weight_dir=weight_dir,
        num_channels=config.num_channels,
        nfolds=config.num_folds,
    )
    t1 = time.time()

    print(
        "Time to run",
        len(folds),
        "folds for",
        len(os.listdir(config.sliced_dir)),
        "=",
        t1 - t0,
        "seconds",
    )

