import argparse
import os
from pathlib import Path
from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from filter_split_data import *
from slice import *
import numpy as np
from data_preprocess.dataset_utils import processing_music_list


def create_dataset(opt):
    # step1: split the data according to the splits files accroding to dataset .txt files
    print("Creating train / test split")
    split_data(opt.dataset_folder) 

    # step2: slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_AIOZ(opt.root_path + f"train/motions", opt.root_path + f"train/wavs", stride = opt.stride, length = opt.length)
    print("Slicing test data")
    slice_AIOZ(opt.root_path + f"test/motions", opt.root_path + f"test/wavs", stride = opt.stride, length = opt.length)

    # step3:process dataset to extract audio features
    print("Extracting features")
    processing_music_list(opt.root_path + "train/wavs_sliced",opt.root_path,"train")
    processing_music_list(opt.root_path + "test/wavs_sliced",opt.root_path,"test")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default = "./AIOZ_Dataset",
        help="folder containing motions and music",
    )
    parser.add_argument("--root-path", type=str,default='./AIOZ_Dataset',help='root path to create dataset') 
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
