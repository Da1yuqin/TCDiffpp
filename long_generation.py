import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from TCDiffpp import TCDiffpp

# music feat 438 extractor
from data.data_preprocess.dataset_utils import processing_music_list
import codecs as cs

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


def test(opt):
    render_len = 512

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    split_filenames = [] 
    genre_split_filenames = []

    # # split file by dancer number
    # split_file_pth = f"./data/dancernum_split/split_files/split_dancerNum_{opt.required_dancer_num}.txt"
    # with cs.open(split_file_pth, 'r') as f: 
    #     for line in f.readlines():
    #         split_filenames.append(line.strip())
    
    # Genre type
    if opt.genre is not None:
        genre_split_file_pth = f"./data/sort_by_music_genre/train/{opt.genre}_split_sequence_names.txt"
        with cs.open(genre_split_file_pth, 'r') as f: 
            for line in f.readlines():
                genre_split_filenames.append(line.strip())


    print("Computing features for input music")
    for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
        songname = os.path.splitext(os.path.basename(wav_file))[0]

        # if opt.genre is not None and songname not in genre_split_filenames: 
        if opt.genre is not None and songname not in genre_split_filenames: 
            continue

        # create temp folder (or use the cache folder if specified)
        if opt.cache_features: 
            save_dir = os.path.join(opt.feature_cache_dir, songname) 
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            dirname = save_dir
        else: 
            temp_dir = TemporaryDirectory()
            temp_dir_list.append(temp_dir)
            dirname = temp_dir.name
        # slice the audio file
        print(f"Slicing {wav_file}")
        slice_audio(wav_file, 2.5, 5.0, dirname) 
        file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey) 
        
        print(f"Computing features for {wav_file}")
        cond_list = processing_music_list(dirname, opt.music_process_dir, "test", is_test=True) 

        # one music's sliced music feature
        all_cond.append(cond_list) # all music feature
        all_filenames.append(file_list)

    model = TCDiffpp(opt.dance_checkpoint, required_dancer_num = opt.required_dancer_num, use_ssm = opt.use_ssm)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i] 
        if len(all_cond[i].shape) == 3:
            model.render_sample(
                data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render, render_len = render_len
            )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
