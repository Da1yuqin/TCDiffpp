import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import numpy as np
import torch
from tqdm import tqdm
import pickle
 
motion_dir = "./AIOZ-dataset/motions_smpl" 
save_path = "./split_files"
os.makedir(save_path, exist=True)
required_dancer_num = 2 

split_list = []
for motion_file in tqdm(glob.glob(os.path.join(motion_dir, "*.pkl")),desc="Processing..."):
    name = os.path.splitext(os.path.basename(motion_file))[0]
    motion = pickle.load(open(motion_file, "rb"))
    motion = motion["root_trans"] 
    if motion.shape[0] == required_dancer_num: 
        split_list.append(name)

f=open(os.path.join(save_path, f"split_dancerNum_{required_dancer_num}.txt"),"w")
for line in split_list:
    f.write(line+'\n')
f.close()
print(f"saved at {save_path} with dancer num {required_dancer_num}")
    
