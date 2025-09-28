from args import parse_train_opt
from TCDiffpp import TCDiffpp
import os
import codecs as cs
import warnings
warnings.filterwarnings('ignore')


def train(opt):
    # split file 
    split_file_pth = opt.split_file
    split_filenames = []
    with cs.open(split_file_pth, 'r') as f: 
        for line in f.readlines():
            split_filenames.append(line.strip())

    model = TCDiffpp(checkpoint_path = opt.checkpoint, learning_rate=opt.learning_rate, \
        window_size=opt.window_size, required_dancer_num = opt.required_dancer_num, split_file = split_filenames, use_ssm=opt.use_ssm)

    if opt.mode == "train":
        model.train_loop(opt)
    elif opt.mode == "val":
        model.val_loop(opt)
    else:
        raise ValueError(f"Invalid mode: {opt.mode}. Must be one of ['train', 'val'].")


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
