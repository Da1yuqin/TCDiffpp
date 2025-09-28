import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()

    ### Project ###
    parser.add_argument("--project", default="./runs/train", help="./project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    
    ### dataset ###
    parser.add_argument("--data_path", type=str, default="/opt/data/private/project3-Dancepartner/0-dataset/dataProcess/", help="raw data path") # 含 train/test 文件夹的 dir, 不要改
    parser.add_argument("--processed_data_dir",type=str, 
                        default="/opt/data/private/IJCV/opensource/EDGE/data/dataset_backups/", help="Dataset backup path") # full 3-dancer cache 路径
    parser.add_argument("--batch_size", type=int, default=37, help="batch size") # layer-8, 4-gpu | 注意 bs 要大于20, 因为 test//10, 然后 rander_cont == 2
    # parser.add_argument("--batch_size", type=int, default=65, help="batch size") # layer-8, 6-gpu
    parser.add_argument("--window_size", type=int, default=150, help="window size")
        # change dancer_num
    parser.add_argument(
        "--required_dancer_num", type = int, default=4, help="don't reuse / cache loaded dataset"
    )
    parser.add_argument( #dancer_num
        "--split_file", type = str, 
        default="/opt/data/private/IJCV/opensource/TCDiffpp_debug/data/dancernum_split/split_files/split_dancerNum_4.txt", help="don't reuse / cache loaded dataset"
    )


    parser.add_argument( # 强制 reload, 不适用 backup, False | 4 dump 了也是寄, 但是处理的很快
        "--force_reload", default = False, action="store_true", help="force reloads the datasets" 
    )
    parser.add_argument("--no_cache", action="store_true", default = False, help="don't reuse / cache loaded dataset")


    ### Out Result ###
    parser.add_argument(
        "--render_dir", type=str, default="./renders/", help="Sample render path" # 
    )
    parser.add_argument(
        "--wandb_pj_name", type=str, default="DanceDecoder-Modulation4", help="project name"
    )
    parser.add_argument(
        "--vis_fk_out", type=str, default="./fk_out4Vis", help="project name"
    )

    ### Training ###
    parser.add_argument( "--learning-rate", type=float, default = 0.0004, help="learning rate") # 52 bs-4gpu | 这个数量级大概一天能初步收敛
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--use_ssm", type=bool, default=False)
    # parser.add_argument("--finetuning_epochs", type=int, default=2000) # Until Coveraged
    parser.add_argument(
        "--save_interval",
        type=int,
        default=200, 
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, 
        default= "",
        help="trained checkpoint path (optional)"
    )

    parser.add_argument("--mode", default = "train", choices=["train", "val"])

    opt = parser.parse_args()
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required_dancer_num", type=int, default=4, help="traj stride")
    parser.add_argument(
        "--genre", 
        type=str, 
        default="Reggae", 
        choices=["Indian", "Electronic", "Pop", "Rap", "RB", "Reggae", "Latin"],
        help="Select the music genre."
    )

    ### DanceDecoder model ###
    parser.add_argument(
        "--dance_checkpoint", type=str, 
        default="/opt/data/private/IJCV/AblationStudy/TCDiffpp_fpdpesmfa/runs/train/exp4/weights/train-7000.pt",
        # default="", 
        help="checkpoint"
    )
    parser.add_argument(
        "--no_render",
        default = False,
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument("--use_ssm", type=bool, default=False)

    ### Out ###
    parser.add_argument(
        "--render_dir", type=str, default="./renders/test/Dance/", help="Sample render path"
    )
    parser.add_argument("--render_traj_dir", type=str, default="./renders/test/traj_renders/", help="traj render dir")
    parser.add_argument(
        "--save_motions", action="store_true", default= True, help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        # default="./eval/motions",
        default = "./fk_out",
        help="Where to save the motions",
    )

    ### Slice Data ###
    parser.add_argument(
        "--cache_features",
        default = True, # 保存 test 所需的数据
        action="store_true",
        help="Save the extracted features for later reuse",
    )
    parser.add_argument(
        "--use_cached_features",
        default = True, # 还没处理的时候用 False, 只用用 True
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="/opt/data/private/project3-Dancepartner/0-dataset/Processed_data/cache_npy_wav/cache/all/", # all
        # default = "/opt/data/private/project3-Dancepartner/temp_dir/cache_npy_wav/cache/train/", # setting:step(2/4) | 每次运行, 把成对的 wav 和 test 装到这里
        help="Where to save/load the features",
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        # default="/opt/data/private/dataset/AIOZ-dataset/musics", # all
        # default="/opt/data/private/IJCV/opensource/_wildmusic/indian", # wild 
        default = "/opt/data/private/project3-Dancepartner/temp_dir/cache_npy_wav/wav_debug/train", # # debug | setting:step(1/4) | 用于测试test的, 输入的未裁剪原始音频 .wav,
        help="folder containing input music",
    )
    parser.add_argument(
        "--music_process_dir",
        type=str,
        default="/opt/data/private/project3-Dancepartner/0-dataset/Processed_data/cache_npy_wav",      
        help="folder containing input music",
    )
    parser.add_argument(
        "--seed_motion_pth",
        type=str,
        default="/opt/data/private/dataset/AIOZ-dataset/motions_smpl/",      
        help="folder containing input music",
    )
    opt = parser.parse_args()
    return opt