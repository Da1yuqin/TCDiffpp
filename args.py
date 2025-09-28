import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()

    ### Project ###
    parser.add_argument("--project", default="./runs/train", help="./project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    
    ### dataset ###
    parser.add_argument("--data_path", type=str, 
            default="./data/AIOZ_Dataset/", 
            help="Path to raw dataset folder (must contain train/test subfolders). Do not modify unless necessary.")
    parser.add_argument("--processed_data_dir",type=str, 
                        default="./data/dataset_backups/", help="Dataset backup path") 
    parser.add_argument("--batch_size", type=int, default=37) 
    parser.add_argument("--window_size", type=int, default=150, help="window size")
    parser.add_argument( 
        "--force_reload", default = False, action="store_true", help="Force reprocessing of the dataset, ignoring cached data."
    )

    parser.add_argument("--no_cache", action="store_true", default = False, help="Disable dataset caching; always reload from disk.")
        
    parser.add_argument(
        "--required_dancer_num", type = int, default=4, help="Number of dancers required in each sample."
    )


    ### Out Result ###
    parser.add_argument(
        "--vis_fk_out", type=str, default="./fk_out4Vis", help="Path to save FK outputs for visualization."
    )
    parser.add_argument(
        "--render_dir", type=str, default="./renders/", help="Path to save rendered sample videos."
    )
    parser.add_argument(
        "--wandb_pj_name", type=str, default="TCDiffpp", help="Project name for Weights & Biases tracking."
    )

    ### Training ###
    parser.add_argument( "--learning-rate", type=float, default = 0.00005, help="learning rate") 
    parser.add_argument("--epochs", type=int, default=10000)  # Until Coveraged
    parser.add_argument("--use_ssm", type=bool, default=True)
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

    opt = parser.parse_args()
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required_dancer_num", type=int, default=4, help="traj stride")

    ### DanceDecoder model ###
    parser.add_argument(
        "--dance_checkpoint", type=str, 
        # default="./runs/train/exp/weights/train-8000.pt",
        default="", 
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
        "--render_dir", type=str, default="./renders/test/LongDance/", help="Sample render path"
    )
    parser.add_argument(
        "--save_motions", action="store_true", default= True, help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default = "./fk_out",
        help="Where to save the motions",
    )

    ### Slice Data ###
    parser.add_argument(
        "--cache_features",
        default = True, 
        action="store_true",
        help="Save the extracted features for later reuse",
    )
    parser.add_argument(
        "--use_cached_features",
        default = True, 
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="./data/Processed_data/cache_npy_wav/cache/all/", 
        help="Where to save/load the features",
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="./data/AIOZ-dataset/musics", 
        help="folder containing input music",
    )
    parser.add_argument(
        "--music_process_dir",
        type=str,
        default="./data/Processed_data/cache_npy_wav",      
        help="folder containing processed input music",
    )
    opt = parser.parse_args()
    return opt