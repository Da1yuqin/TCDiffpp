import glob
import os
import pickle
import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def slice_AIOZmotion(motion_file, stride, length, num_slices, out_dir):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"] # (dancer_num, seq_len, smpl_pos)
    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    # pos /= scale
    start_idx = 0
    window = int(length * 30) 
    stride_step = int(stride * 30)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= pos.shape[1] - window and slice_count < num_slices:
        pos_slice, q_slice = (
            pos[:,start_idx : start_idx + window,:], 
            q[:,start_idx : start_idx + window,:],
        )
        out = {"pos": pos_slice, "q": q_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count

def slice_AIOZ(motion_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))

        if os.path.exists(f"{wav_out}/{m_name}_slice0.wav"): 
            continue
        audio_slices = slice_audio(wav, stride, length, wav_out)

        if os.path.exists(f"{wav_out}/{m_name}_slice0.pkl"): 
            continue
        motion_slices = slice_AIOZmotion(motion, stride, length, audio_slices, motion_out) 


