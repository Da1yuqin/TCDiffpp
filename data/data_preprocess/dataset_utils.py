import os
import librosa
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from data.data_preprocess._preprocess_wav import FeatureExtractor

# Initialize feature extractor
extractor = FeatureExtractor()

def processing_music_list(music_data_path, root_path, folder_state, is_test=False):
    """Processes a list of music files, extracts features, and saves them as .npy files."""
    feats_path = os.path.join(root_path, folder_state, "feats438")
    Path(feats_path).mkdir(parents=True, exist_ok=True)  # Create directory if not exists

    music_names = {file.split('.')[0] for file in os.listdir(music_data_path)}
    cond_list = []

    for name in tqdm(music_names):
        feat_npy_path = os.path.join(feats_path, f"{name}.npy")
        music_npy_path = os.path.join(music_data_path, f"{name}.npy")
        music_wav_path = os.path.join(music_data_path, f"{name}.wav")

        if os.path.exists(music_npy_path):
            music = np.load(music_npy_path)
            if not is_test:
                np.save(feat_npy_path, music)
            else:
                cond_list.append(music)
        elif os.path.exists(feat_npy_path):
            # Skip if already generated
            continue
        elif os.path.exists(music_wav_path):
            music = wav_processing(music_wav_path, name)
            if not is_test:
                np.save(feat_npy_path, music)
            else:
                np.save(music_npy_path, music)
                cond_list.append(music.detach().numpy())
    
    if is_test:
        return torch.from_numpy(np.asarray(cond_list))

def wav_processing(wav_path, audio_name):
    """Extracts features from a .wav file and saves them as .npy."""
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    
    def _get_tempo(audio_name):
        """Extracts tempo (BPM) from the filename."""
        assert len(audio_name) == 4
        if audio_name[:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return int(audio_name[3]) * 10 + 80
        elif audio_name[:3] == 'mHO':
            return int(audio_name[3]) * 5 + 110
        else:
            raise ValueError(f"Invalid audio name: {audio_name}")

    # Load audio
    audio, _ = librosa.load(wav_path, sr=SR)
    
    # Extract features
    melspe_db = extractor.get_melspectrogram(audio, SR)
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, SR, octave=7 if SR == 15360 * 2 else 5)
    onset_env = extractor.get_onset_strength(audio_percussive, SR).reshape(1, -1)
    tempogram = extractor.get_tempogram(onset_env, SR)
    onset_beat = extractor.get_onset_beat(onset_env, SR)[0]

    # Combine extracted features
    feature = np.concatenate([
        mfcc,  # 20
        mfcc_delta,  # 20
        chroma_cqt,  # 12
        onset_env,  # 1
        onset_beat,  # 1
        tempogram
    ], axis=0)

    feature = feature.T  # Transpose for proper shape
    np.save(wav_path.replace('.wav', '.npy'), feature)  # Save as .npy
    return torch.tensor(feature, dtype=torch.float32)
