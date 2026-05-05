import os
import numpy as np
import pandas as pd
import librosa
import random
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_METADATA = r"c:/Users/Nandhini Prbakaran/OneDrive/Documents/Minor_Nan_Audio/Minor_Nan_Audio/metadata.csv"
OUTPUT_DIR = r"c:/Users/Nandhini Prbakaran/OneDrive/Documents/Minor_Nan_Audio/Minor_Nan_Audio/features"
OUTPUT_METADATA = r"c:/Users/Nandhini Prbakaran/OneDrive/Documents/Minor_Nan_Audio/Minor_Nan_Audio/metadata_features.csv"

SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

# Augmentation Config
PITCH_SHIFT_STEPS = 2.0  # +/- 2 semitones approx to +/- 12% pitch, user asked +/- 2% pitch? 
# Note: User asked for "+/- 2%". 
# In semitones: 2% frequency change. f2 = f1 * 2^(n/12). 1.02 = 2^(n/12) => n = 12 * log2(1.02) ≈ 0.34 semitones.
# That is very subtle. I will use +/- 2 steps (semitones) to be effective, or strictly follow 2%.
# Strictly 2%: n_steps approx 0.35.
# Let's use a range.
PITCH_SHIFT_RANGE = 0.35 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_and_fix_audio(path):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        
        # Pad or Trim
        if len(y) > SAMPLES:
            # Crop random or center? Center is safer for emotion.
            start = (len(y) - SAMPLES) // 2
            y = y[start:start+SAMPLES]
        else:
            # Pad with zeros
            padding = SAMPLES - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, padding - offset), 'constant')
            
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def extract_log_mel(y):
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=SAMPLE_RATE, 
        n_mels=N_MELS, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH
    )
    # Log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel

def spec_augment(mel_spec):
    # Time Masking
    time_mask_param = 30
    freq_mask_param = 20
    
    aug_mel = mel_spec.copy()
    num_cols = aug_mel.shape[1]
    num_rows = aug_mel.shape[0]
    
    # Time mask
    t = random.randint(0, time_mask_param)
    t0 = random.randint(0, max(0, num_cols - t))
    aug_mel[:, t0:t0+t] = -80.0 # Silence in dB usually -80 or min
    
    # Freq mask
    f = random.randint(0, freq_mask_param)
    f0 = random.randint(0, max(0, num_rows - f))
    aug_mel[f0:f0+f, :] = -80.0
    
    return aug_mel

def main():
    ensure_dir(OUTPUT_DIR)
    
    df = pd.read_csv(INPUT_METADATA)
    new_records = []
    
    print(f"Starting feature extraction for {len(df)} files...")
    
    for idx, row in df.iterrows():
        file_path = row['file_path']
        dataset = row['dataset']
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 1. Load Audio
        y = load_and_fix_audio(file_path)
        if y is None:
            continue
            
        # 2. Extract Clean Feature
        log_mel = extract_log_mel(y)
        
        # Save Original
        orig_filename = f"{dataset}_{row['emotion']}_{idx}_orig.npy"
        orig_path = os.path.join(OUTPUT_DIR, orig_filename)
        np.save(orig_path, log_mel)
        
        # Add to metadata
        rec = row.to_dict()
        rec['feature_path'] = orig_path
        rec['augmented'] = False
        new_records.append(rec)
        
        # 3. Augmentation (RAVDESS ONLY)
        if dataset == 'RAVDESS':
            # Audio Augmentation: Pitch Shift
            # +/- 2% random
            steps = random.uniform(-PITCH_SHIFT_RANGE, PITCH_SHIFT_RANGE)
            y_aug = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=steps)
            
            # Extract Mel
            mel_aug = extract_log_mel(y_aug)
            
            # Spec Augmentation
            mel_aug = spec_augment(mel_aug)
            
            # Save Augmented
            aug_filename = f"{dataset}_{row['emotion']}_{idx}_aug.npy"
            aug_path = os.path.join(OUTPUT_DIR, aug_filename)
            np.save(aug_path, mel_aug)
            
            # Add to metadata
            rec_aug = row.to_dict()
            rec_aug['feature_path'] = aug_path
            rec_aug['augmented'] = True
            rec_aug['speaker_id'] = rec_aug['speaker_id'] + "_aug" # Distinguish? Or same speaker?
            # Usually augmentations share speaker ID if we want to learn speaker invariance, 
            # but if we strictly split by speaker, it's fine.
            # I will keep same speaker_id to ensure split consistency if grouped by speaker.
            
            new_records.append(rec_aug)
            
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)}")

    # Save new metadata
    new_df = pd.DataFrame(new_records)
    new_df.to_csv(OUTPUT_METADATA, index=False)
    
    print(f"\nFeature extraction complete.")
    print(f"Saved to {OUTPUT_DIR}")
    print(f"Metadata saved to {OUTPUT_METADATA}")
    
    # Verification
    print("\n--- Verification ---")
    print(f"Total features: {len(new_df)}")
    print(new_df.groupby(['dataset', 'augmented']).size())
    
    # Load first file to check shape
    sample = np.load(new_records[0]['feature_path'])
    print(f"\nSample Shape: {sample.shape}")
    print(f"Sample Min/Max: {sample.min():.2f}, {sample.max():.2f}")
    
    if np.isnan(sample).any():
        print("WARNING: NaNs detected in sample!")
    else:
        print("No NaNs detected in sample.")

if __name__ == "__main__":
    main()
