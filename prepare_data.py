import os
import glob
import pandas as pd

# Constants and Mappings
RAVDESS_PATH = r"c:/Users/Nandhini Prbakaran/OneDrive/Documents/Minor_Nan_Audio/Minor_Nan_Audio/ravdess"
TESS_PATH = r"c:/Users/Nandhini Prbakaran/OneDrive/Documents/Minor_Nan_Audio/Minor_Nan_Audio/tess/TESS Toronto emotional speech set data"
OUTPUT_FILE = r"c:/Users/Nandhini Prbakaran/OneDrive/Documents/Minor_Nan_Audio/Minor_Nan_Audio/metadata.csv"

# Unified Emotion Map (6 classes)
# Angry, Disgust, Fear, Happy, Sad, Surprise
EMOTION_ID_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5
}

# RAVDESS: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprised
RAVDESS_EMOTION_CODE = {
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

# TESS: Folder names contain emotion
TESS_EMOTION_MAP = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'sad': 'sad',
    'pleasant_surprise': 'surprise',
    'pleasant_surprised': 'surprise'
}

def process_ravdess(path):
    records = []
    # Search recursively for wav files
    seen_filenames = set()
    files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    
    for f in files:
        filename = os.path.basename(f)
        if filename in seen_filenames:
            continue
            
        parts = filename.split('-')
        
        if len(parts) != 7:
            continue
            
        emotion_code = parts[2]
        speaker_code = parts[6].split('.')[0]
        
        if emotion_code in RAVDESS_EMOTION_CODE:
            seen_filenames.add(filename)
            emotion_label = RAVDESS_EMOTION_CODE[emotion_code]
            records.append({
                'file_path': f,
                'emotion': emotion_label,
                'emotion_id': EMOTION_ID_MAP[emotion_label],
                'speaker_id': f"ravdess_{speaker_code}",
                'dataset': 'RAVDESS'
            })
            
    return records

def process_tess(path):
    records = []
    # Iterate over directories
    if not os.path.exists(path):
        print(f"Warning: TESS path not found: {path}")
        return []

    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            # Parse folder name (e.g., OAF_fear, YAF_pleasant_surprise)
            parts = item.lower().split('_')
            
            # Identify emotion from folder name
            # Usually format is SPEAKER_EMOTION or SPEAKER_COMPLEX_EMOTION
            if len(parts) < 2:
                continue
            
            speaker = parts[0].upper() # OAF or YAF
            raw_emotion = "_".join(parts[1:])
            
            if raw_emotion in TESS_EMOTION_MAP:
                emotion_label = TESS_EMOTION_MAP[raw_emotion]
                
                # Get all wav files in this folder
                wav_files = glob.glob(os.path.join(item_path, "*.wav"))
                for f in wav_files:
                    records.append({
                        'file_path': f,
                        'emotion': emotion_label,
                        'emotion_id': EMOTION_ID_MAP[emotion_label],
                        'speaker_id': f"tess_{speaker}",
                        'dataset': 'TESS'
                    })
    return records

def main():
    print("Processing RAVDESS...")
    ravdess_data = process_ravdess(RAVDESS_PATH)
    print(f"RAVDESS samples found: {len(ravdess_data)}")
    
    print("Processing TESS...")
    tess_data = process_tess(TESS_PATH)
    print(f"TESS samples found: {len(tess_data)}")
    
    all_data = ravdess_data + tess_data
    df = pd.DataFrame(all_data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMetadata saved to {OUTPUT_FILE}")
    
    # Validation & Stats
    print("\nTotal Samples:", len(df))
    print("\nCounts by Dataset:")
    print(df['dataset'].value_counts())
    
    print("\nCounts by Emotion:")
    print(df['emotion'].value_counts())
    
    print("\nCounts by Emotion per Dataset:")
    print(df.groupby(['dataset', 'emotion']).size())

    # Verify no neutral/calm
    assert 'neutral' not in df['emotion'].unique()
    assert 'calm' not in df['emotion'].unique()
    print("\nValidation Passed: No 'neutral' or 'calm' samples found.")

if __name__ == "__main__":
    main()
