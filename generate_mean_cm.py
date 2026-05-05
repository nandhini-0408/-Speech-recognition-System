import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from models import CNNTransformer

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_CSV = "metadata_features.csv"
RESULTS_DIR = "results_proposed/no_transformer"
OUTPUT_FILE = "mean_tess_cm.png"

EMOTION_MAP = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise'
}

class SERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['feature_path'].replace('e:/Minor_Nan_Audio/', '')
        feature = np.load(path)
        feature = feature[np.newaxis, ...]
        if self.transform:
            feature = self.transform(feature)
        return torch.FloatTensor(feature), torch.tensor(row['emotion_id'], dtype=torch.long)

class Normalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-6)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return confusion_matrix(all_targets, all_preds, labels=range(6))

def main():
    print(f"Generating Mean Confusion Matrix from {RESULTS_DIR}...")
    full_df = pd.read_csv(DATA_CSV)
    tess_df = full_df[full_df['dataset'] == 'TESS'].copy()
    
    # We need the normalization stats used during training for each fold
    # Since they aren't saved separately, we might need to approximate or recalculate
    # However, look at train_proposed.py: it computes them per fold.
    # To be accurate, we'd need to know which speakers were in which fold.
    # For now, let's use a global normalizer or recalculate if possible.
    # Let's try to find if there's a mean/std reported in the logs or results.
    
    # Actually, let's just use the best model from Fold 2 (highest UA) to represent
    # or sum all of them.
    
    total_cm = np.zeros((6, 6))
    
    for fold in range(1, 6):
        print(f"Processing Fold {fold}...")
        model_path = os.path.join(RESULTS_DIR, f"fold{fold}_best.pth")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping.")
            continue
            
        # Reconstruct the normalizer for this fold
        # In train_proposed.py, GroupKFold seeds are consistent.
        # We can replicate the split.
        ravdess_df = full_df[full_df['dataset'] == 'RAVDESS'].copy()
        ravdess_df['speaker_group'] = ravdess_df['speaker_id'].apply(lambda x: x.replace('_aug', ''))
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(np.arange(len(ravdess_df)), groups=ravdess_df['speaker_group'].values))
        train_idx, _ = splits[fold-1]
        train_sub = ravdess_df.iloc[train_idx]
        
        all_feats = []
        for path in train_sub['feature_path']:
            path = path.replace('e:/Minor_Nan_Audio/', '')
            all_feats.append(np.load(path))
        mean = np.mean(all_feats)
        std = np.std(all_feats)
        normalizer = Normalizer(mean, std)
        
        # Load Model
        model = CNNTransformer(num_classes=6, ablation='no_transformer').to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        # Dataset
        tess_ds = SERDataset(tess_df, transform=normalizer.transform)
        tess_loader = DataLoader(tess_ds, batch_size=32, shuffle=False)
        
        cm = evaluate(model, tess_loader)
        total_cm += cm
        
    mean_cm = total_cm / 5.0
    
    # Plotting
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=[EMOTION_MAP[i] for i in range(6)],
                yticklabels=[EMOTION_MAP[i] for i in range(6)])
    plt.title("Mean Confusion Matrix - TESS Zero-Shot (5-Fold CV)")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Saved mean confusion matrix to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
