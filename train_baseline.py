import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from models import BaselineCNN, BaselineCNNLSTM
import argparse

# Config
DATA_CSV = r"c:/Users/Nandhini Prbakaran/OneDrive/Documents/Minor_Nan_Audio/Minor_Nan_Audio/metadata_features.csv"
OUTPUT_DIR = r"c:/Users/Nandhini Prbakaran/OneDrive/Documents/Minor_Nan_Audio/Minor_Nan_Audio/results_baseline"
BATCH_SIZE = 32
EPOCHS = 30 # Baseline usually converges fast
LEARNING_RATE = 0.001
# --- Device Selection ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    DEVICE = torch.device("cpu")
    print("WARNING: CUDA is NOT available. Running on CPU (this will be slow!)")
# -----------------------

# Emotions
EMOTION_MAP = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise'
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class SERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['feature_path']
        label = row['emotion_id']
        
        # Load .npy
        # Shape: (128, 188)
        feature = np.load(path)
        
        # Add Channel Dim -> (1, 128, 188)
        feature = feature[np.newaxis, ...]
        
        if self.transform:
            feature = self.transform(feature)
            
        return torch.FloatTensor(feature), torch.tensor(label, dtype=torch.long)

class Normalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        
    def fit(self, df):
        # Compute global mean and std from all files in df
        # To save memory, we can compute online or just load all (if RAM allows).
        # RAM Check: 2000 files * 128 * 188 * 4 bytes approx 200MB. Easy load all.
        print("Computing normalization stats...")
        all_feats = []
        for path in df['feature_path']:
            all_feats.append(np.load(path))
        
        all_feats = np.array(all_feats) # (N, 128, 188)
        self.mean = np.mean(all_feats)
        self.std = np.std(all_feats)
        print(f"Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        
    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-6)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    
    # Metrics
    wa = accuracy_score(all_targets, all_preds) # Weighted Accuracy (Overall)
    ua = recall_score(all_targets, all_preds, average='macro') # Unweighted Accuracy (Avg per-class recall)
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    
    return epoch_loss, wa, ua, f1, cm, all_targets, all_preds

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[EMOTION_MAP[i] for i in range(6)],
                yticklabels=[EMOTION_MAP[i] for i in range(6)])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_experiment(model_type='cnn'):
    print(f"Running Experiment: {model_type}")
    full_df = pd.read_csv(DATA_CSV)
    
    # Split Datasets
    ravdess_df = full_df[full_df['dataset'] == 'RAVDESS'].copy()
    tess_df = full_df[full_df['dataset'] == 'TESS'].copy()
    
    # Prepare results storage
    fold_results = []
    
    # GroupKFold on Speaker ID
    # Note: augmented data has speaker_id + "_aug". 
    # To keep groups clean, we should normalize speaker_id (strip _aug)
    ravdess_df['speaker_group'] = ravdess_df['speaker_id'].apply(lambda x: x.replace('_aug', ''))
    
    gkf = GroupKFold(n_splits=5)
    groups = ravdess_df['speaker_group'].values
    indices = np.arange(len(ravdess_df))
    
    ensure_dir(os.path.join(OUTPUT_DIR, model_type))
    
    fold = 1
    for train_idx, val_idx in gkf.split(indices, groups=groups):
        print(f"\n--- Fold {fold} ---")
        
        train_sub = ravdess_df.iloc[train_idx]
        val_sub = ravdess_df.iloc[val_idx]
        
        # IMPORTANT: Remove Augmented samples from Validation Set?
        # Usually, validation should check generalization. Augmented validation is debated.
        # User said "Validate ONLY on RAVDESS". Clean only is strict validation.
        # But if we validate on augmented, we might overestimate.
        # Let's filter VALIDATION set to be CLEAN ONLY.
        val_sub = val_sub[val_sub['augmented'] == False]
        
        print(f"Train Size: {len(train_sub)} (Clean+Aug)")
        print(f"Val Size: {len(val_sub)} (Clean Only)")
        
        # 1. Normalization (Fit on Train)
        normalizer = Normalizer()
        normalizer.fit(train_sub)
        
        # Datasets
        train_ds = SERDataset(train_sub, transform=normalizer.transform)
        val_ds = SERDataset(val_sub, transform=normalizer.transform)
        tess_ds = SERDataset(tess_df, transform=normalizer.transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        tess_loader = DataLoader(tess_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        if model_type == 'cnn':
            model = BaselineCNN(num_classes=6).to(DEVICE)
        else:
            model = BaselineCNNLSTM(num_classes=6).to(DEVICE)
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train Loop
        best_val_ua = 0.0
        best_model_path = os.path.join(OUTPUT_DIR, model_type, f"fold{fold}_best.pth")
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_wa, val_ua, val_f1, _, _, _ = evaluate(model, val_loader, criterion)
            
            if val_ua > best_val_ua:
                best_val_ua = val_ua
                torch.save(model.state_dict(), best_model_path)
            
            if (epoch+1) % 5 == 0:
                print(f"Ep {epoch+1}/{EPOCHS} | T.Loss: {train_loss:.4f} T.Acc: {train_acc:.4f} | V.Loss: {val_loss:.4f} V.UA: {val_ua:.4f}")
        
        # Final Evaluation with Best Model
        model.load_state_dict(torch.load(best_model_path))
        
        # Val Eval
        _, v_wa, v_ua, v_f1, v_cm, _, _ = evaluate(model, val_loader, criterion)
        plot_confusion_matrix(v_cm, f"Val CM Fold {fold}", os.path.join(OUTPUT_DIR, model_type, f"fold{fold}_val_cm.png"))
        
        # TESS Eval (Zero-Shot)
        _, t_wa, t_ua, t_f1, t_cm, _, _ = evaluate(model, tess_loader, criterion)
        plot_confusion_matrix(t_cm, f"TESS CM Fold {fold}", os.path.join(OUTPUT_DIR, model_type, f"fold{fold}_tess_cm.png"))
        
        print(f"Fold {fold} Results:")
        print(f"  RAVDESS Val -> UA: {v_ua:.4f}, WA: {v_wa:.4f}, F1: {v_f1:.4f}")
        print(f"  TESS Test   -> UA: {t_ua:.4f}, WA: {t_wa:.4f}, F1: {t_f1:.4f}")
        
        fold_results.append({
            'fold': fold,
            'val_ua': v_ua, 'val_wa': v_wa, 'val_f1': v_f1,
            'tess_ua': t_ua, 'tess_wa': t_wa, 'tess_f1': t_f1
        })
        
        fold += 1
        
    # Aggregate
    res_df = pd.DataFrame(fold_results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, model_type, "metrics.csv"), index=False)
    
    print("\n=== Final Aggregated Results ===")
    print(res_df.mean())
    print("\n================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'lstm'])
    args = parser.parse_args()
    
    run_experiment(args.model)
