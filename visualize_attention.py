import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import CNNTransformer

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results_proposed/no_transformer/fold2_best.pth" # Fold 2 was strongest
DATA_CSV = "metadata_features.csv"
OUTPUT_FILE = "attention_map.png"

EMOTION_MAP = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise'
}

def get_attention_weights(model, x):
    """
    Hooks into AttentionPooling to get weights
    """
    weights = None
    def hook(module, input, output):
        nonlocal weights
        # out = torch.sum(x * weights, dim=1)
        # We need to capture 'weights' from the forward pass
        # But 'weights' is a local variable in AttentionPooling.forward.
        # Let's verify AttentionPooling class in models.py
        pass
    
    # Actually, let's just modify the model temporarily or use a wrapper
    # Since we can't easily hook a local variable, let's just 
    # re-implement the logic manually for visualization.
    
    model.eval()
    with torch.no_grad():
        # CNN Part
        x = model.pool1(torch.relu(model.bn1(model.conv1(x))))
        x = model.pool2(torch.relu(model.bn2(model.conv2(x))))
        x = model.pool3(torch.relu(model.bn3(model.conv3(x))))
        x = model.pool4(torch.relu(model.bn4(model.conv4(x))))
        
        x = x.permute(0, 3, 1, 2)
        B, T, C, Freq = x.size()
        x = x.reshape(B, T, C*Freq)
        x = model.projector(x) # (B, T, d_model)
        
        # In CNNTransformer.forward (ablation='no_transformer'):
        # x = self.attn_pooling(x)
        
        # Replicate AttentionPooling.forward logic
        w = model.attn_pooling.attention(x)
        attn_weights = torch.softmax(w, dim=1)
        
        return attn_weights.cpu().numpy().squeeze(), x.cpu().numpy().squeeze()

def main():
    print(f"Visualizing Attention Weights for {MODEL_PATH}...")
    
    # Load Model
    model = CNNTransformer(num_classes=6, ablation='no_transformer').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # Find a good sample (e.g., 'angry' or 'happy' from TESS)
    df = pd.read_csv(DATA_CSV)
    sample_row = df[(df['dataset'] == 'TESS') & (df['emotion'] == 'angry')].iloc[0]
    
    # Load and Normalize
    # We'll use global mean/std from Fold 2 split for consistency
    # (Simplified for visualization: just use the sample's own stats or a fixed value)
    path = sample_row['feature_path'].replace('e:/Minor_Nan_Audio/', '')
    feature = np.load(path)
    x = torch.FloatTensor(feature).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    weights, features = get_attention_weights(model, x)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Spectrogram-like (Mel features)
    img = ax1.imshow(feature, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title(f"Mel Spectrogram (TESS: {sample_row['emotion']})")
    ax1.set_ylabel("Mel Filter Bank")
    
    # 2. Attention Weights
    # The weights correspond to the 11 time steps after 4 max-pooling layers
    # (188 / 2^4 = 11.75 -> 11)
    time_steps = np.arange(len(weights))
    ax2.bar(time_steps, weights, color='red', alpha=0.7)
    ax2.set_title("Attention Weight Distribution")
    ax2.set_ylabel("Weight")
    ax2.set_xlabel("Condensed Time Steps (Aggregated Frames)")
    ax2.set_ylim(0, 1.1 * max(weights))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved attention map to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
