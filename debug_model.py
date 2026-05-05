import torch
from models import CNNTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

try:
    model = CNNTransformer(num_classes=6, ablation='none').to(device)
    print("Model instantiated.")
    
    x = torch.randn(2, 1, 128, 188).to(device)
    print(f"Input shape: {x.shape}")
    
    out = model(x)
    print(f"Output shape: {out.shape}")
    print("Forward pass successful.")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
