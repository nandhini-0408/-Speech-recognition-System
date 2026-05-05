import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(BaselineCNN, self).__init__()
        # Input: (B, 1, 128, 188)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # -> (32, 64, 94)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # -> (64, 32, 47)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) # -> (128, 16, 23)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2) # -> (256, 8, 11)
        
        self.flatten_size = 256 * 8 * 11
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BaselineCNNLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super(BaselineCNNLSTM, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2)) 
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        # Conv Block 4
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 2)) # -> (B, 256, 8, 11)
        
        self.hidden_dim = 256 * 8
        self.lstm_input_dim = self.hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Linear(128 * 2, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # (B, 256, 8, 11) -> (B, 11, 256, 8) -> (B, 11, 2048)
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, C, F_dim = x.size()
        x = x.reshape(batch_size, time_steps, C * F_dim)
        
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        x = self.fc(x)
        return x

# --- Phase 4 Models ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Time, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # x: (Batch, Time, d_model)
        # weights: (Batch, Time, 1)
        w = self.attention(x)
        weights = F.softmax(w, dim=1)
        # Weighted sum: (Batch, d_model)
        out = torch.sum(x * weights, dim=1)
        return out

class CNNTransformer(nn.Module):
    def __init__(self, num_classes=6, d_model=256, nhead=4, num_layers=2, ablation='none'):
        super(CNNTransformer, self).__init__()
        self.ablation = ablation
        
        # 1. CNN Backbone (Same as Baseline)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2)) 
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 2)) 
        # Output: (B, 256, 8, 11)
        
        # Feature Projection: 256*8 = 2048 -> d_model
        self.input_dim = 256 * 8
        self.projector = nn.Linear(self.input_dim, d_model)
        
        # 2. Transformer
        if self.ablation != 'no_transformer':
            self.pos_encoder = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
        # 3. Pooling
        if self.ablation != 'no_attention':
            self.attn_pooling = AttentionPooling(d_model)
            
        # 4. Classifier
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # CNN
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Shape: (B, 256, 8, 11) -> (B, 11, 2048)
        x = x.permute(0, 3, 1, 2)
        B, T, C, Freq = x.size()
        x = x.reshape(B, T, C*Freq)
        
        # Project
        x = self.projector(x) # (B, T, d_model)
        
        # Transformer
        if self.ablation != 'no_transformer':
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            
        # Pooling
        if self.ablation == 'no_attention':
            # Mean Pooling
            x = torch.mean(x, dim=1)
        else:
            # Attention Pooling
            x = self.attn_pooling(x)
            
        # Classification
        x = self.fc(x)
        return x
