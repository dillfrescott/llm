import torch
import torch.nn as nn
import string
import os
from tqdm import tqdm
import gc
from prodigyopt import Prodigy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(device)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, chars, embed_size, heads, num_layers, hidden_size, sequence_length):
        super(TransformerModel, self).__init__()
        self.chars = chars
        self.vocab_size = len(chars)
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, sequence_length)
        
        # Use TransformerEncoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dim_feedforward=hidden_size)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_size, self.vocab_size)

    def forward(self, x, memory=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)  # No memory required for encoder-only
        x = self.fc(x)
        return x, memory

# Dynamic Dataset Creation
def get_batch(text, chars, sequence_length, batch_size):
    X, Y = [], []
    for _ in range(batch_size):
        start_idx = torch.randint(0, len(text) - sequence_length - 1, (1,)).item()
        seq = text[start_idx:start_idx + sequence_length]
        target = text[start_idx + 1:start_idx + 1 + sequence_length]
        X.append([chars.index(c) for c in seq])
        Y.append([chars.index(c) for c in target])
    return torch.tensor(X, dtype=torch.long).to(device), torch.tensor(Y, dtype=torch.long).to(device)

# Read and process data
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return text

# Hyperparameters
text = read_data("shake.txt")
chars = sorted(list(set(text + string.punctuation + ' ')))
sequence_length = 4096
batch_size = 4
embed_size = 1024
hidden_size = 2048
heads = 8
num_layers = 24
epochs = 10000
clip_value = 5.0
weight_decay = 0.01

# Model and optimizer
model = TransformerModel(chars, embed_size, heads, num_layers, hidden_size, sequence_length)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Prodigy(model.parameters(), lr=1.0, weight_decay=weight_decay)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Training Loop
checkpoint_path = "model_checkpoint.pth"
step = 0
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    chars = checkpoint['chars']
    start_epoch = checkpoint['epoch']
    step = checkpoint['step']
    model = model.to(device)
    print("Loaded checkpoint!")

for epoch in range(start_epoch, epochs):
    pbar = tqdm(range(len(text) // batch_size), desc=f"Epoch {epoch+1}/{epochs}")
    for _ in pbar:
        inputs, targets = get_batch(text, chars, sequence_length, batch_size)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):  # Updated autocast
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, len(chars)), targets.view(-1))
        
        scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        scaler.step(optimizer)
        scaler.update()

        pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

        if step % 1000 == 0 and step > 0:
            # Save checkpoint with hyperparameters
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'chars': chars,
                'epoch': epoch,
                'step': step,
                'hyperparameters': {
                    'embed_size': embed_size,
                    'hidden_size': hidden_size,
                    'heads': heads,
                    'num_layers': num_layers,
                    'sequence_length': sequence_length,
                    'batch_size': batch_size,
                    'clip_value': clip_value,
                    'weight_decay': weight_decay,
                    'epochs': epochs
                }
            }, checkpoint_path)
            print("Saved checkpoint!")

        step += 1

print("Training complete!")