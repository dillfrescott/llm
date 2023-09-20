import torch
import torch.nn as nn
import torch.optim as optim
import string
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
    def __init__(self, chars, embed_size, heads, num_layers, hidden_size, sequence_length, lr, epochs, checkpoint_interval, clip_value):
        super(TransformerModel, self).__init__()
        self.chars = chars
        self.vocab_size = len(chars)
        self.embed_size = embed_size
        self.heads = heads
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_interval = checkpoint_interval
        self.clip_value = clip_value

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size, nhead=heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

# Read and process data
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return text

def char_to_index(char, char_list):
    return char_list.index(char)

def index_to_char(index, char_list):
    return char_list[index]

text = read_data("valid.txt")
chars = sorted(list(set(text + string.punctuation + ' ')))
vocab_size = len(chars)

# Hyperparameters
embed_size = 1024
hidden_size = 1024
sequence_length = 512
heads = 8
num_layers = 4
lr = 0.0001
epochs = 100
checkpoint_interval = 2000
clip_value = 1.0

model = TransformerModel(chars, embed_size, heads, num_layers, hidden_size, sequence_length, lr, epochs, checkpoint_interval, clip_value)
model = model.to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Load previous checkpoint if exists
checkpoint_path = "model_checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    chars = checkpoint['chars']
    print("Loaded checkpoint!")

# Training the model
for epoch in range(epochs):
    total_steps = (len(text) - sequence_length) // sequence_length
    pbar = tqdm(range(0, len(text) - sequence_length, sequence_length), desc=f"Epoch {epoch+1}/{epochs}", total=total_steps)
    
    for i in pbar:
        inputs = torch.tensor([char_to_index(c, chars) for c in text[i:i+sequence_length]], dtype=torch.long).to(device)
        targets = torch.tensor([char_to_index(c, chars) for c in text[i+1:i+1+sequence_length]], dtype=torch.long).to(device)
    
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(0))
        loss = criterion(outputs.squeeze(0), targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.clip_value)

        optimizer.step()
        
        # Update the progress bar description to include the current loss
        pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
        
        # Save checkpoint at specified intervals
        if (i // sequence_length) % checkpoint_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'chars': model.chars,
                'hyperparameters': {
                    'vocab_size': model.vocab_size,
                    'embed_size': model.embed_size,
                    'heads': model.heads,
                    'num_layers': model.num_layers,
                    'hidden_size': model.hidden_size,
                    'sequence_length': model.sequence_length,
                    'lr': model.lr,
                    'epochs': model.epochs,
                    'checkpoint_interval': model.checkpoint_interval,
                    'clip_value': model.clip_value,
                }
            }, checkpoint_path)
            print("Saved checkpoint!")

    # Step the learning rate scheduler
    scheduler.step()

print("Training complete!")
