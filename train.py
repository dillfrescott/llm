import torch
import torch.nn as nn
import torch.optim as optim
import string
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the GRU-based language model
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

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
embed_size = 8096
hidden_size = 8096
sequence_length = 4096
lr = 0.0001
epochs = 100
checkpoint_interval = 100

model = GRUModel(vocab_size, embed_size, hidden_size)
model = model.to(device) # Transfer model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

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
        
        # Initialize hidden state for each sequence
        hidden = torch.zeros(1, 1, hidden_size).to(device)
    
        optimizer.zero_grad()
        outputs, hidden = model(inputs.unsqueeze(0), hidden)
        loss = criterion(outputs.squeeze(0), targets)
        loss.backward()
        optimizer.step()

        # Update the progress bar description to include the current loss
        pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
        
        # Save checkpoint at specified intervals
        if (i // sequence_length) % checkpoint_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'chars': chars,
            }, checkpoint_path)
            print("Saved checkpoint!")

print("Training complete!")
