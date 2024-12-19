import torch
import string
import os
import torch.nn as nn

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

def char_to_index(char, model):
    return model.chars.index(char)

def index_to_char(index, model):
    return model.chars[index]

def sample_with_temperature(logits, temperature):
    probs = torch.nn.functional.softmax(logits / temperature, dim=0)
    return torch.multinomial(probs, 1).item()

# Load checkpoint
checkpoint_path = "model_checkpoint.pth"
if not os.path.exists(checkpoint_path):
    print("Checkpoint file not found. Please train the model first.")
    exit()

checkpoint = torch.load(checkpoint_path)
chars = checkpoint['chars']
hyperparameters = checkpoint['hyperparameters']

# Instantiate the model using hyperparameters from the checkpoint
model = TransformerModel(
    chars=chars,
    embed_size=hyperparameters['embed_size'],
    heads=hyperparameters['heads'],
    num_layers=hyperparameters['num_layers'],
    hidden_size=hyperparameters['hidden_size'],
    sequence_length=hyperparameters['sequence_length']
)

# Load the state dict.
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)  # Move the model to the appropriate device
model.eval()

print("Chat with the model! Type 'exit' to end the chat.")
temperature = 0.5
output_length = 200

while True:
    input_text = input("> ").lower()
    if input_text == 'exit':
        break

    char_indices = [char_to_index(char, model) for char in input_text]  # Pass model to char_to_index
    input_tensor = torch.tensor([char_indices]).to(device)  # Move the input tensor to the same device as the model

    # Generate additional characters after the user input
    for _ in range(output_length):
        with torch.no_grad():
            output, _ = model(input_tensor)
            char_idx = sample_with_temperature(output[0, -1], temperature) 
            output_char = index_to_char(char_idx, model)  # Pass model to index_to_char
            print(output_char, end='', flush=True)
            char_indices.append(char_idx)
            input_tensor = torch.tensor([char_indices]).to(device)  # Move the updated input tensor to the same device

    print()  # to move to a new line after the generated sequence

print("Chat ended.")