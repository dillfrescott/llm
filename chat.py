import torch
import string
import os
import torch.nn as nn

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

        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size, nhead=heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(embed_size, self.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

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
hyperparameters = checkpoint['hyperparameters']

# First, we create a dummy list of chars to instantiate the model
dummy_chars = ['a']  # This is just a placeholder

model = TransformerModel(
    chars=checkpoint['chars'],
    embed_size=hyperparameters['embed_size'],
    heads=hyperparameters['heads'],
    num_layers=hyperparameters['num_layers'],
    hidden_size=hyperparameters['hidden_size'],
    sequence_length=hyperparameters['sequence_length'],
    lr=hyperparameters['lr'],
    epochs=hyperparameters['epochs'],
    checkpoint_interval=hyperparameters['checkpoint_interval'],
    clip_value=hyperparameters['clip_value']
)

# Now load the state dict.
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Chat with the model! Type 'exit' to end the chat.")
temperature = 0.5
output_length = 200

while True:
    input_text = input("> ").lower()
    if input_text == 'exit':
        break

    char_indices = [char_to_index(char, model) for char in input_text]  # Pass model to char_to_index
    input_tensor = torch.tensor([char_indices])

    # Generate additional characters after the user input
    for _ in range(output_length):
        output = model(input_tensor)
        char_idx = sample_with_temperature(output[0, -1], temperature) 
        output_char = index_to_char(char_idx, model)  # Pass model to index_to_char
        print(output_char, end='', flush=True)
        char_indices.append(char_idx)
        input_tensor = torch.tensor([char_indices])

    print()  # to move to a new line after the generated sequence

print("Chat ended.")