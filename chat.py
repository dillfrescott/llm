import torch
import string
import os
import torch.nn as nn

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

def char_to_index(char, char_list):
    return char_list.index(char)

def index_to_char(index, char_list):
    return char_list[index]

def sample_with_temperature(logits, temperature):
    probs = torch.nn.functional.softmax(logits / temperature, dim=0)
    return torch.multinomial(probs, 1).item()

# Load checkpoint
checkpoint_path = "model_checkpoint.pth"
if not os.path.exists(checkpoint_path):
    print("Checkpoint file not found. Please train the model first.")
    exit()

checkpoint = torch.load(checkpoint_path)

# Extract hyperparameters from the checkpoint
hyperparameters = checkpoint['hyperparameters']

# Instantiate model with hyperparameters
model = TransformerModel(
    vocab_size=hyperparameters['vocab_size'],
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

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Chat with the model! Type 'exit' to end the chat.")
temperature = 0.8
output_length = 200

while True:
    input_text = input("> ").lower()
    if input_text == 'exit':
        break

    char_indices = [char_to_index(char, chars) for char in input_text]
    input_tensor = torch.tensor([char_indices])

    # Generate additional characters after the user input
    for _ in range(output_length):
        output = model(input_tensor)
        char_idx = sample_with_temperature(output[0, -1], temperature) # sample from the last generated character
        output_char = index_to_char(char_idx, chars)
        print(output_char, end='', flush=True)  # print character immediately
        char_indices.append(char_idx)
        input_tensor = torch.tensor([char_indices])

    print()  # to move to a new line after the generated sequence

print("Chat ended.")