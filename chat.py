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

def generate_with_contrastive_decoding_v2(input_tensor, model, strong_temperature, weak_temperature, output_length):
    char_indices = input_tensor[0].tolist()
    generated_sequence = []

    for _ in range(output_length):
        output = model(input_tensor)

        # Get probabilities with the strong model (lower temperature)
        strong_probs = torch.nn.functional.softmax(output[0, -1] / strong_temperature, dim=0)
        
        # Get probabilities with the weak model (higher temperature)
        weak_probs = torch.nn.functional.softmax(output[0, -1] / weak_temperature, dim=0)
        
        # Calculate contrastive score using the ratio
        contrastive_scores = strong_probs / (weak_probs + 1e-10)  # Adding a small value to avoid division by zero
        
        # Normalize the contrastive scores to get probabilities
        contrastive_probs = contrastive_scores / contrastive_scores.sum()
        
        # Sample from the contrastive probabilities
        char_idx = torch.multinomial(contrastive_probs, 1).item()
        
        generated_sequence.append(char_idx)

        # Append to input and continue
        char_indices.append(char_idx)
        input_tensor = torch.tensor([char_indices])

    return generated_sequence

# Update the main loop to use Contrastive Decoding
print("Chat with the model using Contrastive Decoding! Type 'exit' to end the chat.")
strong_temperature = 0.5  # Original temperature
weak_temperature = 1.5  # Higher temperature for the weak model
output_length = 200

while True:
    input_text = input("> ").lower()
    if input_text == 'exit':
        break

    char_indices = [char_to_index(char, model) for char in input_text]
    input_tensor = torch.tensor([char_indices])

    # Generate sequence using Contrastive Decoding
    generated_sequence = generate_with_contrastive_decoding_v2(input_tensor, model, strong_temperature, weak_temperature, output_length)

    # Convert indices to characters and print
    output_chars = [index_to_char(idx, model) for idx in generated_sequence]
    print(''.join(output_chars))

print("Chat ended.")