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

def calculate_diversity(sequence):
    """Calculate the diversity of the generated sequence as the ratio of unique tokens to total tokens."""
    unique_tokens = len(set(sequence))
    total_tokens = len(sequence)
    return unique_tokens / total_tokens

def adaptive_temperature_adjustment(sequence, temperature, min_temp, max_temp, target_diversity):
    """Adjust the temperature based on the diversity of the generated sequence."""
    diversity = calculate_diversity(sequence)
    
    # If diversity is lower than target, increase temperature
    if diversity < target_diversity:
        temperature *= 1.05
    # If diversity is higher than target, decrease temperature
    elif diversity > target_diversity:
        temperature *= 0.95
    
    # Ensure temperature stays within bounds
    temperature = max(min_temp, min(max_temp, temperature))
    
    return temperature

def generate_with_adaptive_contrastive_decoding(input_tensor, model, strong_temperature, weak_temperature, output_length, min_temp, max_temp, target_diversity):
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

        # Convert index to character and stream it to the terminal
        output_char = index_to_char(char_idx, model)
        print(output_char, end='', flush=True)

        # Append to input and continue
        char_indices.append(char_idx)
        input_tensor = torch.tensor([char_indices])

        # Adjust the strong temperature adaptively after each token generation
        strong_temperature = adaptive_temperature_adjustment(generated_sequence, strong_temperature, min_temp, max_temp, target_diversity)

    # Return the full sequence for any other use if needed
    return generated_sequence

# Update the main loop to use Adaptive Contrastive Decoding
print("Chat with the model using Adaptive Contrastive Decoding! Type 'exit' to end the chat.")
strong_temperature = 0.5  # Initial temperature for the strong model
weak_temperature = 1.5  # Temperature for the weak model
output_length = 200

# Parameters for adaptive temperature adjustment
min_temperature = 0.2  # Minimum bound for the temperature
max_temperature = 1.5  # Maximum bound for the temperature
target_diversity = 0.7  # Target ratio of unique tokens to total tokens

while True:
    input_text = input("> ").lower()
    if input_text == 'exit':
        break

    char_indices = [char_to_index(char, model) for char in input_text]
    input_tensor = torch.tensor([char_indices])

    # Generate sequence using Adaptive Contrastive Decoding
    generated_sequence = generate_with_adaptive_contrastive_decoding(input_tensor, model, strong_temperature, weak_temperature, output_length, min_temperature, max_temperature, target_diversity)

    # Convert indices to characters and print
    output_chars = [index_to_char(idx, model) for idx in generated_sequence]
    print(''.join(output_chars))

print("Chat ended.")