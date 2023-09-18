import torch
from torch.utils.data import DataLoader
import json
import torch.nn as nn
device = "cuda"
class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(GRULanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.gru(embedded)
        out = self.fc(out)
        return out

    def generate_text(self, start_text, max_length=200, device="cuda"):
        generated_text = start_text
        input_text = start_text

        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([[char_to_idx.get(char, 0) for char in input_text]], dtype=torch.long).to(device)

                predictions = self(input_tensor)
                predictions = predictions[:, -1, :].squeeze()

                predicted_idx = torch.argmax(predictions, dim=-1).item()
                predicted_char = self.idx_to_char[predicted_idx]

                generated_text += str(predicted_char)
                input_text = generated_text[-self.seq_length:]

        return generated_text

# Load the vocabulary
with open("vocab.json", "r") as infile:
    char_to_idx = json.load(infile)

idx_to_char = {idx: char for char, idx in char_to_idx.items()}
vocab_size = len(char_to_idx)

# Create a new model with the same architecture
model = GRULanguageModel(vocab_size, embedding_dim=512, hidden_dim=2048, num_layers=6)

# Load the checkpoint weights into the model
checkpoint = torch.load("checkpoint_step_6000.pth", map_location=torch.device("cuda"))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Function to generate text
def generate_text(prompt, max_length=200, temperature=1.0):
    input_text = prompt
    generated_text = ""  # Initialize generated_text as an empty string

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([[char_to_idx.get(char, 0) for char in input_text]], dtype=torch.long).to(device)

            predictions = model(input_tensor)
            predictions = predictions[:, -1, :].squeeze() / temperature  # Adjust for temperature

            # Apply softmax to the scaled predictions
            predicted_probs = torch.softmax(predictions, dim=-1)

            # Sample from the probability distribution
            predicted_idx = torch.multinomial(predicted_probs, 1).item()
            predicted_char = idx_to_char[predicted_idx]

            generated_text += str(predicted_char)
            input_text += str(predicted_char)  # Update the input text with the predicted character

            # Print the character as it's generated (flush=True ensures it's printed immediately)
            print(predicted_char, end='', flush=True)

            if predicted_char == '\n':
                break  # Stop if the model predicts a newline character

    print()  # Print a newline after generating the text

# Chat with the model
print("GRU Language Model Chat (Type 'exit' to quit)")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        break

    print("Model:", end='', flush=True)
    response = generate_text(user_input, temperature=0.6)
    print()