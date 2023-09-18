import torch
import torch.nn as nn
import json

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, seq_length):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.seq_length = seq_length  # Add seq_length as a parameter

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.fc(out)
        return out

    def generate_text(self, start_text, char_to_idx, max_length=200, device="cuda"):
        generated_text = start_text
        input_text = start_text

        with torch.no_grad():
            for _ in range(max_length):
                input_indices = [char_to_idx.get(char, char_to_idx["<UNK>"]) for char in input_text]
                input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

                predictions = self(input_tensor)
                predictions = predictions[:, -1, :].squeeze()

                predicted_idx = torch.argmax(predictions, dim=-1).item()
                predicted_char = idx_to_char.get(predicted_idx, "<UNK>")

                generated_text += str(predicted_char)
                input_text = generated_text[-self.seq_length:]

        return generated_text

# Load the vocabulary as a list of characters
with open("vocab.json", "r") as vocab_file:
    char_to_idx = {char: idx for idx, char in enumerate(json.load(vocab_file))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}  # Reverse mapping

# Add a special token for unknown characters
char_to_idx["<UNK>"] = len(char_to_idx)

# Make sure the vocabulary size matches the model's embedding size
vocab_size = len(char_to_idx)

# Define the model and load the trained model with the correct vocabulary size
model = LSTMLanguageModel(vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3, seq_length=128)

# Load the checkpoint (update the path to your checkpoint file)
checkpoint = torch.load("final_model.pth")

# Load the model state dict, ignoring the embedding layer and the last linear layer
model_state_dict = checkpoint['model_state_dict']
model_state_dict.pop('embedding.weight')
model_state_dict.pop('fc.weight')
model_state_dict.pop('fc.bias')

# Load the state dict into the model
model.load_state_dict(model_state_dict, strict=False)

# Set the device to 'cuda' or 'cpu' depending on your system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Set the device to 'cuda' or 'cpu' depending on your system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Chat with the model. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    generated_text = model.generate_text(user_input, char_to_idx, max_length=200, device=device)

    print("Model:", generated_text)