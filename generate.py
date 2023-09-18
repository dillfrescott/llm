import torch
import argparse
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(GRULanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.char_to_idx = {}
        self.idx_to_char = {}

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.gru(embedded)
        return self.fc(out)

    def set_char_mappings(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

    def generate_text(self, start_text, max_length=200, device="cuda", temperature=1.0):
        input_text = start_text
        response_text = ""

        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([[self.char_to_idx.get(char, 0) for char in input_text]], dtype=torch.long).to(device)
                logits = self(input_tensor)
                logits = logits[:, -1, :] / temperature  # Adjust logits based on temperature
                probabilities = F.softmax(logits, dim=-1).squeeze()
                predicted_idx = torch.multinomial(probabilities, 1).item()  # Sample from the adjusted distribution
                predicted_char = self.idx_to_char[predicted_idx]
                print(predicted_char, end="", flush=True)  # Stream the character to the terminal
                response_text += str(predicted_char)
                input_text = input_text[1:] + predicted_char  # Slide the window for next input

        return response_text
        
def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'char_to_idx': model.char_to_idx,
        'idx_to_char': model.idx_to_char
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(filename, device):
    checkpoint = torch.load(filename, map_location=device)
    
    # Extract embedded model parameters
    vocab_size = checkpoint['vocab_size']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    
    # Recreate the model using the embedded parameters
    model = GRULanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load character mappings if they were saved in the checkpoint
    if 'char_to_idx' in checkpoint and 'idx_to_char' in checkpoint:
        model.set_char_mappings(checkpoint['char_to_idx'], checkpoint['idx_to_char'])
    
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    
    return model, optimizer_state_dict, checkpoint['epoch']

def chat_with_model(model, device, temperature=1.0):
    model.eval()
    model.to(device)

    print("You can now chat with the model. Type 'exit' to end the conversation.")
    
    while True:
        start_text = input("You: ")
        
        # Check for exit command
        if start_text.strip().lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        generated_text = model.generate_text(start_text, device=device, temperature=temperature)
        print(f"Model: {generated_text}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=Path, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature for text generation. Higher values make output more random, lower values make it more deterministic.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model instance
    model = GRULanguageModel(1, 1, 1, 1).to(device)  # Dummy values, will be overridden by checkpoint

    # Load the checkpoint
    model, _, _ = load_checkpoint(args.checkpoint_path, device)

    chat_with_model(model, device)

if __name__ == "__main__":
    main()
