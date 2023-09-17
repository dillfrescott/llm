import torch
import json
import random
import torch.nn as nn
from torch.utils.data import DataLoader

class TransformerNextWordPrediction(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super(TransformerNextWordPrediction, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, num_heads, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        out = self.transformer(src, tgt)
        out = self.fc(out)
        return out
        
def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as vocab_file:
        vocab = json.load(vocab_file)
    return vocab

def generate_text(model, vocab, start_text, max_length=200, device="cuda", temperature=1.0):
    model.eval()
    generated_text = start_text

    with torch.no_grad():
        for _ in range(max_length):
            input_indices = [vocab.index(char) for char in generated_text]
            input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
            
            # Ensure the model is on the same device as the input tensor
            model = model.to(device)

            # Generate the next word with temperature
            predictions = model(input_tensor, input_tensor)
            predicted_logits = predictions[:, -1, :]
            predicted_logits /= temperature
            predicted_probs = torch.softmax(predicted_logits, dim=-1)
            predicted_idx = torch.multinomial(predicted_probs, num_samples=1).item()
            predicted_char = vocab[predicted_idx]

            generated_text += predicted_char

            # Stop if an end token is encountered
            if predicted_char == "\n" or len(generated_text) >= max_length:
                break

    return generated_text

if __name__ == '__main__':
    model_path = "final_model.pth"  # Replace with the actual path to your trained model checkpoint
    vocab_path = "vocab.json"  # Replace with the actual path to your vocabulary JSON file

    vocab = load_vocab(vocab_path)
    model = TransformerNextWordPrediction(len(vocab), d_model=512, num_heads=4, num_layers=6)

    # Load the model state_dict
    model = load_checkpoint(model_path, model)

    start_text = "Your starting text is"  # Replace with your desired starting text
    generated_text = generate_text(model, vocab, start_text, max_length=200, device="cuda")

    print(generated_text)