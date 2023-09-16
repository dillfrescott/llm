import torch
import json
import torch.nn as nn

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, seq_length):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.char_to_idx = None
        self.idx_to_char = None
        self.seq_length = seq_length  # Added seq_length as an attribute

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x, x)  # Use the same input sequence for src and tgt
        out = self.fc(out)
        return out

    def generate_response(self, conversation, temperature=1.0, max_length=100, beam_width=5, device="cuda", context_window=3):
        # Extract the context window from the conversation history
        if len(conversation) > context_window:
            conversation = conversation[-context_window:]
        
        # Initialize with the conversation history
        generated_text = " ".join(conversation)
        conversation_history = generated_text
        self.to(device)  # Move the entire model to the specified device

        with torch.no_grad():
            # Initialize the input tensor with conversation history
            input_tensor = torch.tensor([[self.char_to_idx.get(char, 0) for char in generated_text]], dtype=torch.long).to(device)
            
            for _ in range(max_length):
                predictions = self(input_tensor)

                # Get predictions for the next token
                predictions = predictions[:, -1, :].squeeze().div(temperature).exp()
                predicted_idx = torch.multinomial(predictions, 1).item()
                predicted_char = self.idx_to_char[predicted_idx]
                
                # Append the predicted token to the generated text
                generated_text += str(predicted_char)
                input_tensor = torch.tensor([[self.char_to_idx.get(char, 0) for char in generated_text]], dtype=torch.long).to(device)
                
            # Extract the model's response from the best candidate
            model_response = generated_text[len(" ".join(conversation)):]

        return model_response

# Load vocabulary
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Load the trained model
model = TransformerLanguageModel(
    vocab_size=len(vocab),
    d_model=512,
    nhead=64,
    num_encoder_layers=12,
    seq_length=2048
)
model.load_state_dict(torch.load('checkpoint_epoch_1.pth')['model_state_dict'])
model.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
model.idx_to_char = {idx: char for char, idx in model.char_to_idx.items()}
model.seq_length = 2048
model.eval()

conversation_history = []  # Initialize an empty conversation history

print("Let's chat! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    conversation_history.append(user_input)  # Add the user's input to the conversation history

    # Generate a response from the model based on the conversation history with a configurable context window
    model_response = model.generate_response(conversation_history, max_length=100, context_window=3)
    
    print("Model: " + model_response)

    # Append the model's response to the conversation history
    conversation_history.append(model_response)
