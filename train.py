import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import json

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, seq_length):
        super(TransformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.seq_length = seq_length

    def forward(self, x):
        x = self.embedding(x)
        
        # In an autoregressive setup, use the same sequence for src and tgt
        out = self.transformer(x, x)
        
        out = self.fc(out)
        return out

    def generate_text(self, start_text, max_length=200, device="cuda"):
        generated_text = start_text
        input_text = start_text

        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([[self.char_to_idx.get(char, 0) for char in input_text]], dtype=torch.long).to(device)

                predictions = self(input_tensor)
                predictions = predictions[:, -1, :].squeeze()

                predicted_idx = torch.argmax(predictions, dim=-1).item()
                predicted_char = self.idx_to_char[predicted_idx]

                generated_text += str(predicted_char)  # Convert to string before concatenation
                input_text = generated_text[-self.seq_length:]

        return generated_text

# Custom dataset for language modeling with a fixed sequence length
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.text = text
        self.vocab = sorted(set(self.text))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Preprocess the text to generate sequences of a consistent length
        self.sequences = []
        for i in range(0, len(self.text) - seq_length, seq_length):
            input_seq = self.text[i:i + seq_length]
            target_seq = self.text[i + 1:i + seq_length + 1]
            input_tensor = torch.tensor([self.char_to_idx[char] for char in input_seq])
            target_tensor = torch.tensor([self.char_to_idx[char] for char in target_seq])
            self.sequences.append((input_tensor, target_tensor))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# Initialize SummaryWriter for TensorBoard logging
log_dir = "./logs"  # Set your preferred log directory
writer = SummaryWriter(log_dir)

# Hyperparameters
d_model = 512
hidden_size = 1024
num_layers = 4
seq_length = 2048 # Adjust to your desired sequence length
learning_rate = 0.0001
num_epochs = 1000
save_interval = 1  # Save every 'save_interval' epochs
batch_size = 8  # Adjust batch size as per GPU memory

# Load your text data and create the dataset
with open("output.txt", 'r', encoding='utf-8') as file:
    text = file.read()
dataset = TextDataset(text, seq_length)

# Save the vocabulary to a JSON file
with open("vocab.json", "w") as outfile:
    json.dump(dataset.vocab, outfile)

# Initialize the model, loss function, and optimizer for the Transformer-based model
vocab_size = dataset.vocab_size  # Define vocab_size based on your dataset
nhead = 64  # Define the number of attention heads
num_encoder_layers = 12  # Define the number of transformer encoder layers
model = TransformerLanguageModel(vocab_size, d_model, nhead, num_encoder_layers, seq_length)
model = model.cuda()  # Move model to GPU
model.char_to_idx = dataset.char_to_idx  # Add character-to-index mapping to the model
model.idx_to_char = dataset.idx_to_char  # Add index-to-character mapping to the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Function to save a checkpoint
def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for step, (batch_input, batch_target) in enumerate(dataloader):
        batch_input, batch_target = batch_input.cuda(), batch_target.cuda()  # Move data to GPU
        optimizer.zero_grad()

        predictions = model(batch_input)

        # Reshape the predictions and targets
        predictions = predictions.view(-1, vocab_size)
        batch_target = batch_target.view(-1)
        
        loss = criterion(predictions, batch_target)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # Print step information
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss.item()}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')

    # Update the learning rate based on the loss at this epoch
    scheduler.step(total_loss / len(dataloader))

    # Log loss to TensorBoard
    writer.add_scalar('Loss', total_loss / len(dataloader), global_step=epoch)

    # Save checkpoint every 'save_interval' epochs
    if (epoch + 1) % save_interval == 0:
        save_checkpoint(epoch, model, optimizer, f"checkpoint_epoch_{epoch + 1}.pth")

# Save the final trained model
save_checkpoint(num_epochs - 1, model, optimizer, "final_model.pth")  # Use the last epoch number

# Save the vocabulary to a JSON file
with open("vocab.json", "w") as outfile:
    json.dump(model.char_to_idx, outfile)

# Close the TensorBoard writer
writer.close()