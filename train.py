import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import json
import tensorboard
import subprocess
import argparse

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
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

                generated_text += str(predicted_char)
                input_text = generated_text[-self.seq_length:]

        return generated_text

class NextWordPredictionDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.text = text
        self.vocab = sorted(set(self.text))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        self.input_sequences = []
        self.target_sequences = []
        for i in range(0, len(self.text) - seq_length, seq_length):
            input_seq = self.text[i:i + seq_length]
            target_seq = self.text[i + 1:i + seq_length + 1]
            input_tensor = torch.tensor([self.char_to_idx[char] for char in input_seq])
            target_tensor = torch.tensor([self.char_to_idx[char] for char in target_seq])
            self.input_sequences.append(input_tensor)
            self.target_sequences.append(target_tensor)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]

def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def train_lstm(model, dataloader, optimizer, criterion, scheduler, num_epochs, save_interval, device, log_dir, start_epoch=0):
    model = model.to(device)
    writer = SummaryWriter(log_dir)
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0

        for step, (batch_input, batch_target) in enumerate(dataloader):
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            optimizer.zero_grad()

            # Forward pass through the model
            output = model(batch_input)

            # Reshape the output and target sequences to compute the loss
            output = output.view(-1, vocab_size)
            batch_target = batch_target.view(-1)

            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log loss per batch (step)
            writer.add_scalar('Loss', loss.item(), global_step=step + epoch * len(dataloader))

            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss.item()}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')
        scheduler.step(total_loss / len(dataloader))

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(epoch, model, optimizer, f"checkpoint_epoch_{epoch + 1}.pth")

    save_checkpoint(num_epochs - 1, model, optimizer, "final_model.pth")
    writer.close()

def create_model_and_optimizer(vocab_size, hyperparameters):
    model = LSTMLanguageModel(vocab_size, **hyperparameters['model'])
    optimizer = optim.Adam(model.parameters(), **hyperparameters['optimizer'])
    return model, optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Language Model Training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume training')
    args = parser.parse_args()

    log_dir = "./logs"
    num_epochs = 1000
    save_interval = 1
    seq_length = 1024
    subprocess.Popen(["tensorboard", "--logdir", log_dir])
    with open("output.txt", 'r', encoding='utf-8') as file:
        text = file.read()

    # Load the entire dataset into RAM
    dataset = NextWordPredictionDataset(text, seq_length)
    vocab_size = dataset.vocab_size  # Calculate vocab_size based on your dataset

    # Save the vocab file immediately
    with open("vocab.json", "w") as outfile:
        json.dump(dataset.char_to_idx, outfile)

    hyperparameters = {
        'model': {
            'embedding_dim': 512,
            'hidden_dim': 512,
            'num_layers': 3,
        },
        'optimizer': {
            'lr': 0.001,
        },
    }

    if args.checkpoint is not None:
        # If a checkpoint path is provided, load the model and optimizer state from it
        checkpoint = torch.load(args.checkpoint)
        model, optimizer = create_model_and_optimizer(vocab_size, hyperparameters)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        # Otherwise, create a new model and optimizer
        model, optimizer = create_model_and_optimizer(vocab_size, hyperparameters)
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    train_lstm(model, dataset, optimizer, criterion, scheduler, num_epochs, save_interval, "cuda", log_dir, start_epoch)