import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import json
import subprocess
import argparse


# ====================== Model ======================

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
                input_text = generated_text[-len(input_text):]  # Use the recent sequence length
        return generated_text


# ====================== Dataset ======================

class NextCharPredictionDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

        self.input_sequences = []
        self.target_sequences = []

        for i in range(0, len(text) - seq_length):
            input_seq = text[i:i + seq_length]
            target_seq = text[i + 1:i + seq_length + 1]
            input_tensor = torch.tensor([self.char_to_idx[char] for char in input_seq])
            target_tensor = torch.tensor([self.char_to_idx[char] for char in target_seq])
            self.input_sequences.append(input_tensor)
            self.target_sequences.append(target_tensor)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]
    
    def get_all_data(self):
        inputs = []
        targets = []
        for i in range(len(self)):
            input_seq, target_char = self[i]
            inputs.append(input_seq)
            targets.append(target_char)
        return torch.stack(inputs), torch.stack(targets)


# ====================== Utility Functions ======================

def save_checkpoint(epoch, model, optimizer, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'char_to_idx': model.char_to_idx,
        'idx_to_char': model.idx_to_char,
        # Add any other hyperparameters or training stats you want to save
    }, filename)


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.set_char_mappings(checkpoint['char_to_idx'], checkpoint['idx_to_char'])
    return model, optimizer, epoch


def train_epoch(model, dataloader, optimizer, criterion, device, writer, global_step):
    model.train()
    total_loss = 0.0
    for step, (batch_input, batch_target) in enumerate(dataloader):
        batch_input, batch_target = batch_input.to(device), batch_target.to(device)
        optimizer.zero_grad()

        output = model(batch_input)
        output = output.view(-1, len(model.char_to_idx))
        batch_target = batch_target.view(-1)

        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar('Training Loss', loss.item(), global_step)
        global_step += 1

    return total_loss / len(dataloader), global_step


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_input, batch_target in dataloader:
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)

            output = model(batch_input)
            output = output.view(-1, len(model.char_to_idx))
            batch_target = batch_target.view(-1)

            loss = criterion(output, batch_target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main_training_loop(model, train_dataset, val_dataset, optimizer, criterion, scheduler, device, config):
    # Create DataLoaders for training and validation datasets
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    writer = SummaryWriter(config['log_dir'])
    global_step = 0

    for epoch in range(config['start_epoch'], config['num_epochs']):
        # Train
        model.train()
        train_losses = []
        for inputs_batch, targets_batch in train_dataloader:
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            writer.add_scalar('Training Loss', loss.item(), global_step)
            global_step += 1
        
        # Compute average training loss for the epoch
        train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs_batch, targets_batch in val_dataloader:
                inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
                
                outputs = model(inputs_batch)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets_batch.view(-1))
                val_losses.append(loss.item())
        
        # Compute average validation loss for the epoch
        val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        writer.add_scalar('Validation Loss', val_loss, global_step)
        
        if (epoch + 1) % config['save_interval'] == 0:
            save_checkpoint(epoch, model, optimizer, f"checkpoint_epoch_{epoch}.pth")

    save_checkpoint(config['num_epochs'], model, optimizer, "final_model.pth")
    writer.close()

# ====================== Configuration and Main Script ======================

def get_config(args):
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', '')
    else:
        text = "the quick brown fox jumps over the lazy dog the quick brown fox"

    avg_word_length = len(text) / len(text.split())
    seq_length = int(4 * avg_word_length)  # Roughly 4 times the average word length

    config = {
        'text': text,
        'seq_length': seq_length,
        'embedding_dim': args.embedding_dim or 128,
        'hidden_dim': args.hidden_dim or 256,
        'num_layers': args.num_layers or 2,
        'batch_size': args.batch_size or 64,
        'num_epochs': args.num_epochs or 30,
        'lr': args.lr or 0.001,
        'log_dir': args.log_dir or './logs',
        'save_interval': args.save_interval or 10,
        'start_epoch': 0
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='Train the language model.')
    parser.add_argument('--text_file', type=str, help="Path to the input text file")
    parser.add_argument('--embedding_dim', type=int, help="Size of the word embeddings")
    parser.add_argument('--hidden_dim', type=int, help="Size of the GRU hidden layer")
    parser.add_argument('--num_layers', type=int, help="Number of GRU layers")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--log_dir', type=str, help="Directory to save Tensorboard logs")
    parser.add_argument('--save_interval', type=int, help="Epochs interval to save checkpoints")
    parser.add_argument('--resume', type=str, help='Path to the checkpoint to resume training from.')

    args = parser.parse_args()
    config = get_config(args)
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        # Load any other saved data as needed
    else:
        start_epoch = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NextCharPredictionDataset(config['text'], config['seq_length'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = GRULanguageModel(dataset.vocab_size, config['embedding_dim'], config['hidden_dim'], config['num_layers']).to(device)
    model.set_char_mappings(dataset.char_to_idx, dataset.idx_to_char)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    main_training_loop(model, train_dataset, val_dataset, optimizer, criterion, scheduler, device, config)

if __name__ == "__main__":
    main()