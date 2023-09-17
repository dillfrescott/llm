import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import json

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
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def train_transformer(model, dataloader, optimizer, criterion, scheduler, num_epochs, save_interval, device, log_dir):
    model = model.to(device)
    writer = SummaryWriter(log_dir)
    
    for epoch in range(num_epochs):
        total_loss = 0.0

        for step, (batch_input, batch_target) in enumerate(dataloader):
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)
            optimizer.zero_grad()

            predictions = model(batch_input, batch_target)[:, -1, :]

            loss = criterion(predictions, batch_target[:, -1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss.item()}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')
        scheduler.step(total_loss / len(dataloader))
        writer.add_scalar('Loss', total_loss / len(dataloader), global_step=epoch)

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(epoch, model, optimizer, f"checkpoint_epoch_{epoch + 1}.pth")

    save_checkpoint(num_epochs - 1, model, optimizer, "final_model.pth")
    writer.close()

if __name__ == '__main__':
    log_dir = "./logs"
    num_epochs = 1000
    save_interval = 1
    batch_size = 16
    learning_rate = 0.001
    seq_length = 1024

    with open("output.txt", 'r', encoding='utf-8') as file:
        text = file.read()
    dataset = NextWordPredictionDataset(text, seq_length)

    with open("vocab.json", "w") as outfile:
        json.dump(dataset.vocab, outfile)

    vocab_size = dataset.vocab_size
    num_heads = 8  # Adjust the number of attention heads
    num_layers = 6  # Adjust the number of transformer layers

    model = TransformerNextWordPrediction(vocab_size, d_model=512, num_heads=num_heads, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    train_transformer(model, dataloader, optimizer, criterion, scheduler, num_epochs, save_interval, "cuda", log_dir)

    with open("vocab.json", "w") as outfile:
        json.dump(dataset.char_to_idx, outfile)
