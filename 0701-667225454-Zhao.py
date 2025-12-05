import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


FILENAME = 'names.txt'
MODEL_SAVE_PATH = '0702-667225454-Zhao.ZZZ'
SEQ_LENGTH = 11
HIDDEN_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.005

# Preprocessing
char_to_idx = {'EON': 0}
for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
    char_to_idx[char] = i + 1

idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = 27

def encode_name(name):
    name = name.lower().strip()
    indices = [char_to_idx[c] for c in name]

    x_idxs = indices + [0] * (SEQ_LENGTH - len(indices))
    y_idxs = indices[1:] + [0] * (SEQ_LENGTH - len(indices) + 1)
    
    # Truncate to strictly length 11 just in case
    x_idxs = x_idxs[:SEQ_LENGTH]
    y_idxs = y_idxs[:SEQ_LENGTH]
    
    return x_idxs, y_idxs

def get_data():
    try:
        with open(FILENAME, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return [], []

    X_data = []
    Y_data = []

    for line in lines:
        if not line.strip(): continue
        x_idx, y_idx = encode_name(line)

        x_tensor = torch.zeros(SEQ_LENGTH, VOCAB_SIZE)
        for t, idx in enumerate(x_idx):
            x_tensor[t][idx] = 1
            
        X_data.append(x_tensor)
        Y_data.append(torch.tensor(y_idx, dtype=torch.long))

    return torch.stack(X_data), torch.stack(Y_data)

# Model
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        # batch_first=True expects (Batch, Seq, Feature)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        
        out = out.reshape(-1, self.hidden_size)
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

#Training
def train():
    X, Y = get_data()
    if len(X) == 0: return

    model = CharLSTM(VOCAB_SIZE, HIDDEN_SIZE, VOCAB_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []

    print(f"Training on {len(X)} names...")

    for epoch in range(NUM_EPOCHS):
        hidden = model.init_hidden(X.size(0))
        
        optimizer.zero_grad()
        
        # Forward pass
        output, hidden = model(X, hidden)
        
        # Flatten targets to match output (Batch * Seq)
        loss = criterion(output, Y.view(-1))
        
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Plotting Loss
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    plt.show()

if __name__ == '__main__':
    train()