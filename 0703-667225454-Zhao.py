import torch
import torch.nn as nn
import numpy as np

MODEL_PATH = '0702-667225454-Zhao.ZZZ'
HIDDEN_SIZE = 128
VOCAB_SIZE = 27

char_to_idx = {'EON': 0}
for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
    char_to_idx[char] = i + 1
idx_to_char = {v: k for k, v in char_to_idx.items()}

# Re-define Model Class
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
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

def generate_names(start_char, num_names=20, temperature=1.0):
    model = CharLSTM(VOCAB_SIZE, HIDDEN_SIZE, VOCAB_SIZE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        return

    model.eval()
    
    generated_names = []
    
    # Initial Input
    if start_char.lower() not in char_to_idx:
        return

    print(f"Generating {num_names} names starting with '{start_char}':")
    print("-" * 30)

    with torch.no_grad():
        for _ in range(num_names):
            current_char = start_char.lower()
            name_chars = [current_char]

            hidden = model.init_hidden(1)
            
            for _ in range(20): 
                # Create one-hot input
                x = torch.zeros(1, 1, VOCAB_SIZE)
                x[0, 0, char_to_idx[current_char]] = 1

                output, hidden = model(x, hidden)

                probs = torch.softmax(output / temperature, dim=1).data.view(-1)
                
                # Soften algorithm
                next_idx = torch.multinomial(probs, 1).item()
                
                if next_idx == 0:
                    break
                    
                current_char = idx_to_char[next_idx]
                name_chars.append(current_char)
            
            generated_names.append("".join(name_chars))
            print("".join(name_chars))
            
    return generated_names

if __name__ == '__main__':
    first_initial = 's' 
    last_initial = 'n'
    
    generate_names(first_initial, 20)
    print("\n")
    generate_names(last_initial, 20)