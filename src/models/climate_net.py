import torch
import torch.nn as nn

class EnvironmentalNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(EnvironmentalNet).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)           
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
