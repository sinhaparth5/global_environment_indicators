import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class ModelTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train(self, train_loader, val_loader, epochs=100):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
            train_losses.append(train_loss/len(train_loader))
            val_losses.append(val_loss/len(val_loader))
            
            if (epoch + 1) % 10 ==0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        
        return train_losses, val_losses