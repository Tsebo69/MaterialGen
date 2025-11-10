import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.get('training.learning_rate', 0.001)
        )
        self.criterion = nn.MSELoss()
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, epochs=None):
        epochs = epochs or self.config.get('training.epochs', 100)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss
                }, 'models/best_model.pth')
                
        return best_val_loss

class GANTrainer:
    def __init__(self, generator, discriminator, config):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.g_optimizer = optim.Adam(
            generator.parameters(), 
            lr=config.get('training.generator_lr', 0.0002)
        )
        self.d_optimizer = optim.Adam(
            discriminator.parameters(), 
            lr=config.get('training.discriminator_lr', 0.0002)
        )
        
        self.criterion = nn.BCELoss()
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, data_loader):
        g_losses = []
        d_losses = []
        
        for real_data, _ in data_loader:
            batch_size = real_data.size(0)
            real_data = real_data.to(self.device)
            
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            self.d_optimizer.zero_grad()
            
            real_output = self.discriminator(real_data)
            d_loss_real = self.criterion(real_output, real_labels)
            
            z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
            fake_data = self.generator(z)
            fake_output = self.discriminator(fake_data.detach())
            d_loss_fake = self.criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()
            
            self.g_optimizer.zero_grad()
            fake_output = self.discriminator(fake_data)
            g_loss = self.criterion(fake_output, real_labels)
            g_loss.backward()
            self.g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
        return np.mean(g_losses), np.mean(d_losses)
    
    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            g_loss, d_loss = self.train_epoch(data_loader)
            self.logger.info(f'Epoch {epoch+1}/{epochs}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}')
            
            if epoch % 10 == 0:
                torch.save({
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                    'epoch': epoch
                }, f'models/gan_checkpoint_epoch_{epoch}.pth')