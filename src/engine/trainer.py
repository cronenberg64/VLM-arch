import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import time
from typing import Dict, Any
from omegaconf import DictConfig
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: DictConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler('cuda', enabled=cfg.training.amp)
        
        self.epochs = cfg.training.epochs
        self.accumulate_grad_batches = cfg.training.get('accumulate_grad_batches', 1)
        self.grad_clip = cfg.training.get('grad_clip', 0.0)

    def _setup_optimizer(self):
        opt_name = self.cfg.training.optimizer.lower()
        lr = self.cfg.training.lr
        weight_decay = self.cfg.training.weight_decay
        
        if opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _setup_scheduler(self):
        # Simple Cosine Annealing for now
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.training.epochs)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        self.optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            with autocast('cuda', enabled=self.cfg.training.amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulate_grad_batches

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accumulate_grad_batches == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulate_grad_batches
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss / (i + 1), 'acc': 100. * correct / total})

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with autocast('cuda', enabled=self.cfg.training.amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / len(self.val_loader), 100. * correct / total

    def fit(self):
        best_acc = 0
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
            
            if wandb.run is not None:
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": self.scheduler.get_last_lr()[0],
                    "epoch": epoch
                })
            
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
                print(f"New best model saved with accuracy: {val_acc:.2f}%")

    def save_checkpoint(self, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.cfg
        }
        # Save to current working directory (Hydra manages this)
        save_path = os.path.join(os.getcwd(), "best_model.pth")
        try:
            torch.save(checkpoint, save_path)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint to {save_path}: {e}")
        
        # if wandb.run is not None:
        #     # Use policy="now" to upload immediately and avoid symlink issues on Windows
        #     wandb.save(save_path, policy="now")
