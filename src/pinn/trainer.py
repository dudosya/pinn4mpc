"""Training pipeline for Physics-Informed Neural Network."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
import os
import json

from .model import PINN
from .loss import PhysicsInformedLoss
from ..physics.simulation import InverterSimulator


class PINNTrainer:
    """Trainer for Physics-Informed Neural Network."""
    
    def __init__(
        self,
        model: PINN,
        loss_fn: PhysicsInformedLoss,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_data_loss': [],
            'train_physics_loss': [],
            'val_loss': [],
            'val_data_loss': [],
            'val_physics_loss': [],
            'learning_rate': []
        }
    
    def prepare_data(
        self,
        data: Dict[str, np.ndarray],
        val_split: float = 0.2,
        batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training.
        
        Args:
            data: Dictionary with 'inputs' and 'targets' arrays
            val_split: Validation split ratio
            batch_size: Batch size for training
            
        Returns:
            (train_loader, val_loader)
        """
        # Split inputs into states and controls
        inputs = data['inputs']
        targets = data['targets']
        
        # States are first 4 columns, controls are last 2 columns
        states = inputs[:, :4]
        controls = inputs[:, 4:]
        
        # Split into train/validation
        n_samples = len(inputs)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(states[train_idx], dtype=torch.float32),
            torch.tensor(controls[train_idx], dtype=torch.float32),
            torch.tensor(targets[train_idx], dtype=torch.float32)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(states[val_idx], dtype=torch.float32),
            torch.tensor(controls[val_idx], dtype=torch.float32),
            torch.tensor(targets[val_idx], dtype=torch.float32)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        print(f"Training samples: {n_train:,}")
        print(f"Validation samples: {n_val:,}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            (avg_loss, avg_data_loss, avg_physics_loss)
        """
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        
        for x_current, u, x_true in tqdm(train_loader, desc="Training"):
            # Move to device
            x_current = x_current.to(self.device)
            u = u.to(self.device)
            x_true = x_true.to(self.device)
            
            # Forward pass
            x_pred = self.model(x_current, u)
            
            # Compute loss
            loss, data_loss, physics_loss = self.loss_fn(
                x_pred, x_true, x_current, u
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
            n_batches += 1
        
        # Compute averages
        avg_loss = total_loss / n_batches
        avg_data_loss = total_data_loss / n_batches
        avg_physics_loss = total_physics_loss / n_batches
        
        return avg_loss, avg_data_loss, avg_physics_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            (avg_loss, avg_data_loss, avg_physics_loss)
        """
        self.model.eval()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for x_current, u, x_true in tqdm(val_loader, desc="Validation"):
                # Move to device
                x_current = x_current.to(self.device)
                u = u.to(self.device)
                x_true = x_true.to(self.device)
                
                # Forward pass
                x_pred = self.model(x_current, u)
                
                # Compute loss
                loss, data_loss, physics_loss = self.loss_fn(
                    x_pred, x_true, x_current, u
                )
                
                # Accumulate losses
                total_loss += loss.item()
                total_data_loss += data_loss.item()
                total_physics_loss += physics_loss.item()
                n_batches += 1
        
        # Compute averages
        avg_loss = total_loss / n_batches
        avg_data_loss = total_data_loss / n_batches
        avg_physics_loss = total_physics_loss / n_batches
        
        return avg_loss, avg_data_loss, avg_physics_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 20,
        save_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_data_loss, train_physics_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_data_loss, val_physics_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_data_loss'].append(train_data_loss)
            self.history['train_physics_loss'].append(train_physics_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_data_loss'].append(val_data_loss)
            self.history['val_physics_loss'].append(val_physics_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            print(f"Train Loss: {train_loss:.6f} (Data: {train_data_loss:.6f}, Physics: {train_physics_loss:.6f})")
            print(f"Val Loss: {val_loss:.6f} (Data: {val_data_loss:.6f}, Physics: {val_physics_loss:.6f})")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
                print(f"Saved best model with val loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))
        
        # Load best model
        self.load_checkpoint(os.path.join(save_dir, "best_model.pt"))
        
        # Save final model and history
        self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))
        self.save_history(os.path.join(save_dir, "training_history.json"))
        
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
        
        return self.history
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'loss_fn_params': {
                'physics_weight': self.loss_fn.physics_weight,
                'data_weight': self.loss_fn.data_weight
            }
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
    
    def save_history(self, path: str) -> None:
        """Save training history to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) for v in value]
            else:
                history_serializable[key] = float(value)
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def plot_training_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Data loss
        axes[0, 1].plot(self.history['train_data_loss'], label='Train')
        axes[0, 1].plot(self.history['val_data_loss'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Data Loss')
        axes[0, 1].set_title('Data Loss (MSE)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Physics loss
        axes[1, 0].plot(self.history['train_physics_loss'], label='Train')
        axes[1, 0].plot(self.history['val_physics_loss'], label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Physics Loss')
        axes[1, 0].set_title('Physics Loss (ODE Residual)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def train_pinn_from_scratch(
    n_samples: int = 50000,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    physics_weight: float = 10.0,
    save_dir: str = "checkpoints"
) -> PINNTrainer:
    """
    Complete training pipeline from data generation to model training.
    
    Args:
        n_samples: Number of training samples
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        physics_weight: Weight for physics loss
        save_dir: Directory to save checkpoints
        
    Returns:
        Trained trainer object
    """
    print("=" * 60)
    print("PINN Training Pipeline")
    print("=" * 60)
    
    # Step 1: Generate training data
    print("\n1. Generating training data...")
    simulator = InverterSimulator()
    data = simulator.generate_training_data(
        n_samples=n_samples,
        include_disturbances=True
    )
    
    # Step 2: Create model and loss
    print("\n2. Creating model and loss function...")
    model = PINN()
    loss_fn = PhysicsInformedLoss(physics_weight=physics_weight)
    
    # Step 3: Create trainer
    print("\n3. Setting up trainer...")
    trainer = PINNTrainer(
        model=model,
        loss_fn=loss_fn,
        learning_rate=learning_rate
    )
    
    # Step 4: Prepare data loaders
    print("\n4. Preparing data loaders...")
    train_loader, val_loader = trainer.prepare_data(
        data=data,
        val_split=0.2,
        batch_size=batch_size
    )
    
    # Step 5: Train model
    print("\n5. Training model...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=save_dir
    )
    
    # Step 6: Plot results
    print("\n6. Plotting training history...")
    trainer.plot_training_history(os.path.join(save_dir, "training_history.png"))
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return trainer


def test_training() -> None:
    """Test training pipeline with small dataset."""
    print("Testing training pipeline...")
    
    # Use small dataset for quick test
    trainer = train_pinn_from_scratch(
        n_samples=1000,
        epochs=5,
        batch_size=32,
        save_dir="test_checkpoints"
    )
    
    print("Training pipeline test passed!")


if __name__ == "__main__":
    test_training()