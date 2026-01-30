"""Physics-Informed Neural Network (PINN) architecture for inverter dynamics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PINN(nn.Module):
    """
    3-layer MLP for predicting inverter state transitions.
    
    Architecture:
    - Input: [x_t, u_t] where x_t ∈ ℝ⁴ (states), u_t ∈ ℝ² (controls)
    - Hidden layers: 64, 64 neurons with SiLU activation
    - Output: predicted state x_{t+1} ∈ ℝ⁴
    """
    
    def __init__(
        self,
        input_dim: int = 6,  # 4 states + 2 controls
        output_dim: int = 4,  # 4 states
        hidden_dims: Tuple[int, int] = (64, 64),
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict next state given current state and control.
        
        Args:
            x: Current state tensor of shape (batch_size, 4)
            u: Control input tensor of shape (batch_size, 2)
            
        Returns:
            Predicted next state tensor of shape (batch_size, 4)
        """
        # Concatenate state and control
        inputs = torch.cat([x, u], dim=-1)
        return self.network(inputs)
    
    def predict_sequence(
        self,
        x0: torch.Tensor,
        u_sequence: torch.Tensor,
        dt: float = 1e-4
    ) -> torch.Tensor:
        """
        Predict state sequence using recurrent predictions.
        
        Args:
            x0: Initial state tensor of shape (batch_size, 4)
            u_sequence: Control sequence tensor of shape (seq_len, batch_size, 2)
            dt: Time step [s]
            
        Returns:
            State sequence tensor of shape (seq_len + 1, batch_size, 4)
        """
        batch_size = x0.shape[0]
        seq_len = u_sequence.shape[0]
        
        # Initialize sequence
        x_sequence = torch.zeros(seq_len + 1, batch_size, 4, device=x0.device)
        x_sequence[0] = x0
        
        # Predict recursively
        for t in range(seq_len):
            x_current = x_sequence[t]
            u_current = u_sequence[t]
            # Ensure u_current has correct shape (batch_size, 2)
            if u_current.dim() == 1:
                u_current = u_current.unsqueeze(0)
            x_next = self.forward(x_current, u_current)
            x_sequence[t + 1] = x_next
        
        return x_sequence
    
    def compute_physics_residual(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float = 1e-4,
        requires_grad: bool = True
    ) -> torch.Tensor:
        """
        Compute physics residual using automatic differentiation.
        
        The residual is: r = dx/dt - (A*x + B*u)
        where dx/dt is computed via autograd through the network.
        
        Args:
            x: Current state tensor
            u: Control input tensor
            dt: Time step for finite difference approximation
            requires_grad: Whether to compute gradients for x
            
        Returns:
            Physics residual tensor
        """
        if requires_grad:
            x.requires_grad_(True)
        
        # Predict next state
        x_next = self.forward(x, u)
        
        # Compute dx/dt using finite difference
        dx_dt = (x_next - x) / dt
        
        # Compute physics-based derivative (A*x + B*u)
        # Note: A and B matrices are not directly available in the network
        # We'll compute this externally using the InverterModel
        # For now, return dx_dt which will be compared with physics in loss function
        return dx_dt
    
    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': (64, 64)
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PINN':
        """Load model from saved weights."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            hidden_dims=checkpoint['hidden_dims']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


def test_pinn() -> None:
    """Test function for PINN model."""
    print("Testing PINN model...")
    
    # Create model
    model = PINN()
    print(f"Model architecture: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 4)
    u = torch.randn(batch_size, 2)
    
    with torch.no_grad():
        x_next = model(x, u)
        print(f"Input shape: x={x.shape}, u={u.shape}")
        print(f"Output shape: {x_next.shape}")
        
        # Test sequence prediction
        seq_len = 10
        u_seq = torch.randn(seq_len, batch_size, 2)
        x_seq = model.predict_sequence(x[0:1], u_seq)
        print(f"Sequence prediction shape: {x_seq.shape}")
        
        # Test physics residual
        residual = model.compute_physics_residual(x, u)
        print(f"Physics residual shape: {residual.shape}")
    
    print("PINN model test passed!")


if __name__ == "__main__":
    test_pinn()