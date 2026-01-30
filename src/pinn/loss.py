"""Physics-informed loss functions for PINN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from ..physics.inverter import InverterModel


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss function for PINN training.
    
    Loss = L_data + λ * L_physics
    
    where:
    - L_data: MSE between predicted and ground truth states
    - L_physics: Residual of the ODE (penalizes violation of electrical laws)
    """
    
    def __init__(
        self,
        physics_weight: float = 10.0,
        data_weight: float = 1.0,
        inverter_model: Optional[InverterModel] = None
    ):
        super().__init__()
        self.physics_weight = physics_weight
        self.data_weight = data_weight
        self.inverter_model = inverter_model or InverterModel()
        
        # Convert numpy matrices to torch tensors
        self.A = torch.tensor(self.inverter_model.A, dtype=torch.float32)
        self.B = torch.tensor(self.inverter_model.B, dtype=torch.float32)
        self.dt = 1e-4  # Time step for finite differences
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        x_current: torch.Tensor,
        u: torch.Tensor,
        w: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined physics-informed loss.
        
        Args:
            x_pred: Predicted next state from network
            x_true: Ground truth next state
            x_current: Current state (for physics residual)
            u: Control input
            w: Disturbance vector (optional)
            
        Returns:
            (total_loss, data_loss, physics_loss)
        """
        # Data loss: MSE between prediction and ground truth
        data_loss = F.mse_loss(x_pred, x_true)
        
        # Physics loss: ODE residual
        physics_loss = self._compute_physics_loss(x_pred, x_current, u, w)
        
        # Combined loss
        total_loss = self.data_weight * data_loss + self.physics_weight * physics_loss
        
        return total_loss, data_loss, physics_loss
    
    def _compute_physics_loss(
        self,
        x_next: torch.Tensor,
        x_current: torch.Tensor,
        u: torch.Tensor,
        w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics loss as ODE residual.
        
        The ODE is: dx/dt = A*x + B*u + D*w
        We compute residual: r = (x_next - x_current)/dt - (A*x_current + B*u + D*w)
        
        Args:
            x_next: Predicted next state
            x_current: Current state
            u: Control input
            w: Disturbance vector
            
        Returns:
            Physics loss (MSE of residual)
        """
        # Move tensors to same device as inputs
        device = x_current.device
        A = self.A.to(device)
        B = self.B.to(device)
        
        # Compute finite difference derivative
        dx_dt_pred = (x_next - x_current) / self.dt
        
        # Compute physics-based derivative: A*x + B*u
        dx_dt_physics = torch.matmul(x_current, A.T) + torch.matmul(u, B.T)
        
        # Add disturbance effect if provided
        if w is not None:
            D = torch.tensor(self.inverter_model.D, dtype=torch.float32).to(device)
            dx_dt_physics = dx_dt_physics + torch.matmul(w, D.T)
        
        # Compute residual
        residual = dx_dt_pred - dx_dt_physics
        
        # Physics loss: MSE of residual
        physics_loss = torch.mean(residual ** 2)
        
        return physics_loss
    
    def compute_constraint_violation(
        self,
        x: torch.Tensor,
        current_limit: float = 1.2
    ) -> torch.Tensor:
        """
        Compute constraint violation penalty for current limits.
        
        Args:
            x: State tensor
            current_limit: Current limit in p.u.
            
        Returns:
            Constraint violation penalty
        """
        # Extract currents (first two states)
        i_d = x[:, 0]
        i_q = x[:, 1]
        
        # Compute current magnitude
        i_mag = torch.sqrt(i_d**2 + i_q**2)
        
        # Base current for 10kW system
        i_base = 10000 / (np.sqrt(3) * self.inverter_model.params.V_grid)
        i_mag_pu = i_mag / i_base
        
        # Penalty for exceeding limit
        violation = torch.relu(i_mag_pu - current_limit)
        penalty = torch.mean(violation ** 2)
        
        return penalty
    
    def compute_autograd_physics_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        u: torch.Tensor,
        requires_grad: bool = True
    ) -> torch.Tensor:
        """
        Compute physics loss using automatic differentiation through the network.
        
        This method uses torch.autograd to compute dx/dt directly from the network.
        
        Args:
            model: PINN model
            x: Current state (requires_grad=True)
            u: Control input
            requires_grad: Whether to enable gradient computation
            
        Returns:
            Physics loss computed via autograd
        """
        if requires_grad:
            x.requires_grad_(True)
        
        # Forward pass through network
        x_next = model(x, u)
        
        # Compute dx/dt using finite difference
        dx_dt = (x_next - x) / self.dt
        
        # Compute physics-based derivative
        device = x.device
        A = self.A.to(device)
        B = self.B.to(device)
        dx_dt_physics = torch.matmul(x, A.T) + torch.matmul(u, B.T)
        
        # Compute residual
        residual = dx_dt - dx_dt_physics
        
        # Return MSE of residual
        return torch.mean(residual ** 2)


class CompositeLoss(nn.Module):
    """
    Composite loss with multiple components for MPC training.
    
    Includes:
    - Tracking error (state deviation from reference)
    - Control effort
    - Constraint violations
    - Physics consistency
    """
    
    def __init__(
        self,
        Q: torch.Tensor,  # State weighting matrix
        R: torch.Tensor,  # Control weighting matrix
        physics_weight: float = 1.0,
        constraint_weight: float = 10.0
    ):
        super().__init__()
        self.Q = Q
        self.R = R
        self.physics_weight = physics_weight
        self.constraint_weight = constraint_weight
        
        # Physics loss helper
        self.physics_loss = PhysicsInformedLoss()
    
    def forward(
        self,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        u: torch.Tensor,
        physics_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute composite MPC loss.
        
        Args:
            x: Predicted states
            x_ref: Reference states
            u: Control inputs
            physics_loss: Pre-computed physics loss (optional)
            
        Returns:
            (total_loss, tracking_loss, control_loss, constraint_loss)
        """
        # Tracking error: (x - x_ref)^T Q (x - x_ref)
        tracking_error = x - x_ref
        tracking_loss = torch.mean(torch.sum(tracking_error * (tracking_error @ self.Q.T), dim=-1))
        
        # Control effort: u^T R u
        control_loss = torch.mean(torch.sum(u * (u @ self.R.T), dim=-1))
        
        # Constraint violations (current limits)
        constraint_loss = self.physics_loss.compute_constraint_violation(x)
        
        # Physics loss (if not provided, compute minimal)
        if physics_loss is None:
            # For MPC, we might not have ground truth for physics loss
            physics_loss_value = torch.tensor(0.0, device=x.device)
        else:
            physics_loss_value = physics_loss
        
        # Total loss
        total_loss = (
            tracking_loss +
            control_loss +
            self.constraint_weight * constraint_loss +
            self.physics_weight * physics_loss_value
        )
        
        return total_loss, tracking_loss, control_loss, constraint_loss


def test_loss_functions() -> None:
    """Test loss functions."""
    print("Testing loss functions...")
    
    # Create test data
    batch_size = 16
    x_pred = torch.randn(batch_size, 4)
    x_true = torch.randn(batch_size, 4)
    x_current = torch.randn(batch_size, 4)
    u = torch.randn(batch_size, 2)
    
    # Test PhysicsInformedLoss
    physics_loss_fn = PhysicsInformedLoss(physics_weight=10.0)
    total_loss, data_loss, physics_loss = physics_loss_fn(
        x_pred, x_true, x_current, u
    )
    
    print(f"Physics-informed loss:")
    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  Data loss: {data_loss.item():.6f}")
    print(f"  Physics loss: {physics_loss.item():.6f}")
    
    # Test constraint violation
    constraint_penalty = physics_loss_fn.compute_constraint_violation(x_pred)
    print(f"  Constraint penalty: {constraint_penalty.item():.6f}")
    
    # Test CompositeLoss
    Q = torch.eye(4)
    R = torch.eye(2) * 0.1
    composite_loss_fn = CompositeLoss(Q, R)
    
    x_ref = torch.zeros(batch_size, 4)
    total_loss, tracking, control, constraint = composite_loss_fn(
        x_pred, x_ref, u
    )
    
    print(f"\nComposite MPC loss:")
    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  Tracking loss: {tracking.item():.6f}")
    print(f"  Control loss: {control.item():.6f}")
    print(f"  Constraint loss: {constraint.item():.6f}")
    
    print("Loss functions test passed!")


if __name__ == "__main__":
    test_loss_functions()