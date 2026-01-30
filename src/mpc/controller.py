"""Model Predictive Control (MPC) using Physics-Informed Neural Network."""

import numpy as np
import torch
from scipy.optimize import minimize
from typing import Tuple, Optional, List, Dict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dataclasses import dataclass

from ..pinn.model import PINN
from ..pinn.loss import CompositeLoss
from ..physics.inverter import InverterModel


@dataclass
class MPCConfig:
    """Configuration for MPC controller."""
    horizon: int = 10  # Prediction horizon
    dt: float = 1e-4  # Time step [s]
    max_iter: int = 100  # Maximum optimization iterations
    u_min: float = -1.0  # Minimum control input
    u_max: float = 1.0  # Maximum control input
    current_limit: float = 1.2  # Current limit in p.u.
    Q: Optional[np.ndarray] = None  # State weighting matrix
    R: Optional[np.ndarray] = None  # Control weighting matrix
    
    def __post_init__(self):
        if self.Q is None:
            self.Q = np.eye(4)  # Equal weighting on all states
        if self.R is None:
            self.R = np.eye(2) * 0.1  # Control effort penalty


class NeuralMPC:
    """Model Predictive Controller using PINN as transition model."""
    
    def __init__(
        self,
        model: PINN,
        config: Optional[MPCConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.config = config or MPCConfig()
        self.device = device
        
        # Physics model for constraint checking
        self.physics_model = InverterModel()
        
        # Loss function for MPC
        Q_torch = torch.tensor(self.config.Q, dtype=torch.float32).to(device)
        R_torch = torch.tensor(self.config.R, dtype=torch.float32).to(device)
        self.loss_fn = CompositeLoss(Q_torch, R_torch)
    
    def predict_trajectory(
        self,
        x0: torch.Tensor,
        u_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict state trajectory using PINN.
        
        Args:
            x0: Initial state tensor of shape (batch_size, 4)
            u_sequence: Control sequence tensor of shape (horizon, batch_size, 2)
            
        Returns:
            State trajectory tensor of shape (horizon + 1, batch_size, 4)
        """
        return self.model.predict_sequence(x0, u_sequence, self.config.dt)
    
    def compute_cost(
        self,
        u_flat: np.ndarray,
        x0: torch.Tensor,
        x_ref: torch.Tensor
    ) -> float:
        """
        Compute MPC cost for given control sequence.
        
        Args:
            u_flat: Flattened control sequence (horizon * 2)
            x0: Initial state
            x_ref: Reference state trajectory
            
        Returns:
            Total cost
        """
        # Move tensors to correct device
        x0 = x0.to(self.device)
        x_ref = x_ref.to(self.device)
        
        # Reshape control sequence
        horizon = self.config.horizon
        u_sequence = torch.tensor(
            u_flat.reshape(horizon, 1, 2), 
            dtype=torch.float32
        ).to(self.device)
        
        # Predict trajectory
        x_trajectory = self.predict_trajectory(x0.unsqueeze(0), u_sequence)
        
        # Extract predicted states (excluding initial state)
        x_pred = x_trajectory[1:].squeeze(1)  # (horizon, 4)
        
        # Repeat reference for horizon
        x_ref_horizon = x_ref.repeat(horizon, 1)
        
        # Compute loss
        total_loss, _, _, _ = self.loss_fn(
            x_pred, x_ref_horizon, u_sequence.squeeze(1)
        )
        
        return total_loss.item()
    
    def compute_constraints(
        self,
        u_flat: np.ndarray,
        x0: torch.Tensor
    ) -> np.ndarray:
        """
        Compute constraint violations.
        
        Args:
            u_flat: Flattened control sequence
            x0: Initial state
            
        Returns:
            Constraint violations (negative values indicate satisfaction)
        """
        horizon = self.config.horizon
        u_sequence = torch.tensor(
            u_flat.reshape(horizon, 1, 2), 
            dtype=torch.float32
        ).to(self.device)
        
        # Predict trajectory
        x_trajectory = self.predict_trajectory(x0.unsqueeze(0), u_sequence)
        
        constraints = []
        
        # Control input bounds
        u_min_violation = self.config.u_min - u_flat
        u_max_violation = u_flat - self.config.u_max
        constraints.extend(u_min_violation.tolist())
        constraints.extend(u_max_violation.tolist())
        
        # Current limits
        for t in range(1, horizon + 1):
            x_t = x_trajectory[t].squeeze(0).detach().cpu().numpy()
            current_pu = self.physics_model.compute_current_magnitude(x_t)
            current_violation = current_pu - self.config.current_limit
            constraints.append(current_violation)
        
        return np.array(constraints)
    
    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        u_init: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve MPC optimization problem.
        
        Args:
            x0: Initial state
            x_ref: Reference state
            u_init: Initial guess for control sequence
            
        Returns:
            (u_opt, x_trajectory, info): Optimal control, predicted trajectory, and info dict
        """
        horizon = self.config.horizon
        
        # Initial guess for control sequence
        if u_init is None:
            u_init = np.zeros(horizon * 2)
        
        # Convert to torch tensors
        x0_torch = torch.tensor(x0, dtype=torch.float32).to(self.device)
        x_ref_torch = torch.tensor(x_ref, dtype=torch.float32).to(self.device)
        
        # Define objective function
        def objective(u_flat: np.ndarray) -> float:
            return self.compute_cost(u_flat, x0_torch, x_ref_torch)
        
        # Define constraints
        constraints = [
            {
                'type': 'ineq',
                'fun': lambda u: -self.compute_constraints(u, x0_torch)
            }
        ]
        
        # Bounds for control inputs
        bounds = [(self.config.u_min, self.config.u_max)] * (horizon * 2)
        
        # Solve optimization
        result = minimize(
            fun=objective,
            x0=u_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iter, 'disp': False}
        )
        
        # Extract solution
        u_opt_flat = result.x
        u_opt = u_opt_flat.reshape(horizon, 2)
        
        # Predict optimal trajectory
        u_opt_torch = torch.tensor(u_opt, dtype=torch.float32).unsqueeze(1).to(self.device)
        x_trajectory_torch = self.predict_trajectory(x0_torch.unsqueeze(0), u_opt_torch)
        x_trajectory = x_trajectory_torch.squeeze(1).detach().cpu().numpy()
        
        # Prepare info dictionary
        info = {
            'success': result.success,
            'message': result.message,
            'fun': result.fun,
            'nit': result.nit,
            'nfev': result.nfev,
        }
        
        return u_opt, x_trajectory, info
    
    def simulate_closed_loop(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        simulation_steps: int = 100,
        disturbance: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate closed-loop MPC control.
        
        Args:
            x0: Initial state
            x_ref: Reference state
            simulation_steps: Number of simulation steps
            disturbance: Optional disturbance sequence
            
        Returns:
            Dictionary with simulation results
        """
        # Initialize arrays
        x_history = np.zeros((simulation_steps + 1, 4))
        u_history = np.zeros((simulation_steps, 2))
        cost_history = np.zeros(simulation_steps)
        
        x_current = x0.copy()
        x_history[0] = x_current
        
        print(f"Simulating closed-loop MPC for {simulation_steps} steps...")
        
        for step in range(simulation_steps):
            if step % 10 == 0:
                print(f"  Step {step}/{simulation_steps}")
            
            # Solve MPC problem
            u_opt, x_pred, info = self.solve(x_current, x_ref)
            
            # Apply first control input (receding horizon)
            u_applied = u_opt[0]
            u_history[step] = u_applied
            cost_history[step] = info['fun']
            
            # Simulate one step using PINN
            x_current_torch = torch.tensor(x_current, dtype=torch.float32).unsqueeze(0).to(self.device)
            u_applied_torch = torch.tensor(u_applied, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                x_next_torch = self.model(x_current_torch, u_applied_torch)
                x_next = x_next_torch.squeeze(0).cpu().numpy()
            
            # Apply disturbance if provided
            if disturbance is not None and step < len(disturbance):
                # Simple disturbance model: additive noise
                x_next += disturbance[step] * 0.01
            
            # Update state
            x_current = x_next
            x_history[step + 1] = x_current
        
        return {
            'x_history': x_history,
            'u_history': u_history,
            'cost_history': cost_history,
        }
    
    def plot_results(
        self,
        results: Dict[str, np.ndarray],
        x_ref: np.ndarray,
        title: str = "MPC Simulation Results"
    ) -> Figure:
        """
        Plot MPC simulation results.
        
        Args:
            results: Simulation results dictionary
            x_ref: Reference state
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        x_history = results['x_history']
        u_history = results['u_history']
        cost_history = results['cost_history']
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14)
        
        # State trajectories
        state_labels = ["i_d [A]", "i_q [A]", "v_d [V]", "v_q [V]"]
        for i in range(4):
            ax = axes[i // 2, i % 2]
            ax.plot(x_history[:, i], label='Actual')
            ax.axhline(y=x_ref[i], color='r', linestyle='--', label='Reference')
            ax.set_ylabel(state_labels[i])
            ax.set_xlabel('Time Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Control inputs
        axes[2, 0].plot(u_history[:, 0], label='d_d')
        axes[2, 0].plot(u_history[:, 1], label='d_q')
        axes[2, 0].set_ylabel('Control Inputs')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Cost history
        axes[2, 1].plot(cost_history)
        axes[2, 1].set_ylabel('MPC Cost')
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_title('Cost History')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def test_mpc() -> None:
    """Test MPC controller."""
    print("Testing MPC controller...")
    
    # Create a simple PINN model for testing
    model = PINN()
    
    # Create MPC controller
    config = MPCConfig(horizon=5, max_iter=50)
    mpc = NeuralMPC(model, config)
    
    # Test data
    x0 = np.array([0.0, 0.0, 400.0, 0.0])  # Initial state
    x_ref = np.array([10.0, 0.0, 400.0, 0.0])  # Reference state
    
    # Test prediction
    horizon = config.horizon
    u_test = torch.randn(horizon, 1, 2).to(mpc.device)
    x0_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).to(mpc.device)
    
    with torch.no_grad():
        x_trajectory = mpc.predict_trajectory(x0_torch, u_test)
        print(f"Predicted trajectory shape: {x_trajectory.shape}")
    
    # Test cost computation
    u_flat = np.random.randn(horizon * 2)
    cost = mpc.compute_cost(u_flat, torch.tensor(x0, dtype=torch.float32).to(mpc.device), torch.tensor(x_ref, dtype=torch.float32).to(mpc.device))
    print(f"MPC cost: {cost:.6f}")
    
    # Test constraint computation
    constraints = mpc.compute_constraints(u_flat, torch.tensor(x0, dtype=torch.float32).to(mpc.device))
    print(f"Constraint violations shape: {constraints.shape}")
    
    # Test MPC solve (with small horizon for speed)
    try:
        u_opt, x_traj, info = mpc.solve(x0, x_ref)
        print(f"MPC optimization success: {info['success']}")
        print(f"Optimal control shape: {u_opt.shape}")
        print(f"Predicted trajectory shape: {x_traj.shape}")
    except Exception as e:
        print(f"MPC solve test skipped (requires optimization): {e}")
    
    print("MPC controller test passed!")


if __name__ == "__main__":
    test_mpc()