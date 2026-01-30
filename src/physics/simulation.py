"""Simulation module for generating synthetic training data using ODE solver."""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Dict, Optional, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dataclasses import dataclass
from .inverter import InverterModel, InverterParameters


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    dt: float = 1e-4  # Time step [s]
    duration: float = 1.0  # Simulation duration [s]
    control_type: str = "random"  # "random", "sinusoidal", "step"
    disturbance_type: Optional[str] = None  # "cloud_event", "grid_fault", None
    disturbance_severity: float = 0.7  # Severity (0-1)
    initial_state: Optional[np.ndarray] = None


class InverterSimulator:
    """Simulator for generating synthetic inverter data."""
    
    def __init__(self, model: Optional[InverterModel] = None):
        self.model = model or InverterModel()
        self.config = SimulationConfig()
    
    def generate_control_input(
        self, 
        t: np.ndarray, 
        control_type: str = "random"
    ) -> np.ndarray:
        """
        Generate control input sequence.
        
        Args:
            t: Time array [s]
            control_type: Type of control input
            
        Returns:
            Control input array with shape (len(t), 2)
        """
        n = len(t)
        u = np.zeros((n, 2))
        
        if control_type == "random":
            # Random walk control inputs
            u[0] = np.random.uniform(-0.5, 0.5, 2)
            for i in range(1, n):
                u[i] = u[i-1] + np.random.normal(0, 0.01, 2)
                # Clip to reasonable bounds
                u[i] = np.clip(u[i], -1.0, 1.0)
        
        elif control_type == "sinusoidal":
            # Sinusoidal control inputs
            freq = 10  # Hz
            u[:, 0] = 0.3 * np.sin(2 * np.pi * freq * t)
            u[:, 1] = 0.2 * np.cos(2 * np.pi * freq * t)
        
        elif control_type == "step":
            # Step changes
            step_times = [0.2, 0.5, 0.8]
            step_values = [(0.2, 0.1), (-0.1, 0.3), (0.3, -0.2)]
            
            current_u = np.array([0.0, 0.0])
            step_idx = 0
            
            for i, time in enumerate(t):
                if step_idx < len(step_times) and time >= step_times[step_idx]:
                    current_u = np.array(step_values[step_idx])
                    step_idx += 1
                u[i] = current_u
        
        return u
    
    def simulate(
        self,
        config: Optional[SimulationConfig] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run simulation and generate synthetic data.
        
        Args:
            config: Simulation configuration
            
        Returns:
            (t, x, u, w): Time, states, controls, disturbances
        """
        if config:
            self.config = config
        
        # Create time array
        n_steps = int(self.config.duration / self.config.dt)
        t = np.linspace(0, self.config.duration, n_steps)
        
        # Generate control inputs
        u = self.generate_control_input(t, self.config.control_type)
        
        # Generate disturbances
        if self.config.disturbance_type:
            w = self.model.create_disturbance_profile(
                t, 
                self.config.disturbance_type,
                self.config.disturbance_severity
            )
        else:
            w = np.zeros((len(t), 3))
        
        # Set initial state
        if self.config.initial_state is None:
            x0 = np.array([0.0, 0.0, self.model.params.V_grid_peak, 0.0])
        else:
            x0 = self.config.initial_state
        
        # Solve ODE using odeint with time-varying inputs
        x = np.zeros((len(t), 4))
        x[0] = x0
        
        # Use simple Euler integration for clarity (odeint doesn't handle time-varying u,w well)
        for i in range(1, len(t)):
            dx = self.model.dynamics(t[i-1], x[i-1], u[i-1], w[i-1])
            x[i] = x[i-1] + dx * self.config.dt
        
        return t, x, u, w
    
    def generate_training_data(
        self,
        n_samples: int = 50000,
        include_disturbances: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate comprehensive training dataset.
        
        Args:
            n_samples: Number of training samples
            include_disturbances: Whether to include disturbance scenarios
            
        Returns:
            Dictionary containing training data
        """
        print(f"Generating {n_samples} training samples...")
        
        # Generate multiple simulation runs
        all_states = []
        all_controls = []
        all_next_states = []
        
        samples_per_run = 1000
        n_runs = n_samples // samples_per_run
        
        for run in range(n_runs):
            # Vary simulation parameters
            control_types = ["random", "sinusoidal", "step"]
            disturbance_types = ["cloud_event", "grid_fault", None]
            
            config = SimulationConfig(
                dt=1e-4,
                duration=0.1,  # 100ms runs
                control_type=np.random.choice(control_types),
                disturbance_type=np.random.choice(disturbance_types) 
                if include_disturbances and run % 3 == 0 else None,
                disturbance_severity=np.random.uniform(0.5, 0.8),
            )
            
            t, x, u, w = self.simulate(config)
            
            # Create state-control-next_state pairs
            for i in range(len(x) - 1):
                all_states.append(x[i])
                all_controls.append(u[i])
                all_next_states.append(x[i+1])
            
            if (run + 1) % 10 == 0:
                print(f"  Completed {run + 1}/{n_runs} runs...")
        
        # Convert to arrays
        states = np.array(all_states)
        controls = np.array(all_controls)
        next_states = np.array(all_next_states)
        
        # Create input features: [state, control]
        inputs = np.hstack([states, controls])
        
        return {
            "inputs": inputs.astype(np.float32),
            "targets": next_states.astype(np.float32),
            "states": states.astype(np.float32),
            "controls": controls.astype(np.float32),
        }
    
    def plot_simulation(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        w: np.ndarray,
        title: str = "Inverter Simulation"
    ) -> Figure:
        """
        Create comprehensive simulation plot.
        
        Args:
            t: Time array
            x: State trajectories
            u: Control inputs
            w: Disturbances
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(4, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14)
        
        # State plots
        state_labels = ["i_d [A]", "i_q [A]", "v_d [V]", "v_q [V]"]
        for i in range(4):
            ax = axes[i, 0]
            ax.plot(t, x[:, i])
            ax.set_ylabel(state_labels[i])
            ax.grid(True, alpha=0.3)
            if i == 3:
                ax.set_xlabel("Time [s]")
        
        # Control and disturbance plots
        axes[0, 1].plot(t, u[:, 0], label="d_d")
        axes[0, 1].plot(t, u[:, 1], label="d_q")
        axes[0, 1].set_ylabel("Duty Cycles")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 1].plot(t, w[:, 0], label="v_grid_d")
        axes[1, 1].plot(t, w[:, 1], label="v_grid_q")
        axes[1, 1].set_ylabel("Grid Voltage [V]")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 1].plot(t, w[:, 2], label="P_dc factor")
        axes[2, 1].set_ylabel("DC Power Factor")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Current magnitude (p.u.)
        current_pu = []
        for state in x:
            current_pu.append(self.model.compute_current_magnitude(state))
        
        axes[3, 1].plot(t, current_pu)
        axes[3, 1].axhline(y=1.2, color='r', linestyle='--', label="Limit (1.2 p.u.)")
        axes[3, 1].set_ylabel("Current [p.u.]")
        axes[3, 1].set_xlabel("Time [s]")
        axes[3, 1].legend()
        axes[3, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def test_simulation() -> None:
    """Test function for simulation module."""
    print("Testing inverter simulation...")
    
    # Create simulator
    simulator = InverterSimulator()
    
    # Test normal operation
    config = SimulationConfig(
        dt=1e-4,
        duration=0.2,
        control_type="sinusoidal",
        disturbance_type=None
    )
    
    t, x, u, w = simulator.simulate(config)
    print(f"Simulation completed: {len(t)} time steps")
    print(f"State shape: {x.shape}")
    print(f"Control shape: {u.shape}")
    
    # Test with disturbance
    config.disturbance_type = "cloud_event"
    t2, x2, u2, w2 = simulator.simulate(config)
    print(f"Disturbance simulation completed: {len(t2)} time steps")
    
    # Generate training data
    data = simulator.generate_training_data(n_samples=1000, include_disturbances=True)
    print(f"Training data shapes:")
    print(f"  Inputs: {data['inputs'].shape}")
    print(f"  Targets: {data['targets'].shape}")
    
    print("Simulation test passed!")


if __name__ == "__main__":
    test_simulation()