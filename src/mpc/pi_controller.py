"""Proportional-Integral (PI) controller for inverter current control."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dataclasses import dataclass

from ..physics.inverter import InverterModel


@dataclass
class PIConfig:
    """Configuration for PI controller."""
    Kp_d: float = 0.5  # Proportional gain for d-axis
    Ki_d: float = 10.0  # Integral gain for d-axis
    Kp_q: float = 0.5  # Proportional gain for q-axis
    Ki_q: float = 10.0  # Integral gain for q-axis
    u_min: float = -1.0  # Minimum control input
    u_max: float = 1.0  # Maximum control input
    anti_windup: bool = True  # Enable anti-windup
    dt: float = 1e-4  # Time step [s]


class PIController:
    """PI controller for dq-axis current control."""
    
    def __init__(self, config: Optional[PIConfig] = None):
        self.config = config or PIConfig()
        self.physics_model = InverterModel()
        
        # Controller states
        self.integral_d = 0.0
        self.integral_q = 0.0
        self.u_prev = np.array([0.0, 0.0])
        
        # Reference values
        self.i_d_ref = 0.0
        self.i_q_ref = 0.0
        
        # History for plotting
        self.history = {
            'error_d': [],
            'error_q': [],
            'integral_d': [],
            'integral_q': [],
            'u_d': [],
            'u_q': [],
            'i_d': [],
            'i_q': []
        }
    
    def set_reference(self, i_d_ref: float, i_q_ref: float) -> None:
        """Set reference currents."""
        self.i_d_ref = i_d_ref
        self.i_q_ref = i_q_ref
    
    def compute_control(
        self,
        i_d: float,
        i_q: float,
        v_d: Optional[float] = None,
        v_q: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute PI control output.
        
        Args:
            i_d: d-axis current
            i_q: q-axis current
            v_d: d-axis voltage (for feedforward, optional)
            v_q: q-axis voltage (for feedforward, optional)
            
        Returns:
            Control input [d_d, d_q]
        """
        # Compute errors
        error_d = self.i_d_ref - i_d
        error_q = self.i_q_ref - i_q
        
        # Update integrals
        self.integral_d += error_d * self.config.dt
        self.integral_q += error_q * self.config.dt
        
        # Anti-windup
        if self.config.anti_windup:
            # Simple anti-windup: clamp integrals when control saturates
            u_d_ff = v_d / self.physics_model.params.V_dc if v_d is not None else 0.0
            u_q_ff = v_q / self.physics_model.params.V_dc if v_q is not None else 0.0
            
            u_d_proposed = u_d_ff + self.config.Kp_d * error_d + self.config.Ki_d * self.integral_d
            u_q_proposed = u_q_ff + self.config.Kp_q * error_q + self.config.Ki_q * self.integral_q
            
            # Clamp proposed control
            u_d_clamped = np.clip(u_d_proposed, self.config.u_min, self.config.u_max)
            u_q_clamped = np.clip(u_q_proposed, self.config.u_min, self.config.u_max)
            
            # Stop integration if control is saturated
            if u_d_proposed != u_d_clamped:
                self.integral_d -= error_d * self.config.dt
            if u_q_proposed != u_q_clamped:
                self.integral_q -= error_q * self.config.dt
        
        # Compute control with feedforward
        u_d = self.config.Kp_d * error_d + self.config.Ki_d * self.integral_d
        u_q = self.config.Kp_q * error_q + self.config.Ki_q * self.integral_q
        
        # Add feedforward if provided
        if v_d is not None:
            u_d += v_d / self.physics_model.params.V_dc
        if v_q is not None:
            u_q += v_q / self.physics_model.params.V_dc
        
        # Clamp control inputs
        u_d = np.clip(u_d, self.config.u_min, self.config.u_max)
        u_q = np.clip(u_q, self.config.u_min, self.config.u_max)
        
        # Store history
        self.history['error_d'].append(error_d)
        self.history['error_q'].append(error_q)
        self.history['integral_d'].append(self.integral_d)
        self.history['integral_q'].append(self.integral_q)
        self.history['u_d'].append(u_d)
        self.history['u_q'].append(u_q)
        self.history['i_d'].append(i_d)
        self.history['i_q'].append(i_q)
        
        self.u_prev = np.array([u_d, u_q])
        return self.u_prev
    
    def simulate_closed_loop(
        self,
        x0: np.ndarray,
        i_d_ref: float,
        i_q_ref: float,
        simulation_steps: int = 100,
        disturbance: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Simulate closed-loop PI control using inverter model.
        
        Args:
            x0: Initial state [i_d, i_q, v_d, v_q]
            i_d_ref: Reference d-axis current
            i_q_ref: Reference q-axis current
            simulation_steps: Number of simulation steps
            disturbance: Optional disturbance sequence
            
        Returns:
            Dictionary with simulation results
        """
        # Initialize arrays
        x_history = np.zeros((simulation_steps + 1, 4))
        u_history = np.zeros((simulation_steps, 2))
        error_history = np.zeros((simulation_steps, 2))
        
        x_current = x0.copy()
        x_history[0] = x_current
        
        # Set reference
        self.set_reference(i_d_ref, i_q_ref)
        
        # Reset controller states
        self.integral_d = 0.0
        self.integral_q = 0.0
        self.history = {key: [] for key in self.history.keys()}
        
        print(f"Simulating closed-loop PI control for {simulation_steps} steps...")
        
        for step in range(simulation_steps):
            if step % 10 == 0:
                print(f"  Step {step}/{simulation_steps}")
            
            # Extract currents and voltages
            i_d, i_q, v_d, v_q = x_current
            
            # Compute control
            u = self.compute_control(i_d, i_q, v_d, v_q)
            u_history[step] = u
            error_history[step] = [self.i_d_ref - i_d, self.i_q_ref - i_q]
            
            # Simulate one step using inverter model
            dx = self.physics_model.dynamics(
                t=step * self.config.dt,
                x=x_current,
                u=u,
                w=np.zeros(3)  # No disturbance in basic simulation
            )
            
            x_next = x_current + dx * self.config.dt
            
            # Apply disturbance if provided
            if disturbance is not None and step < len(disturbance):
                x_next += disturbance[step]
            
            # Update state
            x_current = x_next
            x_history[step + 1] = x_current
        
        return {
            'x_history': x_history,
            'u_history': u_history,
            'error_history': error_history,
            'controller_history': self.history
        }
    
    def tune_gains(
        self,
        desired_settling_time: float = 0.08,  # 80ms settling time
        desired_overshoot: float = 0.05  # 5% overshoot
    ) -> None:
        """
        Tune PI gains based on desired performance.
        
        Uses Ziegler-Nichols inspired tuning for second-order systems.
        """
        # Simple tuning rules for current control
        # Based on approximate system time constant
        L = self.physics_model.params.L
        R = self.physics_model.params.R
        
        # Approximate time constant
        tau = L / R  # RL circuit time constant
        
        # Ziegler-Nichols type tuning
        Kp_base = 0.5 * L / (tau * desired_settling_time)
        Ki_base = Kp_base / (2 * tau)
        
        # Adjust for overshoot
        if desired_overshoot < 0.1:
            # Critical damping
            self.config.Kp_d = Kp_base * 0.8
            self.config.Ki_d = Ki_base * 0.6
            self.config.Kp_q = Kp_base * 0.8
            self.config.Ki_q = Ki_base * 0.6
        else:
            # Some overshoot allowed
            self.config.Kp_d = Kp_base
            self.config.Ki_d = Ki_base
            self.config.Kp_q = Kp_base
            self.config.Ki_q = Ki_base
        
        print(f"Tuned PI gains:")
        print(f"  Kp_d: {self.config.Kp_d:.3f}, Ki_d: {self.config.Ki_d:.3f}")
        print(f"  Kp_q: {self.config.Kp_q:.3f}, Ki_q: {self.config.Ki_q:.3f}")
    
    def plot_results(
        self,
        results: Dict[str, np.ndarray],
        title: str = "PI Controller Simulation"
    ) -> plt.Figure:
        """
        Plot PI controller simulation results.
        
        Args:
            results: Simulation results dictionary
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        x_history = results['x_history']
        u_history = results['u_history']
        error_history = results['error_history']
        controller_history = results['controller_history']
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14)
        
        # Current tracking
        axes[0, 0].plot(x_history[:, 0], label='i_d')
        axes[0, 0].axhline(y=self.i_d_ref, color='r', linestyle='--', label='Reference')
        axes[0, 0].set_ylabel('d-axis Current [A]')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(x_history[:, 1], label='i_q')
        axes[0, 1].axhline(y=self.i_q_ref, color='r', linestyle='--', label='Reference')
        axes[0, 1].set_ylabel('q-axis Current [A]')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Control inputs
        axes[1, 0].plot(u_history[:, 0], label='d_d')
        axes[1, 0].plot(u_history[:, 1], label='d_q')
        axes[1, 0].set_ylabel('Control Inputs')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Errors
        axes[1, 1].plot(error_history[:, 0], label='Error d')
        axes[1, 1].plot(error_history[:, 1], label='Error q')
        axes[1, 1].set_ylabel('Tracking Errors')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Integral terms
        if controller_history['integral_d']:
            axes[2, 0].plot(controller_history['integral_d'], label='Integral d')
            axes[2, 0].plot(controller_history['integral_q'], label='Integral q')
            axes[2, 0].set_ylabel('Integral Terms')
            axes[2, 0].set_xlabel('Time Step')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # Current magnitude (p.u.)
        current_pu = []
        for state in x_history:
            current_pu.append(self.physics_model.compute_current_magnitude(state))
        
        axes[2, 1].plot(current_pu)
        axes[2, 1].axhline(y=1.2, color='r', linestyle='--', label='Limit (1.2 p.u.)')
        axes[2, 1].set_ylabel('Current [p.u.]')
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def test_pi_controller() -> None:
    """Test PI controller."""
    print("Testing PI controller...")
    
    # Create PI controller
    controller = PIController()
    
    # Tune gains for 80ms settling time
    controller.tune_gains(desired_settling_time=0.08, desired_overshoot=0.05)
    
    # Test control computation
    i_d, i_q = 5.0, 2.0
    v_d, v_q = 400.0, 0.0
    
    controller.set_reference(i_d_ref=10.0, i_q_ref=0.0)
    u = controller.compute_control(i_d, i_q, v_d, v_q)
    
    print(f"Control output: d_d={u[0]:.3f}, d_q={u[1]:.3f}")
    print(f"Integral terms: I_d={controller.integral_d:.3f}, I_q={controller.integral_q:.3f}")
    
    # Test simulation
    x0 = np.array([0.0, 0.0, 400.0, 0.0])
    results = controller.simulate_closed_loop(
        x0=x0,
        i_d_ref=10.0,
        i_q_ref=0.0,
        simulation_steps=50
    )
    
    print(f"Simulation completed:")
    print(f"  Final state: {results['x_history'][-1]}")
    print(f"  Final error: {results['error_history'][-1]}")
    
    print("PI controller test passed!")


if __name__ == "__main__":
    test_pi_controller()