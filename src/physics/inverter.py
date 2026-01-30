"""State-space model of a three-phase grid-connected inverter with LC filter."""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class InverterParameters:
    """Physical parameters of the grid-connected inverter."""
    L: float = 2.3e-3  # Filter inductance [H]
    C: float = 10e-6  # Filter capacitance [F]
    R: float = 0.1  # Filter resistance [Ω]
    V_dc: float = 800.0  # DC link voltage [V]
    f_sw: float = 10e3  # Switching frequency [Hz]
    f_grid: float = 50.0  # Grid frequency [Hz]
    V_grid: float = 400.0  # Grid line-to-line voltage [V] (RMS)
    
    @property
    def omega_grid(self) -> float:
        """Grid angular frequency [rad/s]."""
        return 2 * np.pi * self.f_grid
    
    @property
    def V_grid_peak(self) -> float:
        """Grid phase voltage peak [V]."""
        return self.V_grid * np.sqrt(2) / np.sqrt(3)


class InverterModel:
    """State-space representation of inverter dynamics in dq-frame."""
    
    def __init__(self, params: Optional[InverterParameters] = None):
        self.params = params or InverterParameters()
        self._setup_state_matrices()
    
    def _setup_state_matrices(self) -> None:
        """Compute the A, B, D matrices for state-space representation."""
        L = self.params.L
        C = self.params.C
        R = self.params.R
        omega = self.params.omega_grid
        
        # State vector: x = [i_d, i_q, v_d, v_q]^T
        # Control input: u = [d_d, d_q]^T (duty cycles in dq-frame)
        # Disturbance: w = [v_grid_d, v_grid_q, P_dc]^T
        
        # A matrix (4x4): dx/dt = A*x
        self.A = np.array([
            [-R/L, omega, -1/L, 0],
            [-omega, -R/L, 0, -1/L],
            [1/C, 0, 0, omega],
            [0, 1/C, -omega, 0],
        ])
        
        # B matrix (4x2): control input effect
        self.B = np.array([
            [self.params.V_dc/L, 0],
            [0, self.params.V_dc/L],
            [0, 0],
            [0, 0],
        ])
        
        # D matrix (4x3): disturbance effect
        self.D = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [-1/C, 0, 0],
            [0, -1/C, 0],
        ])
    
    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Compute state derivative: dx/dt = A*x + B*u + D*w
        
        Args:
            t: Time [s]
            x: State vector [i_d, i_q, v_d, v_q]
            u: Control input [d_d, d_q]
            w: Disturbance vector [v_grid_d, v_grid_q, P_dc]
            
        Returns:
            State derivative dx/dt
        """
        return self.A @ x + self.B @ u + self.D @ w
    
    def get_ode_function(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Return ODE function compatible with scipy.integrate.odeint.
        
        Returns:
            Function f(t, x) where u and w are treated as time-varying inputs
        """
        def ode_func(t: float, x: np.ndarray) -> np.ndarray:
            # For simulation, we need to provide u(t) and w(t)
            # This will be overridden by the simulation module
            u = np.zeros(2)
            w = np.zeros(3)
            return self.dynamics(t, x, u, w)
        
        return ode_func
    
    def compute_power(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Compute active and reactive power from state.
        
        Args:
            x: State vector [i_d, i_q, v_d, v_q]
            
        Returns:
            (P, Q): Active and reactive power [W, VAR]
        """
        i_d, i_q, v_d, v_q = x
        P = 1.5 * (v_d * i_d + v_q * i_q)
        Q = 1.5 * (v_q * i_d - v_d * i_q)
        return P, Q
    
    def compute_current_magnitude(self, x: np.ndarray) -> float:
        """
        Compute current magnitude in per-unit.
        
        Args:
            x: State vector [i_d, i_q, v_d, v_q]
            
        Returns:
            Current magnitude [p.u.]
        """
        i_d, i_q, _, _ = x
        i_mag = np.sqrt(i_d**2 + i_q**2)
        # Base current: I_base = P_base / (sqrt(3) * V_grid)
        # Assuming 10kW system: I_base = 10000 / (sqrt(3) * 400) ≈ 14.43A
        i_base = 10000 / (np.sqrt(3) * self.params.V_grid)
        return i_mag / i_base
    
    def create_disturbance_profile(
        self, 
        t: np.ndarray,
        disturbance_type: str = "cloud_event",
        severity: float = 0.7
    ) -> np.ndarray:
        """
        Create disturbance profile for simulation.
        
        Args:
            t: Time array [s]
            disturbance_type: "cloud_event" or "grid_fault"
            severity: Disturbance severity (0-1)
            
        Returns:
            Array of disturbance vectors w(t) with shape (len(t), 3)
        """
        w = np.zeros((len(t), 3))
        
        if disturbance_type == "cloud_event":
            # 70% irradiance drop over 50ms linear ramp
            t_start = 0.1  # Start at 100ms
            t_ramp = 0.05  # 50ms ramp
            
            for i, time in enumerate(t):
                if t_start <= time < t_start + t_ramp:
                    # Linear ramp down
                    ramp_factor = 1 - severity * (time - t_start) / t_ramp
                elif time >= t_start + t_ramp:
                    # Sustained drop
                    ramp_factor = 1 - severity
                else:
                    ramp_factor = 1.0
                
                # P_dc disturbance (3rd element)
                w[i, 2] = ramp_factor
        
        elif disturbance_type == "grid_fault":
            # 30% voltage dip for 100ms
            t_start = 0.1  # Start at 100ms
            t_duration = 0.1  # 100ms duration
            
            for i, time in enumerate(t):
                if t_start <= time < t_start + t_duration:
                    # Voltage dip
                    v_dip = 1 - severity
                    w[i, 0] = v_dip * self.params.V_grid_peak  # v_grid_d
                    w[i, 1] = 0  # v_grid_q (assume aligned with d-axis)
                else:
                    w[i, 0] = self.params.V_grid_peak
                    w[i, 1] = 0
        
        return w
