"""Tests for physics model."""

import numpy as np
from src.physics.inverter import InverterModel, InverterParameters


def test_inverter_parameters():
    """Test inverter parameter initialization."""
    params = InverterParameters()
    
    assert params.L == 2.3e-3
    assert params.C == 10e-6
    assert params.R == 0.1
    assert params.V_dc == 800.0
    assert params.f_grid == 50.0
    
    # Test computed properties
    assert params.omega_grid == 2 * np.pi * 50.0
    assert params.V_grid_peak > 0


def test_inverter_model_creation():
    """Test inverter model creation."""
    model = InverterModel()
    
    # Check matrices exist
    assert hasattr(model, 'A')
    assert hasattr(model, 'B')
    assert hasattr(model, 'D')
    
    # Check matrix shapes
    assert model.A.shape == (4, 4)
    assert model.B.shape == (4, 2)
    assert model.D.shape == (4, 3)


def test_dynamics():
    """Test dynamics computation."""
    model = InverterModel()
    
    # Test state
    x = np.array([5.0, 2.0, 400.0, 0.0])
    u = np.array([0.1, 0.05])
    w = np.array([400.0, 0.0, 1.0])
    
    # Compute dynamics
    dx = model.dynamics(0.0, x, u, w)
    
    # Check output shape
    assert dx.shape == (4,)
    
    # Check that dynamics are not all zeros
    assert not np.allclose(dx, 0.0)
    
    # Linear system property: dx = A*x + B*u + D*w
    dx_expected = model.A @ x + model.B @ u + model.D @ w
    assert np.allclose(dx, dx_expected, rtol=1e-10)


def test_power_computation():
    """Test power computation."""
    model = InverterModel()
    
    # Test state
    x = np.array([10.0, 0.0, 400.0, 0.0])
    
    # Compute power
    P, Q = model.compute_power(x)
    
    # Check types
    assert isinstance(P, float)
    assert isinstance(Q, float)
    
    # For this state, Q should be 0 (i_q = 0, v_q = 0)
    assert abs(Q) < 1e-10
    
    # P should be positive
    assert P > 0
    
    # Check formula: P = 1.5 * (v_d*i_d + v_q*i_q)
    expected_P = 1.5 * (x[2] * x[0] + x[3] * x[1])
    assert abs(P - expected_P) < 1e-10


def test_current_magnitude():
    """Test current magnitude computation."""
    model = InverterModel()
    
    # Test state with known current
    x = np.array([14.43, 0.0, 400.0, 0.0])  # ~1 p.u. current
    
    # Compute current magnitude
    i_pu = model.compute_current_magnitude(x)
    
    # Should be approximately 1 p.u.
    assert abs(i_pu - 1.0) < 0.1
    
    # Test with zero current
    x_zero = np.array([0.0, 0.0, 400.0, 0.0])
    i_pu_zero = model.compute_current_magnitude(x_zero)
    assert abs(i_pu_zero) < 1e-10


def test_disturbance_profile():
    """Test disturbance profile creation."""
    model = InverterModel()
    
    # Create time array
    t = np.linspace(0, 0.3, 300)  # 300ms
    
    # Test cloud event
    w_cloud = model.create_disturbance_profile(t, "cloud_event", severity=0.7)
    
    # Check shape
    assert w_cloud.shape == (len(t), 3)
    
    # Check that P_dc (3rd column) has disturbance
    # Should start at 1.0, drop to 0.3
    assert w_cloud[0, 2] == 1.0  # Before disturbance
    assert w_cloud[-1, 2] == 0.3  # After disturbance (1 - 0.7)
    
    # Test grid fault
    w_fault = model.create_disturbance_profile(t, "grid_fault", severity=0.3)
    
    # Check shape
    assert w_fault.shape == (len(t), 3)
    
    # Check voltage dip
    assert w_fault[0, 0] == model.params.V_grid_peak  # Before
    # During fault should be reduced


def test_ode_function():
    """Test ODE function generation."""
    model = InverterModel()
    
    # Get ODE function
    ode_func = model.get_ode_function()
    
    # Test function signature
    x = np.array([5.0, 2.0, 400.0, 0.0])
    dx = ode_func(0.0, x)
    
    # Check output
    assert dx.shape == (4,)
    assert isinstance(dx, np.ndarray)


if __name__ == "__main__":
    # Run tests
    test_inverter_parameters()
    test_inverter_model_creation()
    test_dynamics()
    test_power_computation()
    test_current_magnitude()
    test_disturbance_profile()
    test_ode_function()
    
    print("All physics tests passed!")