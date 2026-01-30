"""Main simulation framework for PINN-MPC vs PI controller comparison."""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Tuple, Optional, Any
import os
import json

from src.physics.simulation import InverterSimulator
from src.pinn.trainer import train_pinn_from_scratch
from src.pinn.model import PINN
from src.mpc.controller import NeuralMPC, MPCConfig
from src.mpc.pi_controller import PIController
from src.physics.inverter import InverterModel


class ComparisonSimulation:
    """Comparison simulation between PINN-MPC and PI controller."""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Physics model
        self.physics_model = InverterModel()
        
        # Reference state (normal operating point)
        self.x_ref = np.array([10.0, 0.0, self.physics_model.params.V_grid_peak, 0.0])
        
        # Initial state
        self.x0 = np.array([0.0, 0.0, self.physics_model.params.V_grid_peak, 0.0])
        
        # Simulation parameters
        self.simulation_steps = 500  # 50ms at 1e-4 time step
        self.dt = 1e-4
        
        # Disturbance parameters
        self.disturbance_start = 100  # Start at 10ms
        self.disturbance_duration = 50  # 5ms for cloud event
        
        # Performance metrics
        self.metrics = {}
    
    def create_disturbance(self, disturbance_type: str = "cloud_event") -> np.ndarray:
        """
        Create disturbance sequence for simulation.
        
        Args:
            disturbance_type: "cloud_event" or "grid_fault"
            
        Returns:
            Disturbance sequence
        """
        disturbance = np.zeros((self.simulation_steps, 4))
        
        if disturbance_type == "cloud_event":
            # 70% irradiance drop
            severity = 0.7
            for step in range(self.simulation_steps):
                if self.disturbance_start <= step < self.disturbance_start + self.disturbance_duration:
                    # Linear ramp down
                    ramp_factor = 1 - severity * (step - self.disturbance_start) / self.disturbance_duration
                elif step >= self.disturbance_start + self.disturbance_duration:
                    # Sustained drop
                    ramp_factor = 1 - severity
                else:
                    ramp_factor = 1.0
                
                # Apply as scaling factor to power (affects dynamics through w)
                # For simplicity, we'll scale the reference
                disturbance[step] = self.x_ref * (ramp_factor - 1.0) * 0.1
        
        elif disturbance_type == "grid_fault":
            # 30% voltage dip
            severity = 0.3
            for step in range(self.simulation_steps):
                if self.disturbance_start <= step < self.disturbance_start + self.disturbance_duration:
                    # Voltage dip
                    v_dip = 1 - severity
                    disturbance[step, 2] = self.x_ref[2] * (v_dip - 1.0)
        
        return disturbance
    
    def train_pinn_model(self, n_samples: int = 50000, epochs: int = 100) -> PINN:
        """
        Train PINN model from scratch.
        
        Args:
            n_samples: Number of training samples
            epochs: Training epochs
            
        Returns:
            Trained PINN model
        """
        print("=" * 60)
        print("Training PINN Model")
        print("=" * 60)
        
        trainer = train_pinn_from_scratch(
            n_samples=n_samples,
            epochs=epochs,
            batch_size=64,
            learning_rate=1e-3,
            physics_weight=10.0,
            save_dir=os.path.join(self.save_dir, "pinn_training")
        )
        
        return trainer.model
    
    def run_pi_simulation(self, disturbance_type: str = "cloud_event") -> Dict:
        """
        Run PI controller simulation.
        
        Args:
            disturbance_type: Type of disturbance
            
        Returns:
            Simulation results
        """
        print("\n" + "=" * 60)
        print("Running PI Controller Simulation")
        print("=" * 60)
        
        # Create PI controller
        pi_controller = PIController()
        pi_controller.tune_gains(desired_settling_time=0.08, desired_overshoot=0.05)
        
        # Create disturbance
        disturbance = self.create_disturbance(disturbance_type)
        
        # Run simulation
        results = pi_controller.simulate_closed_loop(
            x0=self.x0,
            i_d_ref=self.x_ref[0],
            i_q_ref=self.x_ref[1],
            simulation_steps=self.simulation_steps,
            disturbance=disturbance
        )
        
        # Compute performance metrics
        metrics = self._compute_metrics(results['x_history'], disturbance_type)
        self.metrics['pi'] = metrics
        
        # Plot results
        fig = pi_controller.plot_results(
            results,
            title=f"PI Controller - {disturbance_type.replace('_', ' ').title()}"
        )
        fig.savefig(
            os.path.join(self.save_dir, f"pi_controller_{disturbance_type}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)
        
        return results
    
    def run_mpc_simulation(self, model: PINN, disturbance_type: str = "cloud_event") -> Dict:
        """
        Run MPC controller simulation.
        
        Args:
            model: Trained PINN model
            disturbance_type: Type of disturbance
            
        Returns:
            Simulation results
        """
        print("\n" + "=" * 60)
        print("Running PINN-MPC Simulation")
        print("=" * 60)
        
        # Create MPC controller
        mpc_config = MPCConfig(horizon=10, max_iter=50)
        mpc = NeuralMPC(model, mpc_config)
        
        # Create disturbance
        disturbance = self.create_disturbance(disturbance_type)
        
        # Run simulation
        results = mpc.simulate_closed_loop(
            x0=self.x0,
            x_ref=self.x_ref,
            simulation_steps=self.simulation_steps,
            disturbance=disturbance
        )
        
        # Compute performance metrics
        metrics = self._compute_metrics(results['x_history'], disturbance_type)
        self.metrics['mpc'] = metrics
        
        # Plot results
        fig = mpc.plot_results(
            results,
            x_ref=self.x_ref,
            title=f"PINN-MPC - {disturbance_type.replace('_', ' ').title()}"
        )
        fig.savefig(
            os.path.join(self.save_dir, f"mpc_controller_{disturbance_type}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)
        
        return results
    
    def _compute_metrics(self, x_history: np.ndarray, disturbance_type: str) -> Dict:
        """
        Compute performance metrics.
        
        Args:
            x_history: State history
            disturbance_type: Type of disturbance
            
        Returns:
            Performance metrics dictionary
        """
        # Extract currents
        i_d_history = x_history[:, 0]
        i_q_history = x_history[:, 1]
        
        # Compute current magnitude in p.u.
        current_pu = []
        for state in x_history:
            current_pu.append(self.physics_model.compute_current_magnitude(state))
        current_pu = np.array(current_pu)
        
        # Settling time (to within 1% of reference)
        settling_threshold = 0.01 * np.abs(self.x_ref[0])
        error = np.abs(i_d_history - self.x_ref[0])
        
        # Find when error stays below threshold
        below_threshold = error < settling_threshold
        settling_step = None
        
        for step in range(self.disturbance_start, len(below_threshold)):
            if np.all(below_threshold[step:step+10]):  # Stay below for 10 steps
                settling_step = step
                break
        
        settling_time = (settling_step - self.disturbance_start) * self.dt if settling_step else float('inf')
        
        # Peak overshoot
        peak_error = np.max(error[self.disturbance_start:self.disturbance_start + 100])
        peak_overshoot = peak_error / np.abs(self.x_ref[0]) if self.x_ref[0] != 0 else 0
        
        # Constraint violations
        max_current = np.max(current_pu)
        constraint_violation = max(0, max_current - 1.2)
        
        # RMSE after disturbance
        post_disturbance_start = self.disturbance_start
        post_disturbance_end = min(self.disturbance_start + 200, len(x_history))
        
        if post_disturbance_end > post_disturbance_start:
            rmse = np.sqrt(np.mean(
                (x_history[post_disturbance_start:post_disturbance_end, 0] - self.x_ref[0]) ** 2
            ))
        else:
            rmse = float('inf')
        
        return {
            'settling_time_ms': settling_time * 1000,
            'peak_overshoot_percent': peak_overshoot * 100,
            'max_current_pu': max_current,
            'constraint_violation_pu': constraint_violation,
            'rmse_after_disturbance': rmse,
            'disturbance_type': disturbance_type
        }
    
    def plot_comparison(self, pi_results: Dict, mpc_results: Dict, disturbance_type: str) -> plt.Figure:
        """
        Create comparison plot between PI and MPC.
        
        Args:
            pi_results: PI controller results
            mpc_results: MPC controller results
            disturbance_type: Type of disturbance
            
        Returns:
            Comparison figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"PI vs PINN-MPC Comparison - {disturbance_type.replace('_', ' ').title()}",
            fontsize=16
        )
        
        # Current tracking (d-axis)
        axes[0, 0].plot(pi_results['x_history'][:, 0], 'b-', label='PI', alpha=0.7)
        axes[0, 0].plot(mpc_results['x_history'][:, 0], 'r-', label='PINN-MPC', alpha=0.7)
        axes[0, 0].axhline(y=self.x_ref[0], color='k', linestyle='--', label='Reference')
        axes[0, 0].axvline(x=self.disturbance_start, color='g', linestyle=':', label='Disturbance Start')
        axes[0, 0].set_ylabel('d-axis Current [A]')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Current Tracking (d-axis)')
        
        # Current magnitude (p.u.)
        pi_current_pu = []
        for state in pi_results['x_history']:
            pi_current_pu.append(self.physics_model.compute_current_magnitude(state))
        
        mpc_current_pu = []
        for state in mpc_results['x_history']:
            mpc_current_pu.append(self.physics_model.compute_current_magnitude(state))
        
        axes[0, 1].plot(pi_current_pu, 'b-', label='PI', alpha=0.7)
        axes[0, 1].plot(mpc_current_pu, 'r-', label='PINN-MPC', alpha=0.7)
        axes[0, 1].axhline(y=1.2, color='r', linestyle='--', label='Limit (1.2 p.u.)')
        axes[0, 1].axvline(x=self.disturbance_start, color='g', linestyle=':', label='Disturbance Start')
        axes[0, 1].set_ylabel('Current [p.u.]')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Current Magnitude (p.u.)')
        
        # Control effort
        axes[1, 0].plot(pi_results['u_history'][:, 0], 'b-', label='PI d_d', alpha=0.7)
        axes[1, 0].plot(pi_results['u_history'][:, 1], 'b--', label='PI d_q', alpha=0.7)
        axes[1, 0].plot(mpc_results['u_history'][:, 0], 'r-', label='MPC d_d', alpha=0.7)
        axes[1, 0].plot(mpc_results['u_history'][:, 1], 'r--', label='MPC d_q', alpha=0.7)
        axes[1, 0].set_ylabel('Control Inputs')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Control Effort')
        
        # Performance metrics table
        metrics_text = "Performance Metrics:\n\n"
        for controller, metrics in self.metrics.items():
            metrics_text += f"{controller.upper()}:\n"
            metrics_text += f"  Settling Time: {metrics['settling_time_ms']:.1f} ms\n"
            metrics_text += f"  Peak Overshoot: {metrics['peak_overshoot_percent']:.1f}%\n"
            metrics_text += f"  Max Current: {metrics['max_current_pu']:.2f} p.u.\n"
            metrics_text += f"  Constraint Violation: {metrics['constraint_violation_pu']:.3f} p.u.\n"
            metrics_text += f"  RMSE: {metrics['rmse_after_disturbance']:.3f}\n\n"
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def run_complete_comparison(self, train_new_model: bool = True) -> None:
        """
        Run complete comparison pipeline.
        
        Args:
            train_new_model: Whether to train a new PINN model
        """
        print("=" * 60)
        print("PINN-MPC vs PI Controller Comparison")
        print("=" * 60)
        
        # Train or load PINN model
        if train_new_model:
            pinn_model = self.train_pinn_model(n_samples=20000, epochs=50)
        else:
            # Load pre-trained model
            pinn_model = PINN()
            # Note: In practice, you would load from checkpoint
            print("Using untrained model for demonstration")
        
        # Test scenarios
        scenarios = ["cloud_event", "grid_fault"]
        
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"Testing {scenario.replace('_', ' ').title()}")
            print(f"{'='*60}")
            
            # Run PI controller
            pi_results = self.run_pi_simulation(scenario)
            
            # Run MPC controller
            mpc_results = self.run_mpc_simulation(pinn_model, scenario)
            
            # Create comparison plot
            comparison_fig = self.plot_comparison(pi_results, mpc_results, scenario)
            comparison_fig.savefig(
                os.path.join(self.save_dir, f"comparison_{scenario}.png"),
                dpi=150, bbox_inches='tight'
            )
            plt.close(comparison_fig)
            
            # Print performance comparison
            print(f"\nPerformance Comparison for {scenario}:")
            print("-" * 40)
            
            pi_metrics = self.metrics['pi']
            mpc_metrics = self.metrics['mpc']
            
            # Compute improvement
            settling_improvement = (pi_metrics['settling_time_ms'] - mpc_metrics['settling_time_ms']) / pi_metrics['settling_time_ms'] * 100
            overshoot_improvement = (pi_metrics['peak_overshoot_percent'] - mpc_metrics['peak_overshoot_percent']) / pi_metrics['peak_overshoot_percent'] * 100
            
            print(f"Settling Time: PI={pi_metrics['settling_time_ms']:.1f}ms, "
                  f"MPC={mpc_metrics['settling_time_ms']:.1f}ms "
                  f"({settling_improvement:+.1f}% improvement)")
            
            print(f"Peak Overshoot: PI={pi_metrics['peak_overshoot_percent']:.1f}%, "
                  f"MPC={mpc_metrics['peak_overshoot_percent']:.1f}% "
                  f"({overshoot_improvement:+.1f}% improvement)")
            
            print(f"Max Current: PI={pi_metrics['max_current_pu']:.3f}p.u., "
                  f"MPC={mpc_metrics['max_current_pu']:.3f}p.u.")
            
            print(f"Constraint Violation: PI={pi_metrics['constraint_violation_pu']:.3f}p.u., "
                  f"MPC={mpc_metrics['constraint_violation_pu']:.3f}p.u.")
            
            print(f"RMSE: PI={pi_metrics['rmse_after_disturbance']:.3f}, "
                  f"MPC={mpc_metrics['rmse_after_disturbance']:.3f}")
            
            # Save metrics to file
            metrics_path = os.path.join(self.save_dir, f"metrics_{scenario}.json")
            with open(metrics_path, 'w') as f:
                json.dump({
                    'pi': pi_metrics,
                    'mpc': mpc_metrics,
                    'improvement': {
                        'settling_time_percent': settling_improvement,
                        'overshoot_percent': overshoot_improvement
                    }
                }, f, indent=2)
            
            print(f"Metrics saved to {metrics_path}")
        
        print("\n" + "=" * 60)
        print("Comparison completed successfully!")
        print("=" * 60)


def main():
    """Main entry point."""
    # Create comparison simulation
    simulation = ComparisonSimulation(save_dir="results")
    
    # Run complete comparison
    simulation.run_complete_comparison(train_new_model=False)


if __name__ == "__main__":
    main()
