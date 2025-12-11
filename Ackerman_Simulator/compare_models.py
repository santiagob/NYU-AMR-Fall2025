#!/usr/bin/env python3
"""
Model Comparison Script
=======================

Demonstrates how to easily switch between kinematic and dynamic vehicle models
and compares their behavior on identical scenarios.

This script:
1. Creates both models with identical initial conditions
2. Runs both for the same control inputs
3. Compares their trajectories and responses
4. Generates side-by-side visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from models.model_factory import create_vehicle, get_vehicle_params

def compare_models_on_straight_line():
    """Compare models on straight-line motion with acceleration"""
    print("\n" + "="*60)
    print("TEST 1: STRAIGHT LINE MOTION (Acceleration)")
    print("="*60)
    
    models = {
        'kinematic': create_vehicle('kinematic', x=0, y=0, yaw=0),
        'dynamic': create_vehicle('dynamic', x=0, y=0, yaw=0),
    }
    
    dt = 0.1
    duration = 5.0
    steps = int(duration / dt)
    
    # Control input: constant throttle, no steering
    throttle = 0.5
    steer_rad = 0.0
    
    history = {
        'kinematic': {
            't': [], 'x': [], 'y': [], 'vx': [], 'vy': [], 'r': []
        },
        'dynamic': {
            't': [], 'x': [], 'y': [], 'vx': [], 'vy': [], 'r': []
        }
    }
    
    # Run simulation
    for step in range(steps):
        t = step * dt
        for model_name, vehicle in models.items():
            vehicle.update(throttle, steer_rad, dt)
            history[model_name]['t'].append(t)
            history[model_name]['x'].append(vehicle.x)
            history[model_name]['y'].append(vehicle.y)
            history[model_name]['vx'].append(vehicle.vx)
            history[model_name]['vy'].append(vehicle.vy)
            history[model_name]['r'].append(vehicle.r)
    
    # Print comparison
    print(f"\n{'Metric':<25} {'Kinematic':<20} {'Dynamic':<20}")
    print("-" * 65)
    print(f"{'Final X position':<25} {history['kinematic']['x'][-1]:<20.4f} {history['dynamic']['x'][-1]:<20.4f}")
    print(f"{'Final speed (vx)':<25} {history['kinematic']['vx'][-1]:<20.4f} {history['dynamic']['vx'][-1]:<20.4f}")
    print(f"{'Max lateral velocity':<25} {max(history['kinematic']['vy']):<20.4f} {max(history['dynamic']['vy']):<20.4f}")
    
    return history


def compare_models_on_turn():
    """Compare models during 25-degree steering turn"""
    print("\n" + "="*60)
    print("TEST 2: 25-DEGREE STEERING TURN")
    print("="*60)
    
    models = {
        'kinematic': create_vehicle('kinematic', x=0, y=0, yaw=0),
        'dynamic': create_vehicle('dynamic', x=0, y=0, yaw=0),
    }
    
    dt = 0.1
    duration = 5.0
    steps = int(duration / dt)
    
    # Control input: constant speed, 25-degree steering
    throttle = 0.5
    steer_rad = np.radians(25)
    
    history = {
        'kinematic': {
            't': [], 'x': [], 'y': [], 'yaw': [], 'vx': [], 'vy': [], 'r': []
        },
        'dynamic': {
            't': [], 'x': [], 'y': [], 'yaw': [], 'vx': [], 'vy': [], 'r': []
        }
    }
    
    # Run simulation
    for step in range(steps):
        t = step * dt
        for model_name, vehicle in models.items():
            vehicle.update(throttle, steer_rad, dt)
            history[model_name]['t'].append(t)
            history[model_name]['x'].append(vehicle.x)
            history[model_name]['y'].append(vehicle.y)
            history[model_name]['yaw'].append(vehicle.yaw)
            history[model_name]['vx'].append(vehicle.vx)
            history[model_name]['vy'].append(vehicle.vy)
            history[model_name]['r'].append(vehicle.r)
    
    # Print comparison
    final_kinematic = {
        'x': history['kinematic']['x'][-1],
        'y': history['kinematic']['y'][-1],
        'yaw': np.degrees(history['kinematic']['yaw'][-1]),
        'vx': history['kinematic']['vx'][-1],
        'vy': history['kinematic']['vy'][-1],
    }
    
    final_dynamic = {
        'x': history['dynamic']['x'][-1],
        'y': history['dynamic']['y'][-1],
        'yaw': np.degrees(history['dynamic']['yaw'][-1]),
        'vx': history['dynamic']['vx'][-1],
        'vy': history['dynamic']['vy'][-1],
    }
    
    print(f"\n{'Metric':<25} {'Kinematic':<20} {'Dynamic':<20}")
    print("-" * 65)
    print(f"{'Final X [m]':<25} {final_kinematic['x']:<20.4f} {final_dynamic['x']:<20.4f}")
    print(f"{'Final Y [m]':<25} {final_kinematic['y']:<20.4f} {final_dynamic['y']:<20.4f}")
    print(f"{'Final yaw [deg]':<25} {final_kinematic['yaw']:<20.4f} {final_dynamic['yaw']:<20.4f}")
    print(f"{'Final speed vx [m/s]':<25} {final_kinematic['vx']:<20.4f} {final_dynamic['vx']:<20.4f}")
    print(f"{'Lateral velocity vy [m/s]':<25} {final_kinematic['vy']:<20.6f} {final_dynamic['vy']:<20.6f}")
    
    # Interesting observation: kinematic model should have vy = 0 (no lateral slip)
    print(f"\n✓ Kinematic model rear-axle lateral velocity: {final_kinematic['vy']:.6f} m/s")
    print(f"  (Should be ~0 due to no-slip constraint)")
    
    return history


def plot_comparisons(hist1, hist2):
    """Plot side-by-side comparison of straight-line and turn scenarios"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Vehicle Model Comparison: Kinematic vs Dynamic', fontsize=16, fontweight='bold')
    
    # Row 1: Straight line motion
    # Plot 1.1: Speed comparison
    ax = axes[0, 0]
    ax.plot(hist1['kinematic']['t'], hist1['kinematic']['vx'], 'b-', linewidth=2, label='Kinematic')
    ax.plot(hist1['dynamic']['t'], hist1['dynamic']['vx'], 'r--', linewidth=2, label='Dynamic')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Speed vx [m/s]')
    ax.set_title('Straight Line: Longitudinal Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 1.2: Lateral velocity (should be 0 for kinematic)
    ax = axes[0, 1]
    ax.plot(hist1['kinematic']['t'], hist1['kinematic']['vy'], 'b-', linewidth=2, label='Kinematic')
    ax.plot(hist1['dynamic']['t'], hist1['dynamic']['vy'], 'r--', linewidth=2, label='Dynamic')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Lateral velocity vy [m/s]')
    ax.set_title('Straight Line: Lateral Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 1.3: Trajectory straight
    ax = axes[0, 2]
    ax.plot(hist1['kinematic']['x'], hist1['kinematic']['y'], 'b-', linewidth=2, label='Kinematic')
    ax.plot(hist1['dynamic']['x'], hist1['dynamic']['y'], 'r--', linewidth=2, label='Dynamic')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Straight Line: Trajectory')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Turn maneuver
    # Plot 2.1: Yaw angle
    ax = axes[1, 0]
    ax.plot(hist2['kinematic']['t'], np.degrees(hist2['kinematic']['yaw']), 'b-', linewidth=2, label='Kinematic')
    ax.plot(hist2['dynamic']['t'], np.degrees(hist2['dynamic']['yaw']), 'r--', linewidth=2, label='Dynamic')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Yaw angle [deg]')
    ax.set_title('Turn: Yaw Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2.2: Yaw rate
    ax = axes[1, 1]
    ax.plot(hist2['kinematic']['t'], hist2['kinematic']['r'], 'b-', linewidth=2, label='Kinematic')
    ax.plot(hist2['dynamic']['t'], hist2['dynamic']['r'], 'r--', linewidth=2, label='Dynamic')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Yaw rate r [rad/s]')
    ax.set_title('Turn: Yaw Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2.3: Trajectory turn
    ax = axes[1, 2]
    ax.plot(hist2['kinematic']['x'], hist2['kinematic']['y'], 'b-', linewidth=2, label='Kinematic')
    ax.plot(hist2['dynamic']['x'], hist2['dynamic']['y'], 'r--', linewidth=2, label='Dynamic')
    ax.scatter(hist2['kinematic']['x'][0], hist2['kinematic']['y'][0], c='b', s=100, marker='o', label='Start')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Turn: Trajectory')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to model_comparison.png")
    plt.show()


def main():
    print("\n" + "="*60)
    print("VEHICLE MODEL COMPARISON")
    print("Kinematic vs Dynamic Models")
    print("="*60)
    
    # Test 1: Straight line
    hist_straight = compare_models_on_straight_line()
    
    # Test 2: Turn
    hist_turn = compare_models_on_turn()
    
    # Plot comparisons
    plot_comparisons(hist_straight, hist_turn)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n✓ Kinematic Model Advantages:")
    print("  - Zero lateral slip (vy ≈ 0 in straight line)")
    print("  - Numerically stable")
    print("  - Predictable behavior")
    print("  - Faster computation")
    
    print("\n✓ Dynamic Model Advantages:")
    print("  - More realistic tire forces")
    print("  - Accounts for slip angles")
    print("  - More detailed physics")
    print("  - Useful for research and validation")
    
    print("\n✓ Key Takeaway:")
    print("  Use --model kinematic or --model dynamic to switch between them:")
    print("  $ python main.py --model kinematic")
    print("  $ python main.py --model dynamic")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
