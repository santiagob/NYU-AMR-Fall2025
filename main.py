"""
Main simulation loop: Path Planning → Vehicle Control → Trajectory Tracking

This script demonstrates the complete pipeline:
1) Define grid obstacles and start/goal
2) Plan a path using A*
3) Initialize the selected vehicle model (kinematic or dynamic)
4) Track the path using a controller (Stanley/PID by default)
5) Plot or log trajectory/stability metrics

Usage:
    python main.py              # default (kinematic)
    # Or from other modules: main(model_type="dynamic")

Outputs:
    - Console logs for planning and controller updates
    - Matplotlib windows (trajectory/metrics) depending on environment
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import argparse
from models.model_factory import create_vehicle, get_vehicle_params
from planners.a_star import AStarPlanner
from controllers.stanley import StanleyController, PIDController

def main(model_type="kinematic"):
    """Main simulation loop using selected vehicle model."""
    print(f"\n{'='*60}")
    print(f"Initializing Simulation with {model_type.upper()} Model")
    print(f"{'='*60}\n")
    
    # Define Obstacles (x, y)
    ox, oy = [], []
    for i in range(60): # Wall borders
        ox.append(i); oy.append(0.0)
        ox.append(i); oy.append(60.0)
    for i in range(60):
        ox.append(0.0); oy.append(i)
        ox.append(60.0); oy.append(i)
    
    # Add some obstacles in the middle
    for i in range(10, 20):
        ox.append(i); oy.append(30.0)
    for i in range(30, 40):
        ox.append(40.0); oy.append(i)

    # Start and Goal
    sx, sy = 5.0, 5.0
    gx, gy = 50.0, 50.0
    robot_radius = 2.0
    grid_size = 1.0

    # --- 2. MOTION PLANNING ---
    print("Running A* Planner...")
    planner = AStarPlanner(ox, oy, grid_size, robot_radius)
    path = planner.planning(sx, sy, gx, gy)
    
    if path is None:
        print("Path Planning Failed!")
        return
        
    path_x = path[:, 0]
    path_y = path[:, 1]
    
    # --- 3. INITIALIZE VEHICLE WITH SELECTED MODEL ---
    print(f"\nInitializing {model_type} vehicle model...")
    vehicle = create_vehicle(model_type, x=sx, y=sy, yaw=np.radians(45))
    
    # Get parameters for the selected model
    params = get_vehicle_params(model_type)
    
    # --- 4. INITIALIZE CONTROLLERS ---
    # Adjust controller gains based on model type
    if model_type.lower() == "kinematic":
        # Kinematic model is more stable, use moderate gains
        k_gain = 0.8
        k_soft = 2.0
        kp_speed = 0.5
        ki_speed = 0.05
        kd_speed = 0.1
        print("Using conservative gains for kinematic model (stable)")
    else:
        # Dynamic model is more sensitive, use reduced gains
        k_gain = 0.6
        k_soft = 2.0
        kp_speed = 0.3
        ki_speed = 0.02
        kd_speed = 0.05
        print("Using reduced gains for dynamic model (stable)")
    
    stanley = StanleyController(k_gain=k_gain, k_soft=k_soft)
    speed_pid = PIDController(Kp=kp_speed, Ki=ki_speed, Kd=kd_speed)
    
    # History for plotting (added hyaw for animation)
    hx, hy, hv, hsteer, hcte, hyaw = [], [], [], [], [], []
    
    target_speed = 5.0 # m/s
    dt = 0.1
    max_time = 50.0
    
    # Lookahead parameters for path following
    idx = 0 # Current path index
    lookahead_distance = 3.0  # meters - advance target point ahead of vehicle
    
    print("Starting Control Loop...")
    for t in np.arange(0, max_time, dt):
        # A. Find nearest point on path (for reference)
        dists = np.hypot(vehicle.state[0] - path_x, vehicle.state[1] - path_y)
        nearest_idx = np.argmin(dists)
        
        # B. Lookahead: Find target point ahead on path
        # Start from nearest point and search forward for lookahead distance
        idx = nearest_idx
        for i in range(nearest_idx, len(path_x)):
            dist_to_point = np.hypot(vehicle.state[0] - path_x[i], 
                                     vehicle.state[1] - path_y[i])
            if dist_to_point >= lookahead_distance:
                idx = i
                break
        # Ensure we don't go past the end
        idx = min(idx, len(path_x) - 1)
        
        # C. Check if goal reached
        dist_to_goal = np.hypot(vehicle.state[0] - gx, vehicle.state[1] - gy)
        if dist_to_goal < 2.0:
            print(f"Goal Reached at t={t:.1f}s!")
            break

        # D. Compute Control using Stanley controller
        steer_cmd, cte = stanley.compute(vehicle.state, path_x, path_y, idx)
        
        # E. Clamp Steering to vehicle limits
        steer_cmd = np.clip(steer_cmd, -params.MAX_STEER, params.MAX_STEER)
        
        # F. Speed Control with PID
        current_speed = np.hypot(vehicle.vx, vehicle.vy)
        throttle = speed_pid.compute(target_speed, current_speed, dt)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # G. Update Vehicle Dynamics
        vehicle.update(throttle, steer_cmd, dt)
        
        # H. Save History for plotting and analysis
        hx.append(vehicle.x)
        hy.append(vehicle.y)
        hyaw.append(vehicle.yaw)
        hv.append(current_speed)
        hsteer.append(steer_cmd)
        hcte.append(cte)

    
    # --- 5. VISUALIZATION / RESULTS ---
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Simulation Results: {model_type.upper()} Model', fontsize=16, fontweight='bold')
    
    # Plot 1: Path and trajectory
    ax = axes[0, 0]
    ax.plot(path_x, path_y, 'g--', linewidth=2, label='Planned Path')
    ax.plot(hx, hy, 'b-', linewidth=2, label='Vehicle Trajectory')
    ax.plot(sx, sy, 'go', markersize=10, label='Start')
    ax.plot(gx, gy, 'ro', markersize=10, label='Goal')
    ax.plot(ox, oy, 'kx', markersize=5, label='Obstacles')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Vehicle Trajectory')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Speed profile
    ax = axes[0, 1]
    t_array = np.arange(len(hv)) * dt
    ax.plot(t_array, hv, 'b-', linewidth=2, label='Actual Speed')
    ax.axhline(y=target_speed, color='g', linestyle='--', linewidth=2, label='Target Speed')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Speed [m/s]')
    ax.set_title('Speed Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Steering angle
    ax = axes[1, 0]
    ax.plot(t_array, np.degrees(hsteer), 'r-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Steering Angle [deg]')
    ax.set_title('Steering Command')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cross-track error
    ax = axes[1, 1]
    ax.plot(t_array, hcte, 'm-', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Cross-Track Error [m]')
    ax.set_title('Path Following Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'simulation_{model_type}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY ({model_type.upper()} MODEL)")
    print(f"{'='*60}")
    print(f"Total distance traveled: {np.sum(np.hypot(np.diff(hx), np.diff(hy))):.2f} m")
    print(f"Final position: ({hx[-1]:.2f}, {hy[-1]:.2f})")
    print(f"Goal position: ({gx:.2f}, {gy:.2f})")
    print(f"Distance to goal: {np.hypot(hx[-1]-gx, hy[-1]-gy):.2f} m")
    print(f"Average speed: {np.mean(hv):.2f} m/s")
    print(f"Max speed: {np.max(hv):.2f} m/s")
    print(f"Average cross-track error: {np.mean(hcte):.2f} m")
    print(f"Max cross-track error: {np.max(hcte):.2f} m")
    print(f"{'='*60}\n")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Ackermann simulator with switchable models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model kinematic    # Run with kinematic model
  python main.py --model dynamic      # Run with dynamic model
  python main.py                      # Default to kinematic model
        """
    )
    parser.add_argument('--model', '-m', 
                       choices=['kinematic', 'dynamic'],
                       default='kinematic',
                       help='Vehicle model to use (default: kinematic)')
    
    args = parser.parse_args()
    
    try:
        main(model_type=args.model)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)