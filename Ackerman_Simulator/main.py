import numpy as np
import matplotlib.pyplot as plt
import math
from models.vehicle_dynamics import DynamicBicycleModel, VehicleParams
from planners.a_star import AStarPlanner
from controllers.stanley import StanleyController, PIDController

def main():
    # --- 1. SETUP SCENARIO ---
    print("Initializing Simulation...")
    
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
    
    # --- 3. SIMULATION LOOP ---
    vehicle = DynamicBicycleModel(x=sx, y=sy, yaw=np.radians(45))
    # Reduced k_gain for stability with dynamic model, increased k_soft for low-speed handling
    stanley = StanleyController(k_gain=0.8, k_soft=2.0)
    # Reduced PID gains for smoother speed control
    speed_pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)
    
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
        steer_cmd = np.clip(steer_cmd, -VehicleParams.MAX_STEER, VehicleParams.MAX_STEER)
        
        # F. Speed Control with PID
        current_speed = np.hypot(vehicle.state[3], vehicle.state[4])
        throttle = speed_pid.compute(target_speed, current_speed, dt)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # G. Update Vehicle Dynamics
        vehicle.update(throttle, steer_cmd, dt)
        
        # H. Save History for plotting and analysis
        hx.append(vehicle.state[0])
        hy.append(vehicle.state[1])
        hyaw.append(vehicle.state[2])  # Added for animation
        hv.append(current_speed)
        hsteer.append(steer_cmd)
        hcte.append(cte)

    # --- 4. VISUALIZATION / RESULTS ---
    print("Plotting Results...")
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Trajectory
    plt.subplot(1, 2, 1)
    plt.plot(ox, oy, ".k", label="Obstacles")
    plt.plot(path_x, path_y, "--r", label="A* Path")
    plt.plot(hx, hy, "-b", label="Actual Trajectory", linewidth=2)
    plt.grid(True)
    plt.legend()
    plt.title("Motion Planning & Path Tracking")
    plt.axis("equal")

    # Plot 2: Performance Metrics (Stability Analysis)
    plt.subplot(1, 2, 2)
    plt.plot(hcte, label="Cross Track Error [m]")
    plt.plot(np.degrees(hsteer), label="Steering [deg]")
    plt.axhline(0, color='k', linestyle='--')
    plt.grid(True)
    plt.legend()
    plt.title("Control Performance (Stability)")
    plt.xlabel("Time Steps")
    
    plt.tight_layout()
    plt.show()

    # ... (End of simulation loop) ...

    # --- 4. VISUALIZATION / RESULTS ---
    print("Generating Results with ploter....")
    
    from utils.plotting import Plotter
    
    # Initialize Plotter
    plotter = Plotter(ox, oy, grid_size)
    
    # 1. Show Static Results (For your "Results" Slide)
    plotter.plot_results(path_x, path_y, hx, hy, hsteer, hcte, hv)
    
    # 2. Run Animation (For your "System Modeling" or Demo Slide)
    # Uncomment to see the animated vehicle following the path
    # plotter.animate_run(path_x, path_y, hx, hy, hyaw, hsteer)

if __name__ == "__main__":
    main()