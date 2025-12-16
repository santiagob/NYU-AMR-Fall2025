import numpy as np
import matplotlib.pyplot as plt
import math
from models.vehicle_dynamics import DynamicBicycleModel, VehicleParams
from planners.a_star import AStarPlanner
from controllers.stanley import StanleyController, PIDController

# Setup scenario
ox, oy = [], []
for i in range(60):
    ox.append(i); oy.append(0.0)
    ox.append(i); oy.append(60.0)
for i in range(60):
    ox.append(0.0); oy.append(i)
    ox.append(60.0); oy.append(i)
for i in range(10, 20):
    ox.append(i); oy.append(30.0)
for i in range(30, 40):
    ox.append(40.0); oy.append(i)

sx, sy = 5.0, 5.0
gx, gy = 50.0, 50.0

planner = AStarPlanner(ox, oy, 1.0, 2.0)
path = planner.planning(sx, sy, gx, gy)
path_x, path_y = path[:, 0], path[:, 1]

# Initialize
vehicle = DynamicBicycleModel(x=sx, y=sy, yaw=np.radians(45))
stanley = StanleyController(k_gain=0.8, k_soft=2.0)
speed_pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)

hx, hy, hv, hsteer, hcte, hyaw = [], [], [], [], [], []
target_speed = 5.0
dt = 0.1
max_time = 50.0
lookahead_distance = 3.0

print(f"Initial yaw: {np.degrees(vehicle.state[2]):.1f} deg")
print(f"Path initial direction: {np.degrees(np.arctan2(path_y[1]-path_y[0], path_x[1]-path_x[0])):.1f} deg")
print()

step = 0
for t in np.arange(0, max_time, dt):
    dists = np.hypot(vehicle.state[0] - path_x, vehicle.state[1] - path_y)
    nearest_idx = np.argmin(dists)
    
    idx = nearest_idx
    for i in range(nearest_idx, len(path_x)):
        dist_to_point = np.hypot(vehicle.state[0] - path_x[i], 
                                 vehicle.state[1] - path_y[i])
        if dist_to_point >= lookahead_distance:
            idx = i
            break
    idx = min(idx, len(path_x) - 1)
    
    dist_to_goal = np.hypot(vehicle.state[0] - gx, vehicle.state[1] - gy)
    if dist_to_goal < 2.0:
        print(f"Goal reached at t={t:.1f}s")
        break
    
    steer_cmd, cte = stanley.compute(vehicle.state, path_x, path_y, idx)
    steer_cmd = np.clip(steer_cmd, -VehicleParams.MAX_STEER, VehicleParams.MAX_STEER)
    
    current_speed = np.hypot(vehicle.state[3], vehicle.state[4])
    throttle = speed_pid.compute(target_speed, current_speed, dt)
    throttle = np.clip(throttle, -1.0, 1.0)
    
    if step % 50 == 0:
        print(f"t={t:.1f}s: pos=({vehicle.state[0]:.1f},{vehicle.state[1]:.1f}) "
              f"yaw={np.degrees(vehicle.state[2]):.1f}° "
              f"v={current_speed:.1f} m/s "
              f"target_idx={idx} cte={cte:.2f}m steer={np.degrees(steer_cmd):.1f}°")
    
    vehicle.update(throttle, steer_cmd, dt)
    
    hx.append(vehicle.state[0])
    hy.append(vehicle.state[1])
    hyaw.append(vehicle.state[2])
    hv.append(current_speed)
    hsteer.append(steer_cmd)
    hcte.append(cte)
    step += 1

print(f"\nFinal position: ({vehicle.state[0]:.1f}, {vehicle.state[1]:.1f})")
print(f"Goal position: ({gx:.1f}, {gy:.1f})")
print(f"Final error: {np.hypot(vehicle.state[0] - gx, vehicle.state[1] - gy):.1f}m")
print(f"Mean CTE: {np.mean(np.abs(hcte)):.2f}m")
print(f"Max CTE: {np.max(np.abs(hcte)):.2f}m")

# Plot
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(ox, oy, ".k", markersize=2, label="Obstacles")
plt.plot(path_x, path_y, "--r", linewidth=2, label="A* Path")
plt.plot(hx, hy, "-b", linewidth=2, label="Actual Trajectory")
plt.plot([sx], [sy], "og", markersize=10, label="Start")
plt.plot([gx], [gy], "xr", markersize=15, label="Goal")
plt.grid(True)
plt.legend()
plt.title("Trajectory Comparison")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.plot(hcte, 'r', label="Cross-Track Error")
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Time Steps")
plt.ylabel("Error [m]")
plt.title("Tracking Performance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
