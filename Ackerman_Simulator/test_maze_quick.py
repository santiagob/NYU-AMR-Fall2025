"""Quick test of improved maze scenario"""
import numpy as np
from models.vehicle_dynamics import DynamicBicycleModel, VehicleParams
from planners.a_star import AStarPlanner
from controllers.stanley import StanleyController, PIDController

# Create improved maze
ox, oy = [], []
map_size = 60
for i in range(map_size):
    ox.append(i); oy.append(0.0)
    ox.append(i); oy.append(map_size)
for i in range(map_size):
    ox.append(0.0); oy.append(i)
    ox.append(map_size); oy.append(i)

# Wide gaps version
for i in range(10, 25):  # Gap at 25-35
    ox.append(i); oy.append(20.0)
for i in range(35, 50):  # Gap at 30-35
    ox.append(i); oy.append(40.0)
for i in range(25, 35):  # Short vertical
    ox.append(30.0); oy.append(i)

# Plan first segment
planner = AStarPlanner(ox, oy, 1.0, 2.5)
path1 = planner.planning(5.0, 10.0, 25.0, 30.0)
path2 = planner.planning(25.0, 30.0, 50.0, 50.0)

print("Segment 1:", len(path1), "points")
print("Segment 2:", len(path2), "points")

# Simulate
path = np.vstack([path1, path2[1:]])
path_x, path_y = path[:, 0], path[:, 1]

vehicle = DynamicBicycleModel(x=5.0, y=10.0, yaw=np.arctan2(path_y[1]-path_y[0], path_x[1]-path_x[0]))
stanley = StanleyController(k_gain=1.2, k_soft=1.5)
speed_pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)

waypoints = [(5.0, 10.0), (25.0, 30.0), (50.0, 50.0)]
current_wp = 1

for t in np.arange(0, 60, 0.1):
    # Lookahead
    dists = np.hypot(vehicle.state[0] - path_x, vehicle.state[1] - path_y)
    nearest_idx = np.argmin(dists)
    
    current_speed = np.hypot(vehicle.state[3], vehicle.state[4])
    lookahead = 2.0 + 0.5 * current_speed
    
    idx = nearest_idx
    for i in range(nearest_idx, len(path_x)):
        if np.hypot(vehicle.state[0] - path_x[i], vehicle.state[1] - path_y[i]) >= lookahead:
            idx = i
            break
    idx = min(idx, len(path_x) - 1)
    
    # Check waypoint
    if current_wp < len(waypoints):
        wx, wy = waypoints[current_wp]
        if np.hypot(vehicle.state[0] - wx, vehicle.state[1] - wy) < 2.5:
            print(f"✓ Waypoint {current_wp} reached at t={t:.1f}s")
            current_wp += 1
    
    # Goal check
    if np.hypot(vehicle.state[0] - 50.0, vehicle.state[1] - 50.0) < 2.5:
        print(f"✓ GOAL REACHED at t={t:.1f}s!")
        print(f"Final position: ({vehicle.state[0]:.1f}, {vehicle.state[1]:.1f})")
        break
    
    # Control
    steer, cte = stanley.compute(vehicle.state, path_x, path_y, idx)
    steer = np.clip(steer, -VehicleParams.MAX_STEER, VehicleParams.MAX_STEER)
    
    # Speed reduction on high steering
    if abs(steer) > np.radians(30):
        speed_factor = 1.0 - 0.5 * (abs(steer) - np.radians(30)) / (VehicleParams.MAX_STEER - np.radians(30))
        target = 2.5 * np.clip(speed_factor, 0.5, 1.0)
    else:
        target = 2.5
    
    throttle = speed_pid.compute(target, current_speed, 0.1)
    throttle = np.clip(throttle, -1.0, 1.0)
    
    vehicle.update(throttle, steer, 0.1)
    
    if int(t * 10) % 50 == 0:
        print(f"t={t:.1f}s: pos=({vehicle.state[0]:.1f},{vehicle.state[1]:.1f}) cte={cte:.2f}m steer={np.degrees(steer):.1f}°")

print(f"\nFinal: {current_wp-1}/{len(waypoints)-1} waypoints reached")
