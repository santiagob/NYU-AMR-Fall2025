"""Quick test of the vehicle visualization functions"""
import numpy as np
import matplotlib.pyplot as plt
from models.vehicle_dynamics import VehicleParams

# Import the new functions
import sys
sys.path.insert(0, '/Users/santiagobernheim/Documents/Projects/NYU/NYU-AMR-Fall2025/Ackerman_Simulator')
from visualize_scenario import get_car_body_vertices, get_wheel_vertices

# Test vehicle at origin
x, y, yaw = 0, 0, np.pi/4  # 45 degrees
steer_rad = np.radians(30)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.grid(True)

# Draw vehicle body
car_x, car_y = get_car_body_vertices(x, y, yaw, VehicleParams.WHEELBASE, VehicleParams.TRACK_WIDTH)
ax.plot(car_x, car_y, 'k-', linewidth=3, label='Vehicle Body')

# Draw wheels
wheel_vertices = get_wheel_vertices(x, y, yaw, steer_rad, VehicleParams.WHEELBASE, VehicleParams.TRACK_WIDTH)
for i, (wx, wy) in enumerate(wheel_vertices):
    ax.plot(wx, wy, 'k-', linewidth=2, label=f'Wheel {i+1}' if i == 0 else '')

ax.legend()
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Vehicle Visualization Test')

plt.savefig('vehicle_test.png', dpi=150)
print("✓ Test image saved to vehicle_test.png")
plt.close()
