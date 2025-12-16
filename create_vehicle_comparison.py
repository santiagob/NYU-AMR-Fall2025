"""
Create a comparison showing the vehicle at different steering angles
"""
import numpy as np
import matplotlib.pyplot as plt
from models.vehicle_dynamics import VehicleParams
import sys
sys.path.insert(0, '/Users/santiagobernheim/Documents/Projects/NYU/NYU-AMR-Fall2025/Ackerman_Simulator')
from visualize_scenario import get_car_body_vertices, get_wheel_vertices

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

steering_angles = [0, 30, -30]  # degrees
titles = ['Straight', 'Left Turn (30°)', 'Right Turn (-30°)']

for ax, steer_deg, title in zip(axes, steering_angles, titles):
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 3)
    
    x, y, yaw = 0, 0, np.pi/2  # Pointing up
    steer_rad = np.radians(steer_deg)
    
    # Draw vehicle body
    car_x, car_y = get_car_body_vertices(x, y, yaw, VehicleParams.WHEELBASE, VehicleParams.TRACK_WIDTH)
    ax.plot(car_x, car_y, 'k-', linewidth=3)
    ax.fill(car_x, car_y, color='lightblue', alpha=0.5)
    
    # Draw wheels
    wheel_vertices = get_wheel_vertices(x, y, yaw, steer_rad, VehicleParams.WHEELBASE, VehicleParams.TRACK_WIDTH)
    wheel_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
    for i, (wx, wy) in enumerate(wheel_vertices):
        ax.plot(wx, wy, 'k-', linewidth=2.5)
        ax.fill(wx, wy, color='gray', alpha=0.8)
    
    # Add velocity arrow
    arrow_len = 2.0
    dx = arrow_len * np.cos(yaw)
    dy = arrow_len * np.sin(yaw)
    ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.4, 
             fc='green', ec='darkgreen', linewidth=2, alpha=0.7)
    
    # Mark rear axle (center of rotation)
    ax.plot(x, y, 'ro', markersize=8, label='Rear Axle (CG)')
    
    ax.set_title(f'{title}\nSteering: {steer_deg}°', fontsize=12, fontweight='bold')
    ax.set_xlabel('X [m]')
    if ax == axes[0]:
        ax.set_ylabel('Y [m]')
    ax.legend(loc='lower right', fontsize=8)

fig.suptitle('Ackerman Vehicle Visualization with Dynamic Bicycle Model\n' + 
             f'Wheelbase: {VehicleParams.WHEELBASE}m, Track Width: {VehicleParams.TRACK_WIDTH}m',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('vehicle_comparison.png', dpi=200)
print("✓ Vehicle comparison saved to vehicle_comparison.png")
plt.close()
