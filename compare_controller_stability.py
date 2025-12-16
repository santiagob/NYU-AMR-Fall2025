#!/usr/bin/env python3
"""
Controller Stability Comparison
================================

Runs identical scenario with all three controllers and compares:
- Cross-track error (CTE) over time
- Steering angle smoothness
- Lateral velocity behavior
- Yaw rate profile
- Speed tracking

Generates side-by-side comparison plots for presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from models.model_factory import create_vehicle, get_vehicle_params
from controllers.pure_pursuit import PurePursuitController
from controllers.stanley import StanleyController, PIDController
from controllers.lqr import LQRController
from planners.a_star import AStarPlanner
from scipy.interpolate import CubicSpline


def smooth_path(path_x, path_y, num_points=None):
    """Smooth path using cubic spline interpolation"""
    if len(path_x) < 3:
        return path_x, path_y
    
    if num_points is None:
        num_points = len(path_x) * 3
    
    try:
        t_original = np.linspace(0, 1, len(path_x))
        cs_x = CubicSpline(t_original, path_x)
        cs_y = CubicSpline(t_original, path_y)
        
        t_smooth = np.linspace(0, 1, num_points)
        smooth_x = cs_x(t_smooth)
        smooth_y = cs_y(t_smooth)
        
        return smooth_x, smooth_y
    except:
        return path_x, path_y


def create_test_scenario():
    """Create a challenging test scenario with curves and straight sections"""
    # Create obstacles
    ox, oy = [], []
    map_size = 60
    
    # Borders
    for i in range(map_size):
        ox.append(i)
        oy.append(0.0)
        ox.append(i)
        oy.append(map_size)
    for i in range(map_size):
        ox.append(0.0)
        oy.append(i)
        ox.append(map_size)
        oy.append(i)
    
    # Internal obstacles creating a challenging path
    for i in range(15, 30):
        ox.append(i)
        oy.append(20.0)
    for i in range(30, 45):
        ox.append(40.0)
        oy.append(i)
    
    # Plan path
    sx, sy = 5.0, 5.0
    gx, gy = 50.0, 50.0
    
    planner = AStarPlanner(ox, oy, 1.0, 2.0)
    path = planner.planning(sx, sy, gx, gy)
    
    if path is None:
        raise RuntimeError("Path planning failed!")
    
    # Smooth path
    path_x, path_y = smooth_path(path[:, 0], path[:, 1])
    
    return path_x, path_y, sx, sy, gx, gy, ox, oy


def run_controller_comparison(model_type="dynamic", target_speed=5.0, duration=60.0):
    """Run all three controllers on the same scenario"""
    
    print(f"\n{'='*70}")
    print(f"Controller Stability Comparison - {model_type.upper()} Model")
    print(f"{'='*70}")
    print(f"Note: Stanley works best when tightly integrated with trajectory planning")
    print(f"      (see visualize_scenario.py for full Stanley integration).")
    print(f"{'='*70}\n")
    
    # Create scenario
    print("Creating test scenario...")
    path_x, path_y, sx, sy, gx, gy, ox, oy = create_test_scenario()
    print(f"Path length: {len(path_x)} points")
    
    # Controller tuning from visualize_scenario.py (proven to work well)
    controller_tuning = {
        'kinematic': {
            'stanley_k': 0.3,      # Reduced from 0.5 for stability
            'stanley_soft': 3.0,   # Increased from 2.0 for smoothness
        },
        'dynamic': {
            'stanley_k': 0.6,      # Reduced from 1.2 for stability
            'stanley_soft': 2.5,   # Increased from 1.5 for smoothness
        }
    }
    
    # Initialize controllers with tuning that works reliably
    # Note: Stanley works excellently in visualize_scenario.py where it's tightly
    # integrated with the lookahead and trajectory system. For comparison, we use
    # Pure Pursuit and LQR which are more robust in isolation.
    
    controllers = {
        'Pure Pursuit': PurePursuitController(
            lookahead_base=3.0,
            lookahead_gain=0.2,
            min_lookahead=2.0,
            max_lookahead=8.0
        ),
        'Stanley': StanleyController(
            k_gain=controller_tuning[model_type]['stanley_k'],
            k_soft=controller_tuning[model_type]['stanley_soft'],
            error_recovery=False  # Disabled for stable comparison
        ),
        'LQR': LQRController(
            q_weights=[2.5, 0.6, 4.0, 0.6],
            r_weight=1.0,
            dt=0.1
        )
    }
    
    # Run each controller
    results = {}
    
    for controller_name, controller in controllers.items():
        print(f"\n--- Running {controller_name} Controller ---")
        
        # Initialize vehicle
        initial_yaw = np.arctan2(path_y[1] - path_y[0], path_x[1] - path_x[0])
        vehicle = create_vehicle(model_type, x=sx, y=sy, yaw=initial_yaw)
        
        # Initialize speed PID
        kp = 0.5 if model_type == "kinematic" else 0.3
        ki = 0.05 if model_type == "kinematic" else 0.02
        kd = 0.1 if model_type == "kinematic" else 0.05
        speed_pid = PIDController(Kp=kp, Ki=ki, Kd=kd)
        
        # Simulation parameters
        dt = 0.1
        max_steps = int(duration / dt)
        
        # History tracking
        history = {
            't': [],
            'x': [],
            'y': [],
            'yaw': [],
            'vx': [],
            'vy': [],
            'r': [],
            'steer': [],
            'cte': [],
            'speed': [],
            'throttle': []
        }
        
        # Simulation loop
        goal_reached = False
        for step in range(max_steps):
            t = step * dt
            
            # Check goal
            dist_to_goal = np.hypot(vehicle.x - gx, vehicle.y - gy)
            if dist_to_goal < 2.5:
                print(f"  ✓ Goal reached at t={t:.1f}s")
                goal_reached = True
                break
            
            # Find lookahead point
            distances = np.hypot(vehicle.x - path_x, vehicle.y - path_y)
            nearest_idx = np.argmin(distances)
            
            current_speed = np.hypot(vehicle.vx, vehicle.vy)
            # Use visualize_scenario's lookahead tuning which works well
            if model_type == 'kinematic':
                lookahead_base = 2.5
                lookahead_gain = 0.3
            else:
                lookahead_base = 2.0
                lookahead_gain = 0.5
            lookahead = lookahead_base + lookahead_gain * current_speed
            lookahead = np.clip(lookahead, 1.0, 5.0)
            
            target_idx = nearest_idx
            for i in range(nearest_idx, len(path_x)):
                if np.hypot(vehicle.x - path_x[i], vehicle.y - path_y[i]) >= lookahead:
                    target_idx = i
                    break
            # Ensure target_idx is within bounds and not past goal
            target_idx = min(target_idx, len(path_x) - 1)
            target_idx = max(target_idx, 0)
            
            # Compute control (use 4-element state like visualize_scenario does)
            steer, cte = controller.compute([vehicle.x, vehicle.y, vehicle.yaw, vehicle.vx], path_x, path_y, target_idx)
            
            # Get vehicle params
            params = get_vehicle_params(model_type)
            steer = np.clip(steer, -params.MAX_STEER, params.MAX_STEER)
            
            # Steering rate limiting for stability (especially for Stanley)
            steer_rate_limit = 0.8 if model_type == "dynamic" else 0.6
            max_delta = steer_rate_limit * dt
            if step > 0:
                steer = np.clip(steer, history['steer'][-1] - max_delta, history['steer'][-1] + max_delta)
            
            # Speed control with cornering adjustment
            if abs(steer) > np.radians(30):
                speed_factor = 1.0 - 0.5 * (abs(steer) - np.radians(30)) / (params.MAX_STEER - np.radians(30))
                target = target_speed * np.clip(speed_factor, 0.5, 1.0)
            else:
                target = target_speed
            
            throttle = speed_pid.compute(target, current_speed, dt)
            throttle = np.clip(throttle, -1.0, 1.0)
            
            # Record history
            history['t'].append(t)
            history['x'].append(vehicle.x)
            history['y'].append(vehicle.y)
            history['yaw'].append(vehicle.yaw)
            history['vx'].append(vehicle.vx)
            history['vy'].append(vehicle.vy)
            history['r'].append(vehicle.r)
            history['steer'].append(steer)
            history['cte'].append(cte)
            history['speed'].append(current_speed)
            history['throttle'].append(throttle)
            
            # Update vehicle
            vehicle.update(throttle, steer, dt)
        
        # Calculate statistics
        stats = {
            'completion_time': history['t'][-1] if goal_reached else duration,
            'goal_reached': goal_reached,
            'avg_cte': np.mean(np.abs(history['cte'])),
            'max_cte': np.max(np.abs(history['cte'])),
            'rms_cte': np.sqrt(np.mean(np.array(history['cte'])**2)),
            'avg_steer': np.mean(np.abs(history['steer'])),
            'max_steer': np.max(np.abs(history['steer'])),
            'steer_rate': np.mean(np.abs(np.diff(history['steer']))) / dt,
            'avg_speed': np.mean(history['speed']),
            'speed_std': np.std(history['speed']),
        }
        
        results[controller_name] = {
            'history': history,
            'stats': stats
        }
        
        # Print stats
        print(f"  Completion time: {stats['completion_time']:.1f}s")
        print(f"  Average CTE: {stats['avg_cte']:.3f}m")
        print(f"  Max CTE: {stats['max_cte']:.3f}m")
        print(f"  RMS CTE: {stats['rms_cte']:.3f}m")
        print(f"  Max steering: {np.degrees(stats['max_steer']):.1f}°")
        print(f"  Steering rate: {np.degrees(stats['steer_rate']):.1f}°/s")
    
    return results, path_x, path_y, ox, oy


def plot_stability_comparison(results, path_x, path_y, ox, oy, save_path=None):
    """Generate comprehensive stability comparison plots"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Define colors for each controller - generate dynamically
    color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']  # Blue, Purple, Orange, Green
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(results.keys())}
    
    # Row 1: Trajectories
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(ox, oy, ".k", markersize=2, alpha=0.3, label='Obstacles')
    ax1.plot(path_x, path_y, '--', color='gray', linewidth=1.5, alpha=0.5, label='Planned Path')
    for name, data in results.items():
        h = data['history']
        ax1.plot(h['x'], h['y'], linewidth=2, color=colors[name], label=name, alpha=0.8)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Trajectories', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Row 1, Col 2: Cross-Track Error Over Time
    ax2 = plt.subplot(3, 3, 2)
    for name, data in results.items():
        h = data['history']
        ax2.plot(h['t'], h['cte'], linewidth=2, color=colors[name], label=name, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Cross-Track Error [m]')
    ax2.set_title('Cross-Track Error (Stability Indicator)', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Row 1, Col 3: CTE Distribution (Box Plot)
    ax3 = plt.subplot(3, 3, 3)
    cte_data = [np.array(results[name]['history']['cte']) for name in results.keys()]
    bp = ax3.boxplot(cte_data, labels=list(results.keys()), patch_artist=True)
    for patch, name in zip(bp['boxes'], results.keys()):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax3.set_ylabel('Cross-Track Error [m]')
    ax3.set_title('CTE Distribution', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=15)
    
    # Row 2, Col 1: Steering Angle Over Time
    ax4 = plt.subplot(3, 3, 4)
    for name, data in results.items():
        h = data['history']
        ax4.plot(h['t'], np.degrees(h['steer']), linewidth=2, color=colors[name], label=name, alpha=0.8)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Steering Angle [deg]')
    ax4.set_title('Steering Command (Control Smoothness)', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Row 2, Col 2: Steering Rate (Derivative)
    ax5 = plt.subplot(3, 3, 5)
    for name, data in results.items():
        h = data['history']
        steer_rate = np.degrees(np.diff(h['steer']) / np.diff(h['t']))
        ax5.plot(h['t'][1:], steer_rate, linewidth=1.5, color=colors[name], label=name, alpha=0.8)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Steering Rate [deg/s]')
    ax5.set_title('Steering Rate (Smoothness)', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Row 2, Col 3: Lateral Velocity
    ax6 = plt.subplot(3, 3, 6)
    for name, data in results.items():
        h = data['history']
        ax6.plot(h['t'], h['vy'], linewidth=2, color=colors[name], label=name, alpha=0.8)
    ax6.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Lateral Velocity [m/s]')
    ax6.set_title('Lateral Velocity (Side Slip)', fontweight='bold', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Row 3, Col 1: Speed Profile
    ax7 = plt.subplot(3, 3, 7)
    for name, data in results.items():
        h = data['history']
        ax7.plot(h['t'], h['speed'], linewidth=2, color=colors[name], label=name, alpha=0.8)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Speed [m/s]')
    ax7.set_title('Speed Profile', fontweight='bold', fontsize=12)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Row 3, Col 2: Yaw Rate
    ax8 = plt.subplot(3, 3, 8)
    for name, data in results.items():
        h = data['history']
        ax8.plot(h['t'], np.degrees(h['r']), linewidth=2, color=colors[name], label=name, alpha=0.8)
    ax8.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Yaw Rate [deg/s]')
    ax8.set_title('Yaw Rate', fontweight='bold', fontsize=12)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # Row 3, Col 3: Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create statistics table
    metrics = ['Avg CTE [m]', 'RMS CTE [m]', 'Max CTE [m]', 
               'Max Steer [deg]', 'Steer Rate [deg/s]', 'Time [s]']
    
    table_data = []
    for name in results.keys():
        stats = results[name]['stats']
        row = [
            f"{stats['avg_cte']:.3f}",
            f"{stats['rms_cte']:.3f}",
            f"{stats['max_cte']:.3f}",
            f"{np.degrees(stats['max_steer']):.1f}",
            f"{np.degrees(stats['steer_rate']):.1f}",
            f"{stats['completion_time']:.1f}"
        ]
        table_data.append(row)
    
    # Transpose for table display
    table = ax9.table(cellText=np.array(table_data).T, 
                     rowLabels=metrics,
                     colLabels=list(results.keys()),
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color table headers
    for i, name in enumerate(results.keys()):
        table[(0, i)].set_facecolor(colors[name])
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Performance Metrics', fontweight='bold', fontsize=12, pad=20)
    
    plt.suptitle('Controller Stability Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")
    
    plt.show()


def print_summary_table(results):
    """Print a comprehensive summary table"""
    print(f"\n{'='*70}")
    print("CONTROLLER STABILITY COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    # Header
    print(f"{'Metric':<30} {'Pure Pursuit':<15} {'Stanley':<15} {'LQR':<15}")
    print("-" * 75)
    
    # Metrics
    metrics = [
        ('Completion Time [s]', 'completion_time', '.1f'),
        ('Goal Reached', 'goal_reached', ''),
        ('Average CTE [m]', 'avg_cte', '.3f'),
        ('RMS CTE [m]', 'rms_cte', '.3f'),
        ('Max CTE [m]', 'max_cte', '.3f'),
        ('Avg Steering [deg]', 'avg_steer', '.1f', True),
        ('Max Steering [deg]', 'max_steer', '.1f', True),
        ('Steering Rate [deg/s]', 'steer_rate', '.1f', True),
        ('Avg Speed [m/s]', 'avg_speed', '.2f'),
        ('Speed Std Dev [m/s]', 'speed_std', '.3f'),
    ]
    
    for metric_info in metrics:
        name = metric_info[0]
        key = metric_info[1]
        fmt = metric_info[2]
        convert_deg = len(metric_info) > 3 and metric_info[3]
        
        row = f"{name:<30}"
        for controller in results.keys():
            value = results[controller]['stats'][key]
            if key == 'goal_reached':
                row += f"{'✓' if value else '✗':<15}"
            elif convert_deg:
                row += f"{np.degrees(value):{fmt}}<15"
            else:
                row += f"{value:{fmt}}<15"
        print(row)
    
    print("-" * 75)
    
    # Winner analysis
    print("\n" + "="*70)
    print("STABILITY ANALYSIS")
    print("="*70 + "\n")
    
    best_cte = min(results.items(), key=lambda x: x[1]['stats']['avg_cte'])
    best_smooth = min(results.items(), key=lambda x: x[1]['stats']['steer_rate'])
    best_time = min(results.items(), key=lambda x: x[1]['stats']['completion_time'])
    
    print(f"🏆 Best Tracking Accuracy:  {best_cte[0]} (Avg CTE: {best_cte[1]['stats']['avg_cte']:.3f}m)")
    print(f"🏆 Smoothest Control:       {best_smooth[0]} (Rate: {np.degrees(best_smooth[1]['stats']['steer_rate']):.1f}°/s)")
    print(f"🏆 Fastest Completion:      {best_time[0]} ({best_time[1]['stats']['completion_time']:.1f}s)")
    
    print("\n" + "="*70)


def save_individual_controller_plots(results, path_x, path_y, ox, oy, output_dir):
    """Save individual plots for each controller"""
    import os
    
    print(f"\nGenerating individual controller plots...")
    
    # Color palette for consistency
    color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(results.keys())}
    
    for controller_name, data in results.items():
        h = data['history']
        
        # Create figure with 4 subplots per controller
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{controller_name} Controller - Detailed Performance', 
                    fontsize=14, fontweight='bold')
        color = colors[controller_name]
        
        # Plot 1: Trajectory
        ax = axes[0, 0]
        ax.plot(ox, oy, ".k", markersize=2, alpha=0.3)
        ax.plot(path_x, path_y, '--', color='gray', linewidth=1.5, alpha=0.5, label='Planned Path')
        ax.plot(h['x'], h['y'], linewidth=2.5, color=color, label=controller_name)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Trajectory', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 2: Cross-Track Error
        ax = axes[0, 1]
        ax.plot(h['t'], h['cte'], linewidth=2.5, color=color)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
        ax.fill_between(h['t'], 0, h['cte'], alpha=0.3, color=color)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Cross-Track Error [m]')
        ax.set_title('Cross-Track Error (CTE)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Steering Angle
        ax = axes[1, 0]
        ax.plot(h['t'], np.degrees(h['steer']), linewidth=2.5, color=color)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
        ax.fill_between(h['t'], 0, np.degrees(h['steer']), alpha=0.3, color=color)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Steering Angle [deg]')
        ax.set_title('Steering Command', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Speed Profile
        ax = axes[1, 1]
        ax.plot(h['t'], h['speed'], linewidth=2.5, color=color, label='Actual Speed')
        ax.axhline(y=5.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Target Speed')
        ax.fill_between(h['t'], 0, h['speed'], alpha=0.2, color=color)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Speed [m/s]')
        ax.set_title('Speed Profile', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filename = os.path.join(output_dir, f'{controller_name.lower().replace(" ", "_")}_performance.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()


def save_metrics_table(results, output_dir):
    """Save metrics table as an image"""
    import os
    
    print(f"Generating metrics comparison table...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Prepare metrics data
    metrics = [
        ('Completion Time [s]', 'completion_time', '.1f', False),
        ('Goal Reached', 'goal_reached', '', False),
        ('Average CTE [m]', 'avg_cte', '.3f', False),
        ('RMS CTE [m]', 'rms_cte', '.3f', False),
        ('Max CTE [m]', 'max_cte', '.3f', False),
        ('Avg Steering [deg]', 'avg_steer', '.1f', True),
        ('Max Steering [deg]', 'max_steer', '.1f', True),
        ('Steering Rate [deg/s]', 'steer_rate', '.1f', True),
        ('Avg Speed [m/s]', 'avg_speed', '.2f', False),
        ('Speed Std Dev [m/s]', 'speed_std', '.3f', False),
    ]
    
    table_data = []
    for metric_name, key, fmt, convert_deg in metrics:
        row = [metric_name]
        for controller in results.keys():
            value = results[controller]['stats'][key]
            if key == 'goal_reached':
                row.append('✓' if value else '✗')
            elif convert_deg:
                row.append(f"{np.degrees(value):{fmt}}")
            else:
                row.append(f"{value:{fmt}}")
        table_data.append(row)
    
    # Create table
    columns = ['Metric'] + list(results.keys())
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center',
                    loc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Controller Performance Metrics Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    
    filename = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def save_comparison_subplots_separately(results, path_x, path_y, ox, oy, output_dir):
    """Save each comparison subplot as individual plots for presentation"""
    import os
    
    print(f"\nGenerating individual comparison plots...")
    
    # Define colors for consistency
    color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(results.keys())}
    
    # 1. Trajectories
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(ox, oy, ".k", markersize=2, alpha=0.3, label='Obstacles')
    ax.plot(path_x, path_y, '--', color='gray', linewidth=1.5, alpha=0.5, label='Planned Path')
    for name, data in results.items():
        h = data['history']
        ax.plot(h['x'], h['y'], linewidth=2.5, color=colors[name], label=name, alpha=0.8)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('Controller Trajectories Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    filename = os.path.join(output_dir, '01_trajectories.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    # 2. CTE Over Time
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, data in results.items():
        h = data['history']
        ax.plot(h['t'], h['cte'], linewidth=2.5, color=colors[name], label=name, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Cross-Track Error [m]', fontsize=12)
    ax.set_title('Cross-Track Error Over Time', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, '02_cte_over_time.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    # 3. CTE Distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    cte_data = [np.array(results[name]['history']['cte']) for name in results.keys()]
    bp = ax.boxplot(cte_data, labels=list(results.keys()), patch_artist=True, widths=0.6)
    for patch, name in zip(bp['boxes'], results.keys()):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(2)
    for cap in bp['caps']:
        cap.set_linewidth(2)
    ax.set_ylabel('Cross-Track Error [m]', fontsize=12)
    ax.set_title('CTE Distribution (Box Plot)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=11)
    filename = os.path.join(output_dir, '03_cte_distribution.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    # 4. Steering Angle Over Time
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, data in results.items():
        h = data['history']
        ax.plot(h['t'], np.degrees(h['steer']), linewidth=2.5, color=colors[name], label=name, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Steering Angle [deg]', fontsize=12)
    ax.set_title('Steering Command Over Time', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, '04_steering_angle.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    # 5. Steering Rate
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, data in results.items():
        h = data['history']
        steer_rate = np.degrees(np.diff(h['steer']) / np.diff(h['t']))
        ax.plot(h['t'][1:], steer_rate, linewidth=2, color=colors[name], label=name, alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Steering Rate [deg/s]', fontsize=12)
    ax.set_title('Steering Rate (Control Smoothness)', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, '05_steering_rate.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    # 6. Lateral Velocity
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, data in results.items():
        h = data['history']
        ax.plot(h['t'], h['vy'], linewidth=2.5, color=colors[name], label=name, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Lateral Velocity [m/s]', fontsize=12)
    ax.set_title('Lateral Velocity (Side Slip Behavior)', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, '06_lateral_velocity.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    # 7. Speed Profile
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, data in results.items():
        h = data['history']
        ax.plot(h['t'], h['speed'], linewidth=2.5, color=colors[name], label=name, alpha=0.8)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Speed [m/s]', fontsize=12)
    ax.set_title('Speed Profile Comparison', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, '07_speed_profile.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    # 8. Yaw Rate
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, data in results.items():
        h = data['history']
        ax.plot(h['t'], np.degrees(h['r']), linewidth=2.5, color=colors[name], label=name, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Yaw Rate [deg/s]', fontsize=12)
    ax.set_title('Yaw Rate (Rotational Dynamics)', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    filename = os.path.join(output_dir, '08_yaw_rate.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()


def main():
    """Main entry point"""
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Compare controller stability')
    parser.add_argument('--model', type=str, default='dynamic', 
                       choices=['kinematic', 'dynamic'],
                       help='Vehicle model type (default: dynamic)')
    parser.add_argument('--speed', type=float, default=5.0,
                       help='Target speed in m/s (default: 5.0)')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Max simulation duration in seconds (default: 60)')
    parser.add_argument('--name', type=str, default=None,
                       help='Name for the comparison folder (default: timestamp)')
    parser.add_argument('--output-dir', type=str, default='./output/controller_comparison',
                       help='Output directory for plots (default: ./output/controller_comparison)')
    
    args = parser.parse_args()
    
    # Create timestamped subfolder
    if args.name:
        folder_name = args.name
    else:
        folder_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    full_output_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"\n📁 Output folder: {full_output_dir}\n")
    
    # Run comparison
    results, path_x, path_y, ox, oy = run_controller_comparison(
        model_type=args.model,
        target_speed=args.speed,
        duration=args.duration
    )
    
    # Print summary
    print_summary_table(results)
    
    # Generate and save plots
    print(f"\n{'='*70}")
    print("SAVING PLOTS")
    print(f"{'='*70}")
    
    # Main comparison plot
    print(f"\nGenerating main comparison plot...")
    comparison_path = os.path.join(full_output_dir, '00_main_comparison.png')
    plot_stability_comparison(results, path_x, path_y, ox, oy, comparison_path)
    
    # Individual controller plots
    save_individual_controller_plots(results, path_x, path_y, ox, oy, full_output_dir)
    
    # Individual comparison subplots
    save_comparison_subplots_separately(results, path_x, path_y, ox, oy, full_output_dir)
    
    # Metrics table
    save_metrics_table(results, full_output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ All plots saved to: {full_output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
