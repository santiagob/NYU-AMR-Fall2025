"""
Interactive Visualization Script
Creates animated videos showing vehicle movement through scenarios
Supports model selection and side-by-side model comparison
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrow, Polygon
import argparse
import sys
from models.model_factory import create_vehicle, get_vehicle_params
from models.vehicle_dynamics import DynamicBicycleModel, VehicleParams
from planners.a_star import AStarPlanner
from controllers.stanley import StanleyController, PIDController
from controllers.lqr import LQRController
from controllers.pure_pursuit import PurePursuitController

# Vehicle visualization parameters (matching Simulator.py)
WHEEL_LENGTH = 0.6
WHEEL_WIDTH = 0.5

def get_car_body_vertices(x, y, yaw, wheelbase, track_width):
    """Returns the coordinates of the car body corners for plotting."""
    body_length = wheelbase + 2 * WHEEL_LENGTH
    body_width = track_width + WHEEL_WIDTH
    rear_overhang = WHEEL_LENGTH
    front_overhang = body_length - wheelbase - rear_overhang
    
    half_width = body_width / 2
    vertices_local = np.array([
        [-rear_overhang, half_width],
        [wheelbase + front_overhang, half_width],
        [wheelbase + front_overhang, -half_width],
        [-rear_overhang, -half_width],
        [-rear_overhang, half_width]
    ])
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    rotated_vertices = (rotation_matrix @ vertices_local.T).T
    global_vertices_x = x + rotated_vertices[:, 0]
    global_vertices_y = y + rotated_vertices[:, 1]
    return global_vertices_x, global_vertices_y

def get_wheel_vertices(x, y, yaw, steer_angle_rad, wheelbase, track_width):
    """Returns the coordinates of the wheels for plotting."""
    wheel_half_len = WHEEL_LENGTH / 2
    wheel_half_width = WHEEL_WIDTH / 2
    half_track = track_width / 2
    wheel_centers_local = {
        'RL': np.array([0, half_track]),
        'RR': np.array([0, -half_track]),
        'FL': np.array([wheelbase, half_track]),
        'FR': np.array([wheelbase, -half_track]),
    }
    delta_fl = delta_fr = 0
    if abs(steer_angle_rad) > 1e-6:
        turn_radius_rear_axle = wheelbase / np.tan(steer_angle_rad)
        delta_fl = np.arctan(wheelbase / (turn_radius_rear_axle - np.sign(steer_angle_rad) * half_track))
        delta_fr = np.arctan(wheelbase / (turn_radius_rear_axle + np.sign(steer_angle_rad) * half_track))
    wheel_angles = {'RL': 0, 'RR': 0, 'FL': delta_fl, 'FR': delta_fr}
    all_wheel_vertices = []
    wheel_vertices_local_template = np.array([
        [-wheel_half_len, wheel_half_width],
        [wheel_half_len, wheel_half_width],
        [wheel_half_len, -wheel_half_width],
        [-wheel_half_len, -wheel_half_width],
        [-wheel_half_len, wheel_half_width]
    ])
    vehicle_rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    for wheel_key in ['FL', 'FR', 'RL', 'RR']:
        center_local = wheel_centers_local[wheel_key]
        wheel_angle = wheel_angles[wheel_key]
        wheel_rotation_matrix = np.array([
            [np.cos(wheel_angle), -np.sin(wheel_angle)],
            [np.sin(wheel_angle), np.cos(wheel_angle)]
        ])
        rotated_wheel_vertices = (wheel_rotation_matrix @ wheel_vertices_local_template.T).T
        wheel_vertices_in_vehicle_frame = rotated_wheel_vertices + center_local
        global_wheel_vertices = (vehicle_rotation_matrix @ wheel_vertices_in_vehicle_frame.T).T + np.array([x, y])
        all_wheel_vertices.append((global_wheel_vertices[:, 0], global_wheel_vertices[:, 1]))
    return all_wheel_vertices

def create_scenario_animation(scenario_name, ox, oy, waypoints, target_speed, 
                              grid_size, robot_radius, output_file='scenario_animation.mp4',
                              use_smoothing=False, model_type='kinematic', compare_models=False,
                              fast_mode=False, controller_type='stanley'):
    """
    Create animated video of vehicle navigating through scenario
    
    Args:
        scenario_name: Name of the scenario
        ox, oy: Obstacle coordinates
        waypoints: List of (x, y) waypoint tuples
        target_speed: Target speed in m/s
        grid_size: A* grid resolution
        robot_radius: Safety buffer
        output_file: Output video filename
        use_smoothing: Whether to smooth the path
        model_type: 'kinematic', 'dynamic', or 'both' for comparison
        compare_models: If True, run both models side-by-side
    """
    print(f"\n{'='*60}")
    print(f"Creating animation for: {scenario_name}")
    print(f"Model: {model_type.upper()} (fast_mode={fast_mode}, controller={controller_type})")
    if compare_models or model_type == 'both':
        print(f"Mode: COMPARISON (Kinematic vs Dynamic)")
    print(f"{'='*60}")
    
    # Plan path
    planner = AStarPlanner(ox, oy, grid_size, robot_radius)
    all_path_x, all_path_y = [], []
    
    for i in range(len(waypoints) - 1):
        sx, sy = waypoints[i]
        gx, gy = waypoints[i + 1]
        path = planner.planning(sx, sy, gx, gy)
        
        if path is None:
            print(f"  ✗ Path planning failed for segment {i+1}!")
            return None
        
        if i == 0:
            all_path_x.extend(path[:, 0])
            all_path_y.extend(path[:, 1])
        else:
            all_path_x.extend(path[1:, 0])
            all_path_y.extend(path[1:, 1])
    
    path_x = np.array(all_path_x)
    path_y = np.array(all_path_y)
    
    print(f"  Path planned: {len(path_x)} points")
    
    # Initialize vehicle
    sx, sy = waypoints[0]
    gx, gy = waypoints[-1]
    initial_yaw = np.arctan2(path_y[1] - path_y[0], path_x[1] - path_x[0])
    
    # Determine which models to use
    if model_type == 'both' or compare_models:
        models_to_use = ['kinematic', 'dynamic']
        title_suffix = " - Model Comparison"
    else:
        models_to_use = [model_type]
        title_suffix = f" - {model_type.capitalize()} Model"
    
    # Create vehicles for each model
    vehicles = {}
    for m_type in models_to_use:
        vehicles[m_type] = create_vehicle(m_type, x=sx, y=sy, yaw=initial_yaw)
    
    # Simulation step
    dt = 0.1

    # Model-specific controller tuning to keep kinematic model stable
    control_tuning = {
        'kinematic': {
            'stanley_k': 0.5,
            'stanley_soft': 2.0,
            'speed_pid': (0.4, 0.02, 0.05),
            'target_speed_scale': 0.9,
            'lookahead_base': 2.5,
            'lookahead_gain': 0.3,
            'steer_rate_limit': 0.8  # rad/s
        },
        'dynamic': {
            'stanley_k': 1.2,
            'stanley_soft': 1.5,
            'speed_pid': (0.5, 0.05, 0.1),
            'target_speed_scale': 1.0,
            'lookahead_base': 2.0,
            'lookahead_gain': 0.5,
            'steer_rate_limit': 1.2
        }
    }

    stanley_controllers = {
        m_type: StanleyController(
            k_gain=control_tuning[m_type]['stanley_k'],
            k_soft=control_tuning[m_type]['stanley_soft']
        ) for m_type in models_to_use
    }
    lqr_controllers = {m_type: LQRController(dt=dt) for m_type in models_to_use}
    pp_controllers = {m_type: PurePursuitController(lookahead_base=control_tuning[m_type]['lookahead_base'],
                                                   lookahead_gain=control_tuning[m_type]['lookahead_gain'],
                                                   min_lookahead=1.5, max_lookahead=8.0)
                     for m_type in models_to_use}
    speed_pids = {
        m_type: PIDController(*control_tuning[m_type]['speed_pid'])
        for m_type in models_to_use
    }
    prev_steers = {m_type: 0.0 for m_type in models_to_use}
    
    # Simulation
    max_time = 60.0  # Limit animation length
    waypoint_tolerance = 2.5

    # Animation/export knobs for speed
    frame_stride = 2 if fast_mode else 1  # Skip every other frame when fast
    export_fps = 10 if fast_mode else 20
    export_bitrate = 1200 if fast_mode else 1800
    
    # Storage for animation (one history per model)
    histories = {}
    for m_type in models_to_use:
        histories[m_type] = {
            'x': [], 'y': [], 'yaw': [], 'v': [], 'steer': [], 'cte': [],
            'target_idx': [], 'time': []
        }
    
    current_waypoint_idx = 1
    
    print(f"  Simulating trajectory...")
    for t in np.arange(0, max_time, dt):
        # Simulate each model independently
        for m_type, vehicle in vehicles.items():
            tuning = control_tuning[m_type]
            # Control logic (same for all models)
            current_speed = np.hypot(vehicle.vx, vehicle.vy)
            lookahead_distance = tuning['lookahead_base'] + tuning['lookahead_gain'] * current_speed
            lookahead_distance = np.clip(lookahead_distance, 1.0, 5.0)
            
            dists = np.hypot(vehicle.x - path_x, vehicle.y - path_y)
            nearest_idx = np.argmin(dists)
            
            idx = nearest_idx
            for i in range(nearest_idx, len(path_x)):
                if np.hypot(vehicle.x - path_x[i], vehicle.y - path_y[i]) >= lookahead_distance:
                    idx = i
                    break
            idx = min(idx, len(path_x) - 1)
            
            # Get parameters for this model
            params = get_vehicle_params(m_type)
            
            # Control
            if controller_type == 'lqr':
                steer, cte = lqr_controllers[m_type].compute([vehicle.x, vehicle.y, vehicle.yaw, vehicle.vx, vehicle.vy, vehicle.r], path_x, path_y, idx)
            elif controller_type == 'pure_pursuit':
                steer, cte = pp_controllers[m_type].compute([vehicle.x, vehicle.y, vehicle.yaw, vehicle.vx], path_x, path_y, idx)
            else:
                steer, cte = stanley_controllers[m_type].compute([vehicle.x, vehicle.y, vehicle.yaw, vehicle.vx], path_x, path_y, idx)
            steer = np.clip(steer, -params.MAX_STEER, params.MAX_STEER)

            # Steering rate limiting for stability (especially kinematic model)
            max_delta = tuning['steer_rate_limit'] * dt
            steer = np.clip(steer, prev_steers[m_type] - max_delta, prev_steers[m_type] + max_delta)
            prev_steers[m_type] = steer
            
            # Speed reduction on high steering
            if abs(steer) > np.radians(30):
                speed_factor = 1.0 - 0.5 * (abs(steer) - np.radians(30)) / (params.MAX_STEER - np.radians(30))
                adaptive_speed = target_speed * np.clip(speed_factor, 0.5, 1.0)
            else:
                adaptive_speed = target_speed * tuning['target_speed_scale']
            
            throttle = speed_pids[m_type].compute(adaptive_speed, current_speed, dt)
            throttle = np.clip(throttle, -1.0, 1.0)
            
            # Update vehicle
            vehicle.update(throttle, steer, dt)
            
            # Record
            histories[m_type]['x'].append(vehicle.x)
            histories[m_type]['y'].append(vehicle.y)
            histories[m_type]['yaw'].append(vehicle.yaw)
            histories[m_type]['v'].append(current_speed)
            histories[m_type]['steer'].append(steer)
            histories[m_type]['cte'].append(cte)
            histories[m_type]['target_idx'].append(idx)
            histories[m_type]['time'].append(t)
        
        # Check goal for first model (same for all since they follow same path)
        vehicle_first = vehicles[models_to_use[0]]
        if current_waypoint_idx < len(waypoints):
            wx, wy = waypoints[current_waypoint_idx]
            if np.hypot(vehicle_first.x - wx, vehicle_first.y - wy) < waypoint_tolerance:
                print(f"  ✓ Waypoint {current_waypoint_idx} reached at t={t:.1f}s")
                current_waypoint_idx += 1
        
        dist_to_goal = np.hypot(vehicle_first.x - gx, vehicle_first.y - gy)
        if dist_to_goal < waypoint_tolerance and current_waypoint_idx >= len(waypoints):
            print(f"  ✓ Goal reached at t={t:.1f}s")
            break
    
    print(f"  Trajectory complete: {len(histories[models_to_use[0]]['x'])} frames")
    
    # Create animation
    print(f"  Generating animation...")
    
    # Determine layout based on comparison mode
    if len(models_to_use) > 1:
        # Comparison mode: side-by-side
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
        ax1_kin = fig.add_subplot(gs[0, 0])
        ax1_dyn = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[:, 1])
        
        # Setup axes for both models
        for ax_idx, (m_type, ax1) in enumerate([(models_to_use[0], ax1_kin), (models_to_use[1], ax1_dyn)]):
            ax1.set_xlim(min(ox) - 5, max(ox) + 5)
            ax1.set_ylim(min(oy) - 5, max(oy) + 5)
            ax1.set_aspect('equal')
            ax1.set_xlabel('X [m]', fontsize=10)
            ax1.set_ylabel('Y [m]', fontsize=10)
            ax1.set_title(f'{m_type.capitalize()} Model', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.plot(ox, oy, '.k', markersize=2, alpha=0.8)
            ax1.plot(path_x, path_y, '--r', linewidth=2, alpha=0.5)
            wp_x = [wp[0] for wp in waypoints]
            wp_y = [wp[1] for wp in waypoints]
            ax1.plot(wp_x[0], wp_y[0], 'go', markersize=12, zorder=10)
            ax1.plot(wp_x[-1], wp_y[-1], 'r*', markersize=18, zorder=10)
        
        ax_dict = {models_to_use[0]: ax1_kin, models_to_use[1]: ax1_dyn}
    else:
        # Single model mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.set_xlim(min(ox) - 5, max(ox) + 5)
        ax1.set_ylim(min(oy) - 5, max(oy) + 5)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X [m]', fontsize=12)
        ax1.set_ylabel('Y [m]', fontsize=12)
        ax1.set_title(f'{scenario_name}{title_suffix}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.plot(ox, oy, '.k', markersize=2, alpha=0.8, label='Obstacles')
        ax1.plot(path_x, path_y, '--r', linewidth=2, alpha=0.5, label='Planned Path')
        wp_x = [wp[0] for wp in waypoints]
        wp_y = [wp[1] for wp in waypoints]
        ax1.plot(wp_x[0], wp_y[0], 'go', markersize=15, label='Start', zorder=10)
        ax1.plot(wp_x[-1], wp_y[-1], 'r*', markersize=20, label='Goal', zorder=10)
        ax1.legend(loc='upper right', fontsize=9)
        ax_dict = {models_to_use[0]: ax1}
    
    # Setup metrics plot
    first_history = histories[models_to_use[0]]
    ax2.set_xlim(0, first_history['time'][-1] if first_history['time'] else 10)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Real-time Metrics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Create line objects for each model
    colors = {'kinematic': 'blue', 'dynamic': 'red'}
    lines_dict = {}
    for m_type in models_to_use:
        cte_line, = ax2.plot([], [], linestyle='-', linewidth=2, 
                            color=colors.get(m_type, 'black'),
                            label=f'{m_type.capitalize()} CTE')
        lines_dict[f'{m_type}_cte'] = cte_line
    
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Info text (on first trajectory plot)
    first_ax = ax_dict[models_to_use[0]]
    info_text = first_ax.text(0.02, 0.98, '', transform=first_ax.transAxes, 
                             fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                             fontfamily='monospace')
    
    def init():
        """Initialize animation"""
        for m_type, ax1 in ax_dict.items():
            ax1.clear()
            ax1.set_xlim(min(ox) - 5, max(ox) + 5)
            ax1.set_ylim(min(oy) - 5, max(oy) + 5)
            ax1.set_aspect('equal')
            ax1.plot(ox, oy, '.k', markersize=2, alpha=0.8)
            ax1.plot(path_x, path_y, '--r', linewidth=2, alpha=0.5)
            wp_x = [wp[0] for wp in waypoints]
            wp_y = [wp[1] for wp in waypoints]
            ax1.plot(wp_x[0], wp_y[0], 'go', markersize=12, zorder=10)
            ax1.plot(wp_x[-1], wp_y[-1], 'r*', markersize=18, zorder=10)
        
        for line in lines_dict.values():
            line.set_data([], [])
        info_text.set_text('')
        
        return list(lines_dict.values()) + [info_text]
        car_body_line.set_data([], [])
        for wheel_line in wheel_lines:
            wheel_line.set_data([], [])
        cte_line.set_data([], [])
        speed_line.set_data([], [])
        steer_line.set_data([], [])
        target_point.set_data([], [])
        return [trajectory_line, car_body_line] + wheel_lines + [target_point, cte_line, speed_line, steer_line, info_text]
    
    # Dynamic patches and vehicle elements
    dynamic_patches = {}
    vehicle_elements = {}
    
    def animate(frame):
        """Update animation frame for all models"""
        nonlocal dynamic_patches, vehicle_elements
        
        # Check max frame
        max_frame = min(len(histories[m]['x']) for m in models_to_use)
        if frame >= max_frame:
            return list(lines_dict.values()) + [info_text]
        
        # Update each model
        updated_elements = []
        for m_type in models_to_use:
            history = histories[m_type]
            ax = ax_dict[m_type]
            
            # Get vehicle state
            x, y, yaw = history['x'][frame], history['y'][frame], history['yaw'][frame]
            v = history['v'][frame]
            steer_rad = history['steer'][frame]
            
            # Initialize vehicle elements for this model if needed
            if m_type not in vehicle_elements:
                # Trajectory line
                traj_line, = ax.plot([], [], color='blue', linewidth=2.5, alpha=0.8)
                # Vehicle body
                car_body, = ax.plot([], [], 'k-', linewidth=2)
                # Wheels
                wheels = [ax.plot([], [], 'k-', linewidth=1.5)[0] for _ in range(4)]
                # Target point
                target_pt, = ax.plot([], [], 'r*', markersize=15)
                # Velocity arrow
                arrow_patch = None
                
                vehicle_elements[m_type] = {
                    'trajectory': traj_line,
                    'body': car_body,
                    'wheels': wheels,
                    'target': target_pt,
                    'arrow': arrow_patch
                }
            
            elems = vehicle_elements[m_type]
            
            # Update trajectory
            elems['trajectory'].set_data(history['x'][:frame+1], history['y'][:frame+1])
            
            # Update vehicle body
            car_x, car_y = get_car_body_vertices(x, y, yaw, VehicleParams.WHEELBASE, VehicleParams.TRACK_WIDTH)
            elems['body'].set_data(car_x, car_y)
            
            # Update wheels
            wheel_vertices = get_wheel_vertices(x, y, yaw, steer_rad, VehicleParams.WHEELBASE, VehicleParams.TRACK_WIDTH)
            for i, (wx, wy) in enumerate(wheel_vertices):
                elems['wheels'][i].set_data(wx, wy)
            
            # Skip velocity arrows in fast mode to reduce draw cost
            if not fast_mode:
                if elems['arrow'] is not None and elems['arrow'] in ax.patches:
                    elems['arrow'].remove()
                arrow_len = min(v * 1.5, 5)
                dx = arrow_len * np.cos(yaw)
                dy = arrow_len * np.sin(yaw)
                arrow = FancyArrow(x, y, dx, dy, width=0.5, head_width=1.5,
                                  head_length=1, fc='green', ec='darkgreen', alpha=0.7)
                ax.add_patch(arrow)
                elems['arrow'] = arrow
            
            # Update target point
            if 'target_idx' in history and frame < len(history['target_idx']):
                target_idx = history['target_idx'][frame]
                elems['target'].set_data([path_x[target_idx]], [path_y[target_idx]])
            
            # Update metrics for first model
            if m_type == models_to_use[0]:
                lines_dict[f'{m_type}_cte'].set_data(history['time'][:frame+1], history['cte'][:frame+1])
                if len(models_to_use) > 1 and m_type != models_to_use[1]:
                    lines_dict[f'{models_to_use[1]}_cte'].set_data(
                        histories[models_to_use[1]]['time'][:frame+1],
                        histories[models_to_use[1]]['cte'][:frame+1]
                    )
                
                # Auto-scale metrics
                all_ctes = history['cte'][:frame+1]
                if all_ctes:
                    ax2.set_ylim(min(all_ctes) - 1, max(all_ctes) + 1)
        
        # Update info text with first model
        h = histories[models_to_use[0]]
        x, y, yaw = h['x'][frame], h['y'][frame], h['yaw'][frame]
        v = h['v'][frame]
        info_str = f"Time: {h['time'][frame]:.1f}s | Model: {models_to_use[0].title()}\n"
        info_str += f"Pos: ({x:.1f}, {y:.1f}) | Speed: {v:.2f} m/s\n"
        info_str += f"CTE: {h['cte'][frame]:.2f} m | Heading: {np.degrees(yaw):.1f}°"
        info_text.set_text(info_str)
        
        return list(lines_dict.values()) + [info_text] + \
               [vehicle_elements[m]['trajectory'] for m in models_to_use] + \
               [vehicle_elements[m]['body'] for m in models_to_use] + \
               [w for m in models_to_use for w in vehicle_elements[m]['wheels']]
    
    # Create animation
    frame_count = len(histories[models_to_use[0]]['x'])
    frame_indices = list(range(0, frame_count, frame_stride)) or [0]
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=frame_indices, interval=50,
                                  blit=False, repeat=True)
    
    # Save animation
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=export_fps, bitrate=export_bitrate, codec='libx264')
        anim.save(output_file, writer=writer)
        print(f"  ✓ Animation saved to: {output_file}")
    except Exception as e:
        print(f"  ✗ Failed to save video (ffmpeg required): {e}")
        print(f"  Attempting GIF export instead...")
        try:
            anim.save(output_file.replace('.mp4', '.gif'), writer='pillow', fps=15)
            print(f"  ✓ Animation saved as GIF: {output_file.replace('.mp4', '.gif')}")
        except Exception as e2:
            print(f"  ✗ GIF export also failed: {e2}")
            print(f"  Displaying animation instead...")
            plt.show()
    
    # Generate static analysis plots
    print(f"  Generating analysis plots...")
    generate_analysis_plots(scenario_name, models_to_use, histories, path_x, path_y, ox, oy, waypoints)
    
    return anim


def generate_analysis_plots(scenario_name, models_to_use, histories, path_x, path_y, ox, oy, waypoints):
    """Generate static analysis plots comparing models (trajectory, metrics, performance radar)."""
    n_models = len(models_to_use)
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Trajectory comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(ox, oy, '.k', markersize=1, alpha=0.6, label='Obstacles')
    ax1.plot(path_x, path_y, '--r', linewidth=2, alpha=0.5, label='Planned Path')
    colors = {'kinematic': 'blue', 'dynamic': 'red'}
    for m_type in models_to_use:
        ax1.plot(histories[m_type]['x'], histories[m_type]['y'], 
                color=colors.get(m_type, 'black'), linewidth=2.5, 
                label=f'{m_type.capitalize()} Trajectory', alpha=0.8)
    wp_x = [wp[0] for wp in waypoints]
    wp_y = [wp[1] for wp in waypoints]
    ax1.plot(wp_x[0], wp_y[0], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(wp_x[-1], wp_y[-1], 'r*', markersize=18, label='Goal', zorder=5)
    ax1.set_xlabel('X [m]', fontsize=11)
    ax1.set_ylabel('Y [m]', fontsize=11)
    ax1.set_title(f'{scenario_name} - Trajectories', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Cross-Track Error (CTE)
    ax2 = plt.subplot(2, 3, 2)
    for m_type in models_to_use:
        ax2.plot(histories[m_type]['time'], histories[m_type]['cte'], 
                color=colors.get(m_type, 'black'), linewidth=2, 
                label=f'{m_type.capitalize()}', alpha=0.8)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('CTE [m]', fontsize=11)
    ax2.set_title('Cross-Track Error (Stability)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Steering Angle
    ax3 = plt.subplot(2, 3, 3)
    for m_type in models_to_use:
        ax3.plot(histories[m_type]['time'], np.degrees(histories[m_type]['steer']), 
                color=colors.get(m_type, 'black'), linewidth=2, 
                label=f'{m_type.capitalize()}', alpha=0.8)
    ax3.set_xlabel('Time [s]', fontsize=11)
    ax3.set_ylabel('Steering Angle [deg]', fontsize=11)
    ax3.set_title('Steering Command', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Speed Profile
    ax4 = plt.subplot(2, 3, 4)
    for m_type in models_to_use:
        ax4.plot(histories[m_type]['time'], histories[m_type]['v'], 
                color=colors.get(m_type, 'black'), linewidth=2, 
                label=f'{m_type.capitalize()}', alpha=0.8)
    ax4.set_xlabel('Time [s]', fontsize=11)
    ax4.set_ylabel('Speed [m/s]', fontsize=11)
    ax4.set_title('Velocity Profile', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Yaw Angle
    ax5 = plt.subplot(2, 3, 5)
    for m_type in models_to_use:
        ax5.plot(histories[m_type]['time'], np.degrees(histories[m_type]['yaw']), 
                color=colors.get(m_type, 'black'), linewidth=2, 
                label=f'{m_type.capitalize()}', alpha=0.8)
    ax5.set_xlabel('Time [s]', fontsize=11)
    ax5.set_ylabel('Yaw Angle [deg]', fontsize=11)
    ax5.set_title('Vehicle Heading', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance Metrics Radar
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    metrics = ['CTE\nAccuracy', 'Speed\nStability', 'Steer\nSmoothing', 'Path\nTracking']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for m_type in models_to_use:
        # Compute metrics (normalized 0-1)
        max_cte = np.max(np.abs(histories[m_type]['cte'])) + 0.1
        cte_score = 1.0 - np.clip(np.mean(np.abs(histories[m_type]['cte'])) / max_cte, 0, 1)
        
        speed_std = np.std(histories[m_type]['v'])
        speed_score = 1.0 - np.clip(speed_std / 2.0, 0, 1)
        
        steer_diff = np.mean(np.abs(np.diff(histories[m_type]['steer'])))
        steer_score = 1.0 - np.clip(steer_diff / 0.1, 0, 1)
        
        path_error = np.mean(np.abs(histories[m_type]['cte']))
        path_score = 1.0 - np.clip(path_error / 3.0, 0, 1)
        
        values = [cte_score, speed_score, steer_score, path_score]
        values += values[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2.5, markersize=6,
                label=f'{m_type.capitalize()}', color=colors.get(m_type, 'black'), alpha=0.8)
        ax6.fill(angles, values, alpha=0.15, color=colors.get(m_type, 'black'))
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics, fontsize=10)
    ax6.set_ylim(0, 1)
    ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax6.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
    ax6.grid(True)
    ax6.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.3, 1.1))
    ax6.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    analysis_file = f'analysis_{scenario_name.lower().replace(" ", "_")}.png'
    plt.savefig(analysis_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Analysis plots saved to: {analysis_file}")
    plt.close()

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("  SCENARIO VISUALIZATION TOOL")
    print("="*60)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize driving scenarios with different models')
    parser.add_argument('--model', type=str, default='kinematic', 
                       choices=['kinematic', 'dynamic', 'both'],
                       help='Vehicle model to use: kinematic, dynamic, or both (default: kinematic)')
    parser.add_argument('--controller', type=str, default='stanley', choices=['stanley', 'lqr', 'pure_pursuit'],
                       help='Path-tracking controller to use (default: stanley)')
    parser.add_argument('--fast', action='store_true', help='Enable fast_mode (skip frames, lower FPS)')
    args = parser.parse_args()
    
    # Model selection
    compare = args.model == 'both'
    model_type = 'kinematic' if args.model == 'kinematic' else 'dynamic'
    
    print(f"\nModel Selection: {args.model}")
    if compare:
        print("  → Running COMPARISON mode (kinematic vs dynamic side-by-side)")
    else:
        print(f"  → Running SINGLE model: {model_type}")
    print(f"Controller: {args.controller}")
    if args.fast:
        print("Fast mode: ENABLED")
    
    # Map presets with obstacles + multi-waypoint routes
    scenarios = [
        {
            'name': 'Simple Diagonal',
            'map': 'diagonal',
            'size': 60,
            'waypoints': [(5.0, 5.0), (50.0, 50.0)],
            'speed': 5.0,
            'output': f'diagonal_animation_{args.model}.mp4'
        },
        {
            'name': 'Complex Maze',
            'map': 'maze',
            'size': 60,
            'waypoints': [(5.0, 10.0), (25.0, 30.0), (50.0, 50.0)],
            'speed': 2.5,
            'output': f'maze_animation_{args.model}.mp4'
        },
        {
            'name': 'Urban Blocks',
            'map': 'urban',
            'size': 80,
            'waypoints': [(5.0, 5.0), (35.0, 5.0), (35.0, 40.0), (70.0, 40.0), (70.0, 70.0)],
            'speed': 4.0,
            'output': f'urban_animation_{args.model}.mp4'
        },
        {
            'name': 'Warehouse Aisles',
            'map': 'warehouse',
            'size': 70,
            'waypoints': [(5.0, 5.0), (5.0, 30.0), (20.0, 30.0), (20.0, 60.0), (60.0, 60.0)],
            'speed': 3.5,
            'output': f'warehouse_animation_{args.model}.mp4'
        },
        {
            'name': 'Zigzag Park',
            'map': 'park',
            'size': 80,
            'waypoints': [(5.0, 5.0), (20.0, 20.0), (35.0, 5.0), (50.0, 20.0), (65.0, 5.0), (75.0, 25.0)],
            'speed': 4.0,
            'output': f'park_animation_{args.model}.mp4'
        }
    ]
    
    # Select scenario
    print("\nAvailable scenarios:")
    for i, s in enumerate(scenarios):
        print(f"  {i+1}. {s['name']}")
    
    choice = input("\nSelect scenario (1-2) or press Enter for default [1]: ").strip()
    if not choice:
        choice = '1'
    
    scenario_idx = int(choice) - 1
    if scenario_idx < 0 or scenario_idx >= len(scenarios):
        scenario_idx = 0
    
    scenario = scenarios[scenario_idx]
    
    # Create obstacles
    from run_validation import create_obstacle_environment
    ox, oy = create_obstacle_environment(scenario.get('map', 'maze'), scenario.get('size', 60))
    grid_size = 1.0
    robot_radius = 2.0 if scenario['map'] in ['diagonal', 'urban', 'park'] else 2.5
    
    # Generate animation
    create_scenario_animation(
        scenario['name'],
        ox, oy,
        scenario['waypoints'],
        scenario['speed'],
        grid_size,
        robot_radius,
        scenario['output'],
        model_type=model_type,
        compare_models=compare,
        fast_mode=args.fast,
        controller_type=args.controller
    )
    
    print("\n" + "="*60)
    print("  VISUALIZATION COMPLETE")
    print("="*60)
