"""
Validation Test Suite for Ackerman Simulator
Tests multiple scenarios and generates comparison metrics
"""
import numpy as np
import matplotlib.pyplot as plt
from models.vehicle_dynamics import DynamicBicycleModel, VehicleParams
from planners.a_star import AStarPlanner
from controllers.stanley import StanleyController, PIDController

def smooth_path_bezier(path_x, path_y, num_points=None, smoothness=3):
    """
    Smooth a path using Bezier curve interpolation.
    This eliminates sharp corners in A* paths, allowing better path tracking.
    
    Args:
        path_x, path_y: Original path waypoints
        num_points: Number of points in smoothed path (default: 3x original)
        smoothness: Number of control points to use (higher = smoother but slower)
    
    Returns:
        Smoothed path coordinates (x, y)
    """
    if len(path_x) < 3:
        return path_x, path_y
    
    if num_points is None:
        num_points = len(path_x) * 3
    
    # Convert to parameter t from 0 to 1
    original_t = np.linspace(0, 1, len(path_x))
    
    # Simple cubic spline interpolation using numpy (more stable than custom Bezier)
    from scipy.interpolate import CubicSpline
    
    try:
        cs_x = CubicSpline(original_t, path_x)
        cs_y = CubicSpline(original_t, path_y)
        
        # Generate smooth path
        smooth_t = np.linspace(0, 1, num_points)
        smooth_x = cs_x(smooth_t)
        smooth_y = cs_y(smooth_t)
        
        return smooth_x, smooth_y
    except:
        # If spline fails, return original path
        return path_x, path_y

def calculate_path_curvature(path_x, path_y):
    """Calculate curvature at each point along path"""
    if len(path_x) < 3:
        return np.zeros(len(path_x))
    
    # First derivative (tangent)
    dx = np.gradient(path_x)
    dy = np.gradient(path_y)
    
    # Second derivative (curvature)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    # Curvature = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
    curvature = np.abs(dx * d2y - dy * d2x) / np.power(dx**2 + dy**2 + 1e-6, 1.5)
    
    return curvature

def run_scenario(name, ox, oy, waypoints, target_speed=5.0, grid_size=1.0, robot_radius=2.0, use_smoothing=True):
    """
    Run a multi-waypoint test scenario and return comprehensive metrics
    
    Args:
        name: Scenario name
        ox, oy: Obstacle coordinates
        waypoints: List of (x, y) tuples representing waypoints to visit in order
        target_speed: Target speed in m/s
        grid_size: A* grid resolution
        robot_radius: Safety buffer for obstacle avoidance
        use_smoothing: Apply Bezier curve smoothing to paths (default: True)
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"Waypoints: {len(waypoints)} points")
    if use_smoothing:
        print(f"Path Smoothing: ENABLED")
    print(f"{'='*60}")
    
    # Initialize planner once
    planner = AStarPlanner(ox, oy, grid_size, robot_radius)
    
    # Plan paths between consecutive waypoints
    all_path_x, all_path_y = [], []
    planning_times = []
    
    for i in range(len(waypoints) - 1):
        sx, sy = waypoints[i]
        gx, gy = waypoints[i + 1]
        
        print(f"  Planning segment {i+1}/{len(waypoints)-1}: ({sx:.1f},{sy:.1f}) → ({gx:.1f},{gy:.1f})")
        
        import time
        t_start = time.time()
        path = planner.planning(sx, sy, gx, gy)
        t_plan = time.time() - t_start
        planning_times.append(t_plan)
        
        if path is None:
            print(f"  ✗ Path planning failed for segment {i+1}!")
            return None
        
        # Apply path smoothing if enabled
        if use_smoothing and len(path) > 3:
            path_x_smooth, path_y_smooth = smooth_path_bezier(
                path[:, 0], path[:, 1], 
                num_points=len(path) * 2,
                smoothness=3
            )
            path = np.column_stack((path_x_smooth, path_y_smooth))
        
        print(f"  ✓ Path found: {len(path)} points (smoothed) in {t_plan:.3f}s")
        
        # Concatenate paths (skip first point if not first segment to avoid duplicates)
        if i == 0:
            all_path_x.extend(path[:, 0])
            all_path_y.extend(path[:, 1])
        else:
            all_path_x.extend(path[1:, 0])
            all_path_y.extend(path[1:, 1])
    
    path_x = np.array(all_path_x)
    path_y = np.array(all_path_y)
    
    print(f"  Total path length: {len(path_x)} points")
    print(f"  Total planning time: {sum(planning_times):.3f}s")
    
    # Calculate average path curvature to tune controller gains
    curvature = calculate_path_curvature(path_x, path_y)
    mean_curvature = np.mean(curvature)
    max_curvature = np.max(curvature)
    
    # Adaptive controller gains based on path complexity
    # Higher curvature → higher k_gain for better tracking
    if max_curvature > 0.15:  # High curvature path
        stanley_k = 1.2
        stanley_k_soft = 1.5
        print(f"  High-curvature path detected (max κ={max_curvature:.3f}) → Aggressive gains")
    elif max_curvature > 0.05:  # Medium curvature
        stanley_k = 0.9
        stanley_k_soft = 2.0
        print(f"  Medium-curvature path (max κ={max_curvature:.3f}) → Balanced gains")
    else:  # Low curvature
        stanley_k = 0.8
        stanley_k_soft = 2.0
        print(f"  Low-curvature path (max κ={max_curvature:.3f}) → Smooth gains")
    
    # Initialize vehicle at first waypoint
    sx, sy = waypoints[0]
    gx, gy = waypoints[-1]
    initial_yaw = np.arctan2(path_y[1] - path_y[0], path_x[1] - path_x[0])
    
    vehicle = DynamicBicycleModel(x=sx, y=sy, yaw=initial_yaw)
    stanley = StanleyController(k_gain=stanley_k, k_soft=stanley_k_soft)
    speed_pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)
    
    # Simulation history
    hx, hy, hv, hsteer, hcte, hyaw, hthrottle = [], [], [], [], [], [], []
    waypoint_times = []  # Time when each waypoint is reached
    current_waypoint_idx = 1  # Next waypoint to reach (0 is start)
    
    dt = 0.1
    max_time = 200.0  # Longer time for multi-waypoint missions
    lookahead_base = 2.0  # Base lookahead distance
    lookahead_gain = 0.5  # Speed-dependent gain (Ld = base + gain * v)
    waypoint_tolerance = 2.5
    segment_completion_counter = 0  # Counter to detect if vehicle overshoots segment
    
    print(f"\n  Executing trajectory...")
    import time
    t_sim_start = time.time()
    
    for t in np.arange(0, max_time, dt):
        # Calculate speed-adaptive lookahead distance
        current_speed = np.hypot(vehicle.state[3], vehicle.state[4])
        lookahead_distance = lookahead_base + lookahead_gain * current_speed
        lookahead_distance = np.clip(lookahead_distance, 1.0, 5.0)  # Clamp to reasonable range
        
        # Find nearest and lookahead points on path
        dists = np.hypot(vehicle.state[0] - path_x, vehicle.state[1] - path_y)
        nearest_idx = np.argmin(dists)
        
        # Check if vehicle is at the end of path (past segment completion)
        # If nearest point hasn't changed in 30 iterations and we're near path end,
        # assume segment is complete
        if nearest_idx >= len(path_x) - 5:
            segment_completion_counter += 1
        else:
            segment_completion_counter = 0
        
        # If stuck at path end, check waypoint achievement
        if segment_completion_counter > 30 or nearest_idx == len(path_x) - 1:
            if current_waypoint_idx < len(waypoints):
                wx, wy = waypoints[current_waypoint_idx]
                dist_to_waypoint = np.hypot(vehicle.state[0] - wx, vehicle.state[1] - wy)
                if dist_to_waypoint < waypoint_tolerance * 1.5:  # Slightly relaxed tolerance
                    waypoint_times.append(t)
                    print(f"  ✓ Waypoint {current_waypoint_idx}/{len(waypoints)-1} reached at t={t:.1f}s")
                    current_waypoint_idx += 1
                    segment_completion_counter = 0
        
        idx = nearest_idx
        for i in range(nearest_idx, len(path_x)):
            dist = np.hypot(vehicle.state[0] - path_x[i], vehicle.state[1] - path_y[i])
            if dist >= lookahead_distance:
                idx = i
                break
        idx = min(idx, len(path_x) - 1)
        
        # Check if waypoint reached (more lenient detection)
        if current_waypoint_idx < len(waypoints):
            wx, wy = waypoints[current_waypoint_idx]
            dist_to_waypoint = np.hypot(vehicle.state[0] - wx, vehicle.state[1] - wy)
            if dist_to_waypoint < waypoint_tolerance:
                waypoint_times.append(t)
                print(f"  ✓ Waypoint {current_waypoint_idx}/{len(waypoints)-1} reached at t={t:.1f}s")
                current_waypoint_idx += 1
                segment_completion_counter = 0
        
        # Check if final goal reached
        dist_to_goal = np.hypot(vehicle.state[0] - gx, vehicle.state[1] - gy)
        if dist_to_goal < waypoint_tolerance and current_waypoint_idx >= len(waypoints):
            print(f"  ✓ Final goal reached at t={t:.1f}s")
            break
        
        # Compute control
        steer_cmd, cte = stanley.compute(vehicle.state, path_x, path_y, idx)
        steer_cmd = np.clip(steer_cmd, -VehicleParams.MAX_STEER, VehicleParams.MAX_STEER)
        
        # Adaptive speed reduction when steering angle is large (stability)
        # If asking for high steering, reduce speed to maintain control authority
        steer_magnitude = abs(steer_cmd)
        max_steer_rad = VehicleParams.MAX_STEER
        
        # Speed reduction factor: 1.0 at low steer, 0.5 at max steer
        if steer_magnitude > np.radians(30):
            speed_reduction = 1.0 - 0.5 * (steer_magnitude - np.radians(30)) / (max_steer_rad - np.radians(30))
            speed_reduction = np.clip(speed_reduction, 0.5, 1.0)
            adaptive_target_speed = target_speed * speed_reduction
        else:
            adaptive_target_speed = target_speed
        
        current_speed = np.hypot(vehicle.state[3], vehicle.state[4])
        throttle = speed_pid.compute(adaptive_target_speed, current_speed, dt)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # Update dynamics
        vehicle.update(throttle, steer_cmd, dt)
        
        # Record history
        hx.append(vehicle.state[0])
        hy.append(vehicle.state[1])
        hyaw.append(vehicle.state[2])
        hv.append(current_speed)
        hsteer.append(steer_cmd)
        hcte.append(cte)
        hthrottle.append(throttle)
    
    t_sim_total = time.time() - t_sim_start
    
    # Calculate comprehensive metrics
    final_error = np.hypot(vehicle.state[0] - gx, vehicle.state[1] - gy)
    mean_cte = np.mean(np.abs(hcte))
    max_cte = np.max(np.abs(hcte))
    std_cte = np.std(hcte)
    mean_speed = np.mean(hv)
    max_steer = np.max(np.abs(hsteer))
    mean_steer_rate = np.mean(np.abs(np.diff(hsteer))) / dt if len(hsteer) > 1 else 0
    
    # Path length metrics
    actual_distance = np.sum(np.sqrt(np.diff(hx)**2 + np.diff(hy)**2))
    planned_distance = np.sum(np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2))
    path_efficiency = planned_distance / actual_distance if actual_distance > 0 else 0
    
    # Control effort metrics
    steering_effort = np.sum(np.abs(hsteer)) * dt
    throttle_effort = np.sum(np.abs(hthrottle)) * dt
    
    # Success criteria
    waypoints_reached = len(waypoint_times)
    all_waypoints_reached = waypoints_reached == len(waypoints) - 1
    goal_reached = final_error < waypoint_tolerance
    
    metrics = {
        'name': name,
        'success': goal_reached and all_waypoints_reached,
        'time': len(hx) * dt,
        'sim_time': t_sim_total,
        'planning_time': sum(planning_times),
        'final_error': final_error,
        'waypoints_reached': waypoints_reached,
        'total_waypoints': len(waypoints) - 1,
        'waypoint_times': waypoint_times,
        
        # Tracking metrics
        'mean_cte': mean_cte,
        'max_cte': max_cte,
        'std_cte': std_cte,
        
        # Path metrics
        'actual_distance': actual_distance,
        'planned_distance': planned_distance,
        'path_efficiency': path_efficiency,
        
        # Speed metrics
        'mean_speed': mean_speed,
        'target_speed': target_speed,
        'speed_error': abs(mean_speed - target_speed),
        
        # Control metrics
        'max_steer': np.degrees(max_steer),
        'mean_steer_rate': np.degrees(mean_steer_rate),
        'steering_effort': steering_effort,
        'throttle_effort': throttle_effort,
        
        # History data
        'path_x': path_x,
        'path_y': path_y,
        'hx': np.array(hx),
        'hy': np.array(hy),
        'hcte': np.array(hcte),
        'hsteer': np.array(hsteer),
        'hv': np.array(hv),
        'ox': ox,
        'oy': oy,
        'waypoints': waypoints
    }
    
    # Print summary
    print(f"\n  Performance Summary:")
    print(f"    Success: {'✓ PASS' if metrics['success'] else '✗ FAIL'}")
    print(f"    Waypoints: {waypoints_reached}/{len(waypoints)-1} reached")
    print(f"    Final Error: {final_error:.2f}m")
    print(f"    Time: {metrics['time']:.1f}s (sim: {t_sim_total:.2f}s, plan: {sum(planning_times):.2f}s)")
    print(f"    Distance: {actual_distance:.1f}m (planned: {planned_distance:.1f}m, efficiency: {path_efficiency:.2%})")
    print(f"    Mean CTE: {mean_cte:.2f}m ± {std_cte:.2f}m (max: {max_cte:.2f}m)")
    print(f"    Mean Speed: {mean_speed:.2f} m/s (target: {target_speed:.2f} m/s)")
    print(f"    Max Steering: {np.degrees(max_steer):.1f}° (rate: {np.degrees(mean_steer_rate):.1f}°/s)")
    
    return metrics

def create_obstacle_environment(scenario_type, map_size=60):
    """Create different obstacle configurations with variable map sizes"""
    ox, oy = [], []
    
    # Boundary walls (common to all)
    for i in range(map_size):
        ox.append(i); oy.append(0.0)
        ox.append(i); oy.append(map_size)
    for i in range(map_size):
        ox.append(0.0); oy.append(i)
        ox.append(map_size); oy.append(i)
    
    if scenario_type == "diagonal":
        # Original diagonal obstacles
        for i in range(10, 20):
            ox.append(i); oy.append(30.0)
        for i in range(30, 40):
            ox.append(40.0); oy.append(i)
    
    elif scenario_type == "corridor":
        # Narrow corridor
        for i in range(15, 45):
            ox.append(i); oy.append(25.0)
            ox.append(i); oy.append(35.0)
        
    elif scenario_type == "maze":
        # Complex maze with wide navigable passages
        for i in range(10, 25):  # Wide gap at 25-33
            ox.append(i); oy.append(20.0)
        for i in range(35, 50):  # Wide gap at 30-35
            ox.append(i); oy.append(40.0)
        for i in range(25, 35):  # Short vertical wall with gaps
            ox.append(30.0); oy.append(i)
    
    elif scenario_type == "large_warehouse":
        # Large warehouse layout with wide aisles
        spacing = map_size // 5  # Fewer, wider aisles
        for row in range(1, 4):  # Reduced number of rows
            y_pos = row * spacing
            for col in range(5, map_size - spacing - 10, spacing + 5):  # Wider spacing
                # Create aisle obstacles (shortened for gaps)
                for i in range(spacing - 8):  # Shorter obstacles
                    if col + i < map_size - 10:
                        ox.append(col + i)
                        oy.append(y_pos)
    
    elif scenario_type == "parking_lot":
        # Parking lot with obstacles
        for row in range(2, map_size - 10, 8):
            for col in range(5, map_size - 10, 12):
                # Create parking obstacles (cars)
                for i in range(6):
                    for j in range(3):
                        if col + i < map_size - 5:
                            ox.append(col + i)
                            oy.append(row + j)
    
    elif scenario_type == "urban_grid":
        # City-like grid with buildings
        block_size = 15
        street_width = 5
        for bx in range(0, map_size, block_size + street_width):
            for by in range(0, map_size, block_size + street_width):
                # Create building blocks
                for i in range(block_size):
                    for j in range(block_size):
                        if 5 < bx + i < map_size - 5 and 5 < by + j < map_size - 5:
                            ox.append(bx + i)
                            oy.append(by + j)
    
    elif scenario_type == "slalom":
        # Slalom course with alternating obstacles
        num_gates = map_size // 10
        for gate in range(num_gates):
            y_pos = 10 + gate * 10
            if gate % 2 == 0:
                x_start = 10
            else:
                x_start = map_size - 20
            
            for i in range(15):
                ox.append(x_start + i)
                oy.append(y_pos)
    
    elif scenario_type == "forest":
        # Random forest-like obstacles
        np.random.seed(42)
        num_trees = int(map_size * 1.5)
        for _ in range(num_trees):
            tree_x = np.random.randint(10, map_size - 10)
            tree_y = np.random.randint(10, map_size - 10)
            # Create tree cluster (radius 2)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx*dx + dy*dy <= 4:
                        ox.append(tree_x + dx)
                        oy.append(tree_y + dy)
    
    elif scenario_type == "extreme_maze":
        # EXTREME: Intentionally tight maze to showcase system limits
        # This scenario is EXPECTED TO FAIL - demonstrates vehicle constraints
        for i in range(10, 30):  # No gaps - tight passage
            ox.append(i); oy.append(20.0)
        for i in range(30, 50):  # No gaps
            ox.append(i); oy.append(40.0)
        for i in range(20, 40):  # Full vertical wall
            ox.append(30.0); oy.append(i)
        # Add additional tight chicane
        for i in range(15, 25):
            ox.append(15.0); oy.append(i)
        for i in range(35, 45):
            ox.append(45.0); oy.append(i)
    
    return ox, oy

def main():
    """Run comprehensive validation test suite with multiple scenarios"""
    print("\n" + "="*80)
    print(" "*20 + "ACKERMAN SIMULATOR - VALIDATION TEST SUITE")
    print("="*80)
    
    # Define test scenarios with multiple waypoints
    scenarios = [
        {
            'name': 'Simple Diagonal',
            'type': 'diagonal',
            'map_size': 60,
            'waypoints': [(5.0, 5.0), (50.0, 50.0)],
            'speed': 5.0,
            'grid_size': 1.0,
            'robot_radius': 2.0,
            'smoothing': False
        },
        {
            'name': 'Tight Corridor',
            'type': 'corridor',
            'map_size': 60,
            'waypoints': [(5.0, 30.0), (30.0, 30.0), (55.0, 30.0)],
            'speed': 4.0,
            'grid_size': 1.0,
            'robot_radius': 2.0,
            'smoothing': True
        },
        {
            'name': 'Complex Maze',
            'type': 'maze',
            'map_size': 60,
            'waypoints': [(5.0, 10.0), (25.0, 30.0), (50.0, 50.0)],
            'speed': 2.5,  # Conservative speed for reliable tracking
            'grid_size': 1.0,
            'robot_radius': 2.5,
            'smoothing': False
        },
        {
            'name': 'Large Warehouse Tour',
            'type': 'large_warehouse',
            'map_size': 100,
            'waypoints': [(10.0, 10.0), (90.0, 20.0), (90.0, 80.0), (10.0, 90.0)],
            'speed': 3.5,  # Conservative speed for long mission
            'grid_size': 2.0,
            'robot_radius': 3.0,
            'smoothing': False
        },
        {
            'name': 'Parking Lot Navigation',
            'type': 'parking_lot',
            'map_size': 80,
            'waypoints': [(10.0, 10.0), (40.0, 70.0), (70.0, 40.0), (70.0, 70.0)],
            'speed': 4.0,
            'grid_size': 1.0,
            'robot_radius': 2.0,
            'smoothing': True
        },
        {
            'name': 'Urban Grid Delivery',
            'type': 'urban_grid',
            'map_size': 120,
            'waypoints': [(10.0, 10.0), (50.0, 30.0), (90.0, 70.0), (110.0, 110.0)],
            'speed': 5.0,
            'grid_size': 2.0,
            'robot_radius': 3.0,
            'smoothing': True
        },
        {
            'name': 'Slalom Challenge',
            'type': 'slalom',
            'map_size': 80,
            'waypoints': [(30.0, 5.0), (30.0, 40.0), (30.0, 75.0)],
            'speed': 5.0,
            'grid_size': 1.0,
            'robot_radius': 2.0,
            'smoothing': True
        },
        {
            'name': 'Forest Path',
            'type': 'forest',
            'map_size': 100,
            'waypoints': [(15.0, 15.0), (50.0, 50.0), (85.0, 85.0)],
            'speed': 4.5,
            'grid_size': 1.0,
            'robot_radius': 2.5,
            'smoothing': True
        },
        {
            'name': '⚠️ EXTREME Maze (Limit Test)',
            'type': 'extreme_maze',
            'map_size': 60,
            'waypoints': [(5.0, 10.0), (25.0, 30.0), (50.0, 50.0)],
            'speed': 2.0,  # Slow speed but still expected to fail
            'grid_size': 1.0,
            'robot_radius': 2.0,
            'smoothing': False
        }
    ]
    
    # Run all scenarios
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"Running Test {i}/{len(scenarios)}: {scenario['name']}")
        print(f"Map Size: {scenario['map_size']}x{scenario['map_size']}, "
              f"Waypoints: {len(scenario['waypoints'])}, "
              f"Speed: {scenario['speed']} m/s")
        print(f"{'='*80}")
        
        ox, oy = create_obstacle_environment(scenario['type'], scenario['map_size'])
        
        metrics = run_scenario(
            scenario['name'],
            ox, oy,
            scenario['waypoints'],
            target_speed=scenario['speed'],
            grid_size=scenario['grid_size'],
            robot_radius=scenario['robot_radius'],
            use_smoothing=scenario.get('smoothing', True)
        )
        
        if metrics:
            results.append(metrics)
    
    # Comprehensive summary table
    print(f"\n{'='*120}")
    print(" "*45 + "VALIDATION SUMMARY")
    print(f"{'='*120}")
    print(f"{'Scenario':<25} {'Status':<10} {'WP':<8} {'Time':<8} {'Dist':<10} {'Eff':<8} {'CTE':<12} {'Speed':<10} {'Steer':<10}")
    print("-" * 120)
    
    pass_count = 0
    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        if r['success']:
            pass_count += 1
        
        wp_status = f"{r['waypoints_reached']}/{r['total_waypoints']}"
        efficiency = f"{r['path_efficiency']:.1%}"
        cte_info = f"{r['mean_cte']:.2f}±{r['std_cte']:.2f}"
        speed_info = f"{r['mean_speed']:.1f}/{r['target_speed']:.1f}"
        steer_info = f"{r['max_steer']:.1f}°"
        
        print(f"{r['name']:<25} {status:<10} {wp_status:<8} {r['time']:>6.1f}s "
              f"{r['actual_distance']:>8.1f}m {efficiency:<8} {cte_info:<12} "
              f"{speed_info:<10} {steer_info:<10}")
    
    print("-" * 120)
    print(f"Overall Success Rate: {pass_count}/{len(results)} ({pass_count/len(results)*100:.1f}%)")
    
    # Performance statistics
    if results:
        print(f"\n{'='*120}")
        print(" "*45 + "AGGREGATE STATISTICS")
        print(f"{'='*120}")
        
        all_cte = [r['mean_cte'] for r in results]
        all_efficiency = [r['path_efficiency'] for r in results]
        all_times = [r['time'] for r in results]
        all_distances = [r['actual_distance'] for r in results]
        
        print(f"  Cross-Track Error:  Mean={np.mean(all_cte):.2f}m, "
              f"Median={np.median(all_cte):.2f}m, "
              f"Max={np.max(all_cte):.2f}m")
        print(f"  Path Efficiency:    Mean={np.mean(all_efficiency):.1%}, "
              f"Median={np.median(all_efficiency):.1%}, "
              f"Min={np.min(all_efficiency):.1%}")
        print(f"  Mission Time:       Mean={np.mean(all_times):.1f}s, "
              f"Median={np.median(all_times):.1f}s, "
              f"Max={np.max(all_times):.1f}s")
        print(f"  Distance Traveled:  Mean={np.mean(all_distances):.1f}m, "
              f"Total={np.sum(all_distances):.1f}m")
    
    # Visualization
    n_scenarios = len(results)
    fig = plt.figure(figsize=(18, 5 * n_scenarios))
    
    for i, r in enumerate(results):
        # Trajectory plot with waypoints
        ax1 = plt.subplot(n_scenarios, 4, i*4 + 1)
        ax1.plot(r['ox'], r['oy'], '.k', markersize=1.0, alpha=1.0, label='Obstacles')
        ax1.plot(r['path_x'], r['path_y'], '--r', linewidth=1.5, alpha=0.7, label='Planned Path')
        ax1.plot(r['hx'], r['hy'], '-b', linewidth=2, label='Actual Trajectory')
        
        # Plot waypoints
        waypoints = r['waypoints']
        wp_x = [wp[0] for wp in waypoints]
        wp_y = [wp[1] for wp in waypoints]
        ax1.plot(wp_x[0], wp_y[0], 'go', markersize=12, label='Start', zorder=5)
        ax1.plot(wp_x[1:-1], wp_y[1:-1], 'yo', markersize=10, label='Waypoints', zorder=5)
        ax1.plot(wp_x[-1], wp_y[-1], 'r*', markersize=15, label='Goal', zorder=5)
        
        # Add waypoint numbers
        for j, (wx, wy) in enumerate(waypoints):
            ax1.text(wx, wy, str(j), fontsize=8, ha='center', va='center', 
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        status_color = 'green' if r['success'] else 'red'
        ax1.set_title(f"{r['name']}\n[{r['waypoints_reached']}/{r['total_waypoints']} WP, "
                     f"Err: {r['final_error']:.1f}m]", color=status_color, fontweight='bold')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.legend(loc='best', fontsize=8)
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # Cross-track error with statistics
        ax2 = plt.subplot(n_scenarios, 4, i*4 + 2)
        time_steps = np.arange(len(r['hcte'])) * 0.1
        ax2.plot(time_steps, r['hcte'], 'r-', linewidth=1.5, label='CTE')
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(r['mean_cte'], color='orange', linestyle=':', linewidth=2, label=f"Mean: {r['mean_cte']:.2f}m")
        ax2.axhline(-r['mean_cte'], color='orange', linestyle=':', linewidth=2)
        ax2.fill_between(time_steps, -r['std_cte'], r['std_cte'], alpha=0.2, color='orange', label=f"±1σ: {r['std_cte']:.2f}m")
        
        # Mark waypoint achievement times
        for wpt in r['waypoint_times']:
            ax2.axvline(wpt, color='green', linestyle='--', alpha=0.3)
        
        ax2.set_title(f"Cross-Track Error\n[Max: {r['max_cte']:.2f}m]")
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('CTE [m]')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Control inputs (steering + speed)
        ax3 = plt.subplot(n_scenarios, 4, i*4 + 3)
        ax3_twin = ax3.twinx()
        
        ln1 = ax3.plot(time_steps, np.degrees(r['hsteer']), 'g-', linewidth=1.5, label='Steering')
        ax3.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Steering [deg]', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        
        ln2 = ax3_twin.plot(time_steps, r['hv'], 'b-', linewidth=1.5, alpha=0.7, label='Speed')
        ax3_twin.axhline(r['target_speed'], color='b', linestyle=':', linewidth=2, alpha=0.5)
        ax3_twin.set_ylabel('Speed [m/s]', color='b')
        ax3_twin.tick_params(axis='y', labelcolor='b')
        
        ax3.set_title(f"Control Inputs\n[Max Steer: {r['max_steer']:.1f}°, "
                     f"Avg Speed: {r['mean_speed']:.1f}m/s]")
        
        # Combine legends
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics polar plot
        ax4 = plt.subplot(n_scenarios, 4, i*4 + 4, projection='polar')
        categories = ['CTE\nControl', 'Path\nEfficiency', 'Speed\nTracking', 'Steering\nSmooth']
        
        # Normalize metrics to 0-1 scale (higher is better)
        cte_score = max(0, 1 - r['mean_cte'] / 5.0)  # 0 CTE = 1.0, 5m CTE = 0
        efficiency_score = r['path_efficiency']
        speed_score = 1 - r['speed_error'] / r['target_speed']
        steer_score = max(0, 1 - r['mean_steer_rate'] / 50.0)  # Smooth steering
        
        values = [cte_score, efficiency_score, speed_score, steer_score]
        values += values[:1]  # Close the polygon
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax4.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax4.fill(angles, values, alpha=0.25, color='blue')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=8)
        ax4.set_ylim(0, 1)
        ax4.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax4.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
        ax4.set_title(f"Performance Metrics\n[Overall: {np.mean(values[:-1])*100:.1f}%]", pad=20)
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Detailed results saved to validation_results.png")
    
    # Generate performance comparison chart
    if len(results) > 1:
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        names = [r['name'] for r in results]
        
        # CTE comparison
        axes[0, 0].bar(names, [r['mean_cte'] for r in results], color='coral')
        axes[0, 0].set_ylabel('Mean CTE [m]')
        axes[0, 0].set_title('Cross-Track Error Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Efficiency comparison
        axes[0, 1].bar(names, [r['path_efficiency']*100 for r in results], color='lightblue')
        axes[0, 1].set_ylabel('Path Efficiency [%]')
        axes[0, 1].set_title('Path Following Efficiency')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time comparison
        axes[1, 0].bar(names, [r['time'] for r in results], color='lightgreen')
        axes[1, 0].set_ylabel('Mission Time [s]')
        axes[1, 0].set_title('Completion Time')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate by waypoints
        wp_data = [(r['waypoints_reached'], r['total_waypoints']) for r in results]
        success_rates = [wp[0]/wp[1]*100 if wp[1] > 0 else 0 for wp in wp_data]
        bars = axes[1, 1].bar(names, success_rates, color=['green' if r['success'] else 'red' for r in results])
        axes[1, 1].set_ylabel('Waypoint Success Rate [%]')
        axes[1, 1].set_title('Waypoint Achievement')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(100, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('validation_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ Comparison charts saved to validation_comparison.png")
    
    plt.show()
    
    print(f"\n{'='*120}")
    print(" "*45 + "VALIDATION COMPLETE")
    print(f"{'='*120}\n")

if __name__ == "__main__":
    main()
