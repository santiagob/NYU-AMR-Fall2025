"""
Validation Test Suite for Ackerman Simulator
Tests multiple scenarios and generates comparison metrics
"""
import numpy as np
import matplotlib.pyplot as plt
from models.vehicle_dynamics import DynamicBicycleModel, VehicleParams
from planners.a_star import AStarPlanner
from controllers.stanley import StanleyController, PIDController

def run_scenario(name, ox, oy, sx, sy, gx, gy, target_speed=5.0):
    """Run a single test scenario and return metrics"""
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"{'='*60}")
    
    # Plan path
    planner = AStarPlanner(ox, oy, 1.0, 2.0)
    path = planner.planning(sx, sy, gx, gy)
    
    if path is None:
        print("Path planning failed!")
        return None
    
    path_x, path_y = path[:, 0], path[:, 1]
    
    # Initialize
    vehicle = DynamicBicycleModel(x=sx, y=sy, yaw=np.arctan2(gy-sy, gx-sx))
    stanley = StanleyController(k_gain=0.8, k_soft=2.0)
    speed_pid = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)
    
    # Simulate
    hx, hy, hv, hsteer, hcte, hyaw = [], [], [], [], [], []
    dt = 0.1
    max_time = 60.0
    lookahead_distance = 3.0
    
    for t in np.arange(0, max_time, dt):
        dists = np.hypot(vehicle.state[0] - path_x, vehicle.state[1] - path_y)
        nearest_idx = np.argmin(dists)
        
        idx = nearest_idx
        for i in range(nearest_idx, len(path_x)):
            dist = np.hypot(vehicle.state[0] - path_x[i], vehicle.state[1] - path_y[i])
            if dist >= lookahead_distance:
                idx = i
                break
        idx = min(idx, len(path_x) - 1)
        
        dist_to_goal = np.hypot(vehicle.state[0] - gx, vehicle.state[1] - gy)
        if dist_to_goal < 2.0:
            print(f"✓ Goal reached at t={t:.1f}s")
            break
        
        steer_cmd, cte = stanley.compute(vehicle.state, path_x, path_y, idx)
        steer_cmd = np.clip(steer_cmd, -VehicleParams.MAX_STEER, VehicleParams.MAX_STEER)
        
        current_speed = np.hypot(vehicle.state[3], vehicle.state[4])
        throttle = speed_pid.compute(target_speed, current_speed, dt)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        vehicle.update(throttle, steer_cmd, dt)
        
        hx.append(vehicle.state[0])
        hy.append(vehicle.state[1])
        hyaw.append(vehicle.state[2])
        hv.append(current_speed)
        hsteer.append(steer_cmd)
        hcte.append(cte)
    
    # Calculate metrics
    final_error = np.hypot(vehicle.state[0] - gx, vehicle.state[1] - gy)
    mean_cte = np.mean(np.abs(hcte))
    max_cte = np.max(np.abs(hcte))
    mean_speed = np.mean(hv)
    max_steer = np.max(np.abs(hsteer))
    
    metrics = {
        'name': name,
        'success': final_error < 2.0,
        'time': len(hx) * dt,
        'final_error': final_error,
        'mean_cte': mean_cte,
        'max_cte': max_cte,
        'mean_speed': mean_speed,
        'max_steer': np.degrees(max_steer),
        'path_x': path_x,
        'path_y': path_y,
        'hx': np.array(hx),
        'hy': np.array(hy),
        'hcte': np.array(hcte),
        'hsteer': np.array(hsteer),
        'ox': ox,
        'oy': oy
    }
    
    print(f"Final Error: {final_error:.2f}m")
    print(f"Mean CTE: {mean_cte:.2f}m")
    print(f"Max CTE: {max_cte:.2f}m")
    print(f"Mean Speed: {mean_speed:.2f} m/s")
    print(f"Max Steering: {np.degrees(max_steer):.1f}°")
    
    return metrics

def create_obstacle_environment(scenario_type):
    """Create different obstacle configurations"""
    ox, oy = [], []
    
    # Boundary walls (common to all)
    for i in range(60):
        ox.append(i); oy.append(0.0)
        ox.append(i); oy.append(60.0)
    for i in range(60):
        ox.append(0.0); oy.append(i)
        ox.append(60.0); oy.append(i)
    
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
        # More complex maze
        for i in range(10, 30):
            ox.append(i); oy.append(20.0)
        for i in range(30, 50):
            ox.append(i); oy.append(40.0)
        for i in range(20, 40):
            ox.append(30.0); oy.append(i)
    
    return ox, oy

def main():
    """Run validation test suite"""
    print("\n" + "="*60)
    print("ACKERMAN SIMULATOR - VALIDATION TEST SUITE")
    print("="*60)
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'Simple Diagonal',
            'type': 'diagonal',
            'start': (5.0, 5.0),
            'goal': (50.0, 50.0),
            'speed': 5.0
        },
        {
            'name': 'Tight Corridor',
            'type': 'corridor',
            'start': (5.0, 30.0),
            'goal': (55.0, 30.0),
            'speed': 4.0
        },
        {
            'name': 'Complex Maze',
            'type': 'maze',
            'start': (5.0, 10.0),
            'goal': (50.0, 50.0),
            'speed': 5.0
        }
    ]
    
    # Run all scenarios
    results = []
    for scenario in scenarios:
        ox, oy = create_obstacle_environment(scenario['type'])
        sx, sy = scenario['start']
        gx, gy = scenario['goal']
        
        metrics = run_scenario(
            scenario['name'],
            ox, oy, sx, sy, gx, gy,
            target_speed=scenario['speed']
        )
        
        if metrics:
            results.append(metrics)
    
    # Summary table
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<20} {'Success':<10} {'Time':<8} {'Final Err':<12} {'Mean CTE':<12}")
    print("-" * 60)
    
    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{r['name']:<20} {status:<10} {r['time']:>6.1f}s {r['final_error']:>10.2f}m {r['mean_cte']:>10.2f}m")
    
    # Visualization
    n_scenarios = len(results)
    fig = plt.figure(figsize=(16, 4 * n_scenarios))
    
    for i, r in enumerate(results):
        # Trajectory plot
        ax1 = plt.subplot(n_scenarios, 3, i*3 + 1)
        ax1.plot(r['ox'], r['oy'], '.k', markersize=1)
        ax1.plot(r['path_x'], r['path_y'], '--r', linewidth=2, label='Planned')
        ax1.plot(r['hx'], r['hy'], '-b', linewidth=2, label='Actual')
        ax1.set_title(f"{r['name']} - Trajectory")
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.legend()
        ax1.axis('equal')
        ax1.grid(True)
        
        # Cross-track error
        ax2 = plt.subplot(n_scenarios, 3, i*3 + 2)
        time_steps = np.arange(len(r['hcte'])) * 0.1
        ax2.plot(time_steps, r['hcte'], 'r-', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title(f"{r['name']} - Cross-Track Error")
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('CTE [m]')
        ax2.grid(True)
        
        # Steering angle
        ax3 = plt.subplot(n_scenarios, 3, i*3 + 3)
        ax3.plot(time_steps, np.degrees(r['hsteer']), 'g-', linewidth=2)
        ax3.set_title(f"{r['name']} - Steering Angle")
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Steering [deg]')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=150)
    print(f"\n✓ Results saved to validation_results.png")
    plt.show()
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
