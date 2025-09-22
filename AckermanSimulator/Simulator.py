import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# --- Vehicle Parameters ---
WHEELBASE = 2.5  # meters (L)
TRACK_WIDTH = 1.5  # meters (W)
MAX_STEER_ANGLE_DEG = 60  # degrees


def compute_kinematics(state, speed, steer_rad, wheelbase):
    """Compute instantaneous kinematic derivatives for the bicycle model.

    Args:
        state (tuple): (x, y, yaw) current state.
        speed (float): longitudinal speed (m/s).
        steer_rad (float): front wheel steer angle (radians).
        wheelbase (float): vehicle wheelbase (m).

    Returns:
        tuple: (dx, dy, dyaw) kinematic derivatives.
    """
    x, y, yaw = state

    # If steer is effectively zero, dyaw is zero and motion is straight
    if abs(steer_rad) < 1e-8:
        dx = speed * math.cos(yaw)
        dy = speed * math.sin(yaw)
        dyaw = 0.0
    else:
        # Bicycle model: yaw rate = v / L * tan(delta)
        dyaw = (speed / wheelbase) * math.tan(steer_rad)
        # Use the instantaneous yaw to compute x/y velocities
        dx = speed * math.cos(yaw)
        dy = speed * math.sin(yaw)

    return dx, dy, dyaw


def propagate_state(state, speed, steer_deg, dt, wheelbase):
    """Propagate the vehicle state for one time step using simple Euler integration.

    This function is provided as the callable API requested: input control
    variables (speed, steer_deg) and output the new kinematic state.

    Args:
        state (tuple): (x, y, yaw) current state.
        speed (float): longitudinal speed (m/s).
        steer_deg (float): steering angle in degrees (will be clipped by
                           `MAX_STEER_ANGLE_DEG` before use).
        dt (float): time step (s).
        wheelbase (float): vehicle wheelbase (m).

    Returns:
        tuple: (x_new, y_new, yaw_new, used_steer_rad)
    """
    # Clip and convert steer angle
    steer_deg_clipped = max(-MAX_STEER_ANGLE_DEG, min(MAX_STEER_ANGLE_DEG, steer_deg))
    steer_rad = math.radians(steer_deg_clipped)

    # Compute kinematic derivatives
    dx, dy, dyaw = compute_kinematics(state, speed, steer_rad, wheelbase)

    # Simple Euler integration
    x, y, yaw = state
    x_new = x + dx * dt
    y_new = y + dy * dt
    yaw_new = yaw + dyaw * dt

    # Normalize yaw
    yaw_new = math.atan2(math.sin(yaw_new), math.cos(yaw_new))

    return x_new, y_new, yaw_new, steer_rad

# --- Simulation Parameters ---
SIM_DURATION = 15  # seconds
DT = 0.1  # seconds (time step)
INITIAL_STATE = [0, 0, 0]  # [x, y, yaw (radians)]
INITIAL_SPEED = 0  # m/s

# --- Vehicle Class ---
class Vehicle:
    """
    Represents an Ackermann steering vehicle with its state and kinematics.
    The state (x, y, yaw) refers to the center of the rear axle.
    """
    def __init__(self, x, y, yaw, speed, wheelbase, track_width, max_steer_angle_rad, wheel_length=0.6, wheel_width=0.5):
        self.x = x  # x-position (center of rear axle)
        self.y = y  # y-position (center of rear axle)
        self.yaw = yaw  # yaw angle (radians, heading of the vehicle)
        self.speed = speed  # speed (m/s)

        self.wheelbase = wheelbase
        self.track_width = track_width
        self.max_steer_angle_rad = max_steer_angle_rad

        # For visualization: define vehicle body and wheel dimensions
        self.body_length = wheelbase + 2* wheel_length  # Total length including overhangs
        self.body_width = track_width + wheel_width  # Total width including fenders
        self.rear_overhang = wheel_length
        self.front_overhang = self.body_length - self.wheelbase - self.rear_overhang

        self.wheel_length = wheel_length

        self.wheel_width = wheel_width

    def update(self, dt, target_speed, target_steer_angle_deg):
        """Update the vehicle state using the (refactored) kinematic helpers.

        This method now delegates the actual kinematic computation to
        `compute_kinematics` and `propagate_state` (both defined at module
        level). The split makes it easy to call the kinematics directly from
        unit tests or other tools.

        Args:
            dt (float): Time step.
            target_speed (float): Desired speed (m/s).
            target_steer_angle_deg (float): Desired steering angle (deg).

        Returns:
            float: The actual steering angle in radians used for the update
                   (after clipping) — kept for backward compatibility with the
                   rest of the code (visualization uses this).
        """
        # Update speed (direct set; no acceleration model here)
        self.speed = target_speed

        # Use the new helper to propagate state
        x_new, y_new, yaw_new, used_steer_rad = propagate_state(
            state=(self.x, self.y, self.yaw),
            speed=self.speed,
            steer_deg=target_steer_angle_deg,
            dt=dt,
            wheelbase=self.wheelbase
        )

        self.x = x_new
        self.y = y_new
        self.yaw = yaw_new

        return used_steer_rad

    def get_car_body_vertices(self):
        """
        Calculates the vertices of the car body rectangle for visualization.
        The car body is defined relative to the rear axle center (self.x, self.y)
        and oriented by self.yaw.
        Returns:
            tuple: (x_coords, y_coords) of the car body vertices.
        """
        half_width = self.body_width / 2

        # Vertices in vehicle's local coordinate system (rear axle center is (0,0))
        # (x_local, y_local)
        vertices_local = np.array([
            [-self.rear_overhang, half_width],         # Rear-left
            [self.wheelbase + self.front_overhang, half_width],  # Front-left
            [self.wheelbase + self.front_overhang, -half_width], # Front-right
            [-self.rear_overhang, -half_width],        # Rear-right
            [-self.rear_overhang, half_width]          # Close the rectangle
        ])

        # Create rotation matrix for vehicle's yaw
        rotation_matrix = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw), np.cos(self.yaw)]
        ])

        # Rotate and translate vertices to global coordinates
        rotated_vertices = (rotation_matrix @ vertices_local.T).T
        global_vertices_x = self.x + rotated_vertices[:, 0]
        global_vertices_y = self.y + rotated_vertices[:, 1]

        return global_vertices_x, global_vertices_y

    def get_wheel_vertices(self, steer_angle_rad):
        """
        Calculates the vertices of all four wheels for visualization,
        applying Ackermann steering geometry to the front wheels.
        Args:
            steer_angle_rad (float): The steering angle of the front axle (average)
                                     as used in the bicycle model update.
        Returns:
            list: A list of (x_coords, y_coords) tuples for each wheel (FL, FR, RL, RR).
        """
        wheel_half_len = self.wheel_length / 2
        wheel_half_width = self.wheel_width / 2
        half_track = self.track_width / 2

        # Wheel positions relative to vehicle's rear axle center (0,0) in vehicle frame
        wheel_centers_local = {
            'RL': np.array([0, half_track]),
            'RR': np.array([0, -half_track]),
            'FL': np.array([self.wheelbase, half_track]),
            'FR': np.array([self.wheelbase, -half_track]),
        }

        # Calculate individual front wheel steering angles for Ackermann visualization
        delta_fl = 0
        delta_fr = 0
        if abs(steer_angle_rad) > 1e-6:
            # Calculate turning radius of the rear axle center based on the bicycle model's steer angle
            turn_radius_rear_axle = self.wheelbase / np.tan(steer_angle_rad)

            # Calculate individual wheel angles using Ackermann geometry
            # The sign of steer_angle_rad determines which side is inner/outer
            delta_fl = np.arctan(self.wheelbase / (turn_radius_rear_axle - np.sign(steer_angle_rad) * half_track))
            delta_fr = np.arctan(self.wheelbase / (turn_radius_rear_axle + np.sign(steer_angle_rad) * half_track))
        else:
            delta_fl = 0
            delta_fr = 0

        wheel_angles = {
            'RL': 0, # Rear wheels are fixed straight
            'RR': 0,
            'FL': delta_fl,
            'FR': delta_fr,
        }

        all_wheel_vertices = []

        # Vertices of a single wheel in its local coordinate system (centered at 0,0, aligned with x-axis)
        wheel_vertices_local_template = np.array([
            [-wheel_half_len, wheel_half_width],
            [wheel_half_len, wheel_half_width],
            [wheel_half_len, -wheel_half_width],
            [-wheel_half_len, -wheel_half_width],
            [-wheel_half_len, wheel_half_width] # Close the rectangle
        ])

        # Rotation matrix for the entire vehicle's yaw
        vehicle_rotation_matrix = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw), np.cos(self.yaw)]
        ])

        for wheel_key in ['FL', 'FR', 'RL', 'RR']: # Order for consistent plotting
            center_local = wheel_centers_local[wheel_key]
            wheel_angle = wheel_angles[wheel_key]

            # 1. Rotate wheel vertices by its individual steering angle
            wheel_rotation_matrix = np.array([
                [np.cos(wheel_angle), -np.sin(wheel_angle)],
                [np.sin(wheel_angle), np.cos(wheel_angle)]
            ])
            rotated_wheel_vertices = (wheel_rotation_matrix @ wheel_vertices_local_template.T).T

            # 2. Translate wheel vertices to its position in the vehicle's local frame
            wheel_vertices_in_vehicle_frame = rotated_wheel_vertices + center_local

            # 3. Rotate and translate wheel vertices to global coordinates
            global_wheel_vertices = (vehicle_rotation_matrix @ wheel_vertices_in_vehicle_frame.T).T + np.array([self.x, self.y])

            all_wheel_vertices.append((global_wheel_vertices[:, 0], global_wheel_vertices[:, 1]))

        return all_wheel_vertices


# --- Main Simulation Loop ---
def main():
    # Initialize vehicle object
    vehicle = Vehicle(
        x=INITIAL_STATE[0],
        y=INITIAL_STATE[1],
        yaw=INITIAL_STATE[2],
        speed=INITIAL_SPEED, # Initial speed
        wheelbase=WHEELBASE,
        track_width=TRACK_WIDTH,
        max_steer_angle_rad=np.radians(MAX_STEER_ANGLE_DEG)
    )

    # Store simulation history for plotting (grow as simulation runs)
    x_history = [vehicle.x]
    y_history = [vehicle.y]
    yaw_history = [vehicle.yaw]
    speed_history = [vehicle.speed]
    steer_angle_history_rad = [0.0]

    # Real-time control state
    current_speed_cmd = 0.0
    current_steer_cmd = 0.0
    paused = False
    held_keys = set()

    # Control tuning (per-second increments when key held)
    SPEED_INC_PER_SEC = 1.0  # m/s^2 equivalent for key hold
    STEER_INC_PER_SEC = 40.0  # deg/s when holding left/right

    # Helper to reset simulation
    def reset_sim():
        nonlocal vehicle, x_history, y_history, yaw_history, speed_history, steer_angle_history_rad, current_speed_cmd, current_steer_cmd
        vehicle.x, vehicle.y, vehicle.yaw = INITIAL_STATE
        vehicle.speed = INITIAL_SPEED
        current_speed_cmd = INITIAL_SPEED
        current_steer_cmd = 0.0
        x_history = [vehicle.x]
        y_history = [vehicle.y]
        yaw_history = [vehicle.yaw]
        speed_history = [vehicle.speed]
        steer_angle_history_rad = [0.0]

    # Key event handlers
    def on_key_press(event):
        nonlocal paused, current_speed_cmd
        key = event.key
        if key is None:
            return
        # Map arrow keys and single chars
        if key in ('left', 'right', 'up', 'down'):
            held_keys.add(key)
        elif key == ' ' or key == 'space':
            paused = not paused
        elif key == 'r':
            reset_sim()
        elif key == 's':
            # stop quickly
            held_keys.discard('up')
            held_keys.discard('down')
            # immediate stop
            current_speed_cmd = 0.0
            vehicle.speed = 0.0
        # ignore other keys

    def on_key_release(event):
        key = event.key
        if key in held_keys:
            held_keys.discard(key)


    # --- Visualization with Matplotlib Animation ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')

    # Initialize plot elements
    path_line, = ax.plot([], [], 'b-', linewidth=1.5, label='Vehicle Path')
    car_body_patch, = ax.plot([], [], 'k-', linewidth=3, label='Vehicle Body')
    # Wheels: Front-Left, Front-Right, Rear-Left, Rear-Right
    wheel_patches = [ax.plot([], [], 'k-', linewidth=2)[0] for _ in range(4)]
    
    # Text elements for dynamic title/info
    current_title = ax.set_title('')

    ax.legend()

    # Set initial plot limits (will be adjusted dynamically to follow the car)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    def init():
        """
        Initialization function for the animation.
        Sets initial empty data for all plot elements.
        """
        path_line.set_data([], [])
        car_body_patch.set_data([], [])
        for wheel_patch in wheel_patches:
            wheel_patch.set_data([], [])
        current_title.set_text('')
        return [path_line, car_body_patch, *wheel_patches, current_title]

    def update(frame):
        # Each call to update represents one time-step of length DT.
        nonlocal current_speed_cmd, current_steer_cmd, paused

        # If paused, only redraw current state
        if not paused:
            # Adjust commands based on held keys
            if 'up' in held_keys:
                current_speed_cmd += SPEED_INC_PER_SEC * DT
            if 'down' in held_keys:
                current_speed_cmd -= SPEED_INC_PER_SEC * DT
            if 'left' in held_keys:
                current_steer_cmd += STEER_INC_PER_SEC * DT
            if 'right' in held_keys:
                current_steer_cmd -= STEER_INC_PER_SEC * DT

            # Limit steer command to allowed range
            current_steer_cmd = max(-MAX_STEER_ANGLE_DEG, min(MAX_STEER_ANGLE_DEG, current_steer_cmd))

            # Propagate vehicle state one time-step
            used_steer = vehicle.update(DT, current_speed_cmd, current_steer_cmd)

            # Append histories
            x_history.append(vehicle.x)
            y_history.append(vehicle.y)
            yaw_history.append(vehicle.yaw)
            speed_history.append(vehicle.speed)
            steer_angle_history_rad.append(used_steer)

        # Use the latest state for visualization
        idx = -1
        path_line.set_data(x_history, y_history)

        temp_vehicle = Vehicle(
            x=x_history[idx],
            y=y_history[idx],
            yaw=yaw_history[idx],
            speed=speed_history[idx],
            wheelbase=WHEELBASE,
            track_width=TRACK_WIDTH,
            max_steer_angle_rad=np.radians(MAX_STEER_ANGLE_DEG)
        )

        car_x, car_y = temp_vehicle.get_car_body_vertices()
        car_body_patch.set_data(car_x, car_y)

        wheel_vertices = temp_vehicle.get_wheel_vertices(steer_angle_history_rad[idx])
        for i, (wx, wy) in enumerate(wheel_vertices):
            wheel_patches[i].set_data(wx, wy)

        # Update title with current simulation time and vehicle speed
        current_time = (len(x_history) - 1) * DT
        current_speed = speed_history[idx]
        current_title.set_text(f'Ackermann Steering Vehicle Simulation (Time: {current_time:.1f} s, Speed: {current_speed:.2f} m/s)')

        # Adjust plot limits to follow the car
        margin = 10 # meters, defines the visible area around the car
        ax.set_xlim(x_history[idx] - margin, x_history[idx] + margin)
        ax.set_ylim(y_history[idx] - margin, y_history[idx] + margin)

        return [path_line, car_body_patch, *wheel_patches, current_title]

    # Create the animation
    # Connect key events
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    ani = animation.FuncAnimation(
        fig, update, frames=range(1000000),
        init_func=init, blit=False, interval=DT * 1000, repeat=False
    )

    plt.show()

if __name__ == "__main__":
    main()