import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

class Plotter:
    def __init__(self, ox, oy, resolution):
        """
        ox, oy: Obstacle X and Y coordinates lists
        resolution: Grid resolution (for visualizing grid if needed)
        """
        self.ox = ox
        self.oy = oy
        self.resolution = resolution

    def plot_results(self, path_x, path_y, hx, hy, hsteer, hcte, hv):
        """
        Generates static charts for the 'Results' section of your presentation.
        """
        plt.figure(figsize=(14, 8))

        # --- Subplot 1: Trajectory (Map) ---
        plt.subplot(2, 2, 1)
        plt.plot(self.ox, self.oy, ".k", label="Obstacles")
        plt.plot(path_x, path_y, "--r", label="Planned Path (A*)")
        plt.plot(hx, hy, "-b", label="Actual Path (Dynamic)", linewidth=2)
        plt.title("Motion Planning & Trajectory Tracking")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        # --- Subplot 2: Cross Track Error (Stability) ---
        plt.subplot(2, 2, 2)
        plt.plot(hcte, 'r')
        plt.title("Stability Analysis: Cross-Track Error")
        plt.xlabel("Time Steps")
        plt.ylabel("Error [m]")
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.grid(True)
        # Note for presentation: Explain that convergence to 0 implies stability.

        # --- Subplot 3: Control Inputs (Steering) ---
        plt.subplot(2, 2, 3)
        plt.plot(np.degrees(hsteer), 'g')
        plt.title("Control Input: Steering Angle")
        plt.xlabel("Time Steps")
        plt.ylabel("Steering [deg]")
        plt.grid(True)

        # --- Subplot 4: Velocity Profile ---
        plt.subplot(2, 2, 4)
        plt.plot(hv, 'b')
        plt.title("Velocity Profile")
        plt.xlabel("Time Steps")
        plt.ylabel("Speed [m/s]")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def animate_run(self, path_x, path_y, hx, hy, hyaw, hsteer):
        """
        Creates an animation of the vehicle driving.
        Great for the 'Physical working principles' slide.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Plot static background
        ax.plot(self.ox, self.oy, ".k", label="Obstacles")
        ax.plot(path_x, path_y, "--r", label="Planned Path")
        
        # Dynamic elements
        trail, = ax.plot([], [], "-b", linewidth=1, alpha=0.5) # The path behind the car
        car_body, = ax.plot([], [], "-k", linewidth=3)
        wheels, = ax.plot([], [], ".r", markersize=5) # Simple representation of wheels
        
        ax.set_xlim(min(path_x)-5, max(path_x)+5)
        ax.set_ylim(min(path_y)-5, max(path_y)+5)

        def update(i):
            if i >= len(hx): return trail, car_body, wheels

            # Update Trail
            trail.set_data(hx[:i], hy[:i])

            # Update Car Body (Simple Box)
            # Car Dimensions (Approximate for viz)
            L = 4.5; W = 2.0 
            cx, cy, cyaw = hx[i], hy[i], hyaw[i]
            
            # Calculate corners
            outline = np.array([
                [-L/2, L/2, L/2, -L/2, -L/2],
                [W/2, W/2, -W/2, -W/2, W/2]
            ])
            
            # Rotate
            rot = np.array([[np.cos(cyaw), -np.sin(cyaw)], [np.sin(cyaw), np.cos(cyaw)]])
            outline = (rot @ outline)
            
            # Translate
            outline[0, :] += cx
            outline[1, :] += cy
            
            car_body.set_data(outline[0, :], outline[1, :])
            
            return trail, car_body, wheels

        ani = animation.FuncAnimation(fig, update, frames=range(0, len(hx), 2), blit=True, interval=20)
        plt.title("Ackerman Dynamic Simulation")
        plt.show()