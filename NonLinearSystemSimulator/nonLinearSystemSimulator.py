import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# System parameters
a = 0.5      # damping coefficient
b = 1.0      # cubic nonlinearity coefficient
u = 0.0      # external input

# Define d(x) - you can modify this function
def d(x):
    """Nonlinear term d(x) - modify as needed"""
    return 0.2 * x  # example: linear damping term

# Define the system of ODEs
def system(state, t):
    """
    state = [x, x_dot]
    Returns [x_dot, x_ddot]
    """
    x, x_dot = state
    
    x_ddot = u - a * x_dot - b * x**3 - d(x)
    
    return [x_dot, x_ddot]

# Create phase portrait
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ===== Left plot: Vector field with trajectories =====
# Create grid for vector field
x_range = np.linspace(-3, 3, 20)
xdot_range = np.linspace(-3, 3, 20)
X, XDOT = np.meshgrid(x_range, xdot_range)

# Calculate vector field
U = np.zeros_like(X)
V = np.zeros_like(XDOT)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        derivatives = system([X[i, j], XDOT[i, j]], 0)
        U[i, j] = derivatives[0]
        V[i, j] = derivatives[1]

# Plot vector field
ax1.quiver(X, XDOT, U, V, alpha=0.6, color='lightgray')

# Plot trajectories from different initial conditions
initial_conditions = [
    [2, 0], [1.5, 1], [1, 2], [-1, 1.5],
    [-2, 0], [-1.5, -1], [0.5, -2], [2, -1.5]
]

t = np.linspace(0, 20, 500)
colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))

for i, ic in enumerate(initial_conditions):
    trajectory = odeint(system, ic, t)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], linewidth=1.5)
    ax1.plot(ic[0], ic[1], 'o', color=colors[i], markersize=8)  # starting point
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=colors[i], markersize=6)  # ending point

ax1.set_xlabel('x (Position)', fontsize=12)
ax1.set_ylabel('ẋ (Velocity)', fontsize=12)
ax1.set_title('Phase Portrait with Vector Field', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# ===== Right plot: Time series =====
# Show one trajectory as time series
ic_example = [2, 0]
trajectory_example = odeint(system, ic_example, t)

ax2.plot(t, trajectory_example[:, 0], label='x(t)', linewidth=2)
ax2.plot(t, trajectory_example[:, 1], label='ẋ(t)', linewidth=2)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('State Variables', fontsize=12)
ax2.set_title(f'Time Series (IC: x={ic_example[0]}, ẋ={ic_example[1]})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print system parameters
print(f"System parameters:")
print(f"  a (damping) = {a}")
print(f"  b (cubic term) = {b}")
print(f"  u (input) = {u}")
