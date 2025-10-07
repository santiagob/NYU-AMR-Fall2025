# Ackermann Dynamics Simulator

Quick reference for the updated CLI and scheduling workflow used by this simulator.

## Features

- Dynamic bicycle model with realistic tire and vehicle parameters
- Supports both forward and reverse driving
- Interactive GUI with keyboard controls for speed and steering
- Batch simulation using schedules loaded from CSV files
- Export of trajectory and dynamics plots, as well as simulation data in CSV format

## Requirements

- Python 3.7+
- numpy
- matplotlib

## Usage

### Interactive Mode

Run the simulator with:

```bash
python Simulator.py
```

**Controls:**
- Arrow keys: Increase/decrease speed and steering
- Space: Pause/resume simulation
- R: Reset simulation
- E: Toggle external schedule (if loaded)
- S: Stop vehicle

### Batch Mode (Export)

To run a batch simulation and export results:

```bash
python Simulator.py --schedule path/to/schedule.csv --export output/sim1
```

This will generate:
- `output/sim1_trajectory.png`: Trajectory plot
- `output/sim1_dynamics.png`: State/dynamics plots
- `output/sim1_history.csv`: Simulation data

### Schedule CSV Format

A schedule CSV should have columns like:

```
start,end,speed,steer
0,5,5,0
5,10,10,20
...
```

## File Structure

- `Simulator.py`: Main simulator script
- `schedules/`: Example schedule CSVs (optional)
- `output/`: Exported images and CSVs (created automatically)

