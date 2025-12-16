# Ackerman_Simulator

This folder contains a lightweight Ackerman steering simulator with interchangeable controllers, kinematic/dynamic models, and simple planners.

## Quick Start

1) Create/activate the project venv

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r ../requirements.txt
# or if running standalone from this folder
pip install -r requirments.txt  # note: file name has a typo
```

3) Run a quick check

```bash
python test_maze_quick.py
```

You should see progress logs over time steps. This is a smoke test.

## Running the Simulator

- Main demo:
```bash
python main.py
```

- Compare models:
```bash
python compare_models.py
```

- Create/visualize vehicle comparisons:
```bash
python create_vehicle_comparison.py
python visualize_scenario.py
```

- Validation & debug helpers:
```bash
python run_validation.py
python test_debug.py
python test_vehicle_viz.py
```

## Project Layout

- Controllers: PID-like steering strategies
  - [controllers/lqr.py](controllers/lqr.py)
  - [controllers/pure_pursuit.py](controllers/pure_pursuit.py)
  - [controllers/stanley.py](controllers/stanley.py)

- Models: vehicle motion
  - [models/vehicle_kinematics.py](models/vehicle_kinematics.py) — kinematic bicycle model
  - [models/vehicle_dynamics.py](models/vehicle_dynamics.py) — simplified dynamic model
  - [models/model_factory.py](models/model_factory.py) — factory to choose model/controller

- Planners: path generation
  - [planners/a_star.py](planners/a_star.py)

- Utils: plotting & helpers
  - [utils/plotting.py](utils/plotting.py)

- Entry points / demos
  - [main.py](main.py) — typical run
  - [compare_models.py](compare_models.py)
  - [create_vehicle_comparison.py](create_vehicle_comparison.py)
  - [visualize_scenario.py](visualize_scenario.py)

## Configuration

Most scripts have parameters near the top (e.g., speed, wheelbase, controller choice). Edit the script directly for quick experimentation or add your own config module.

- Switch controller/model via `model_factory.py`.
- Adjust plotting in `utils/plotting.py`.

## Troubleshooting

- Virtual environment not active:
  - Ensure `source .venv/bin/activate` before running.

- Missing dependencies:
  - Install with `pip install -r ../requirements.txt` (root-level) or `pip install -r requirments.txt` (local, note the file name typo).

- Matplotlib backend issues on macOS:
  - Try `pip install pillow` and set `matplotlib.use("Agg")` for headless environments.

- Version conflicts (NumPy/SciPy):
  - Use recent versions (NumPy ≥ 2.0, SciPy ≥ 1.13). If you see build errors, pin to versions in `requirements.txt`.

- FFmpeg missing (video export fails):
  - Install FFmpeg via Homebrew:
    ```bash
    brew update
    brew install ffmpeg
    ```
  - After installation, rerun `visualize_scenario.py`. If FFmpeg is unavailable, the script automatically falls back to GIF export using Pillow.

## Extending

- Add a new controller under `controllers/` and register it in `model_factory.py`.
- Implement alternative dynamics in `models/`.
- Build planners to generate waypoint sequences.

## Example Commands

```bash
# Run main with venv
source ../.venv/bin/activate
python main.py

# Compare kinematic vs dynamic behavior
python compare_models.py

# Visualize a scenario
python visualize_scenario.py
```
