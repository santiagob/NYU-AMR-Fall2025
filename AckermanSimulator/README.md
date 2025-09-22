# Ackermann Simulator

Quick reference for the updated CLI and scheduling workflow used by this simulator.

## Controls (interactive GUI)
- Hold `Up` / `Down`: Increase / decrease speed command
- Hold `Left` / `Right`: Steer left / right (degrees)
- `Space`: Pause / Unpause
- `r`: Reset simulation
- `s`: Immediate stop (zero speed)
- `e`: Toggle external schedule on/off

When an external schedule is enabled, scheduled entries override keyboard commands for the duration of the active interval.

## Schedule files
- Example schedules are stored in `AckermanSimulator/schedules/` (e.g. `example_constant.csv`, `example_ramp_turn.csv`, `example_maneuver.csv`).

Format (CSV, header optional):
- Standard constant row: `start,end,speed,steer` (times in seconds, `speed` in m/s, `steer` in degrees).
	- Example: `0.0,3.0,2.0,10.0`

- Parametric ramp rows: include `mode=ramp` (or `mode=accel`) and either `v0`+`accel` or `v0`+`v1`.
	- Example:
		```csv
		start,end,mode,v0,accel,steer
		0.0,5.0,ramp,0.0,0.5,0.0
		5.0,10.0,ramp,2.5,0.0,10.0
		```
	- The loader expands ramp rows into per-`DT` (time-step) entries automatically so the simulation can match intervals precisely.

## CLI Usage (headless or interactive)

- Run interactive GUI and load a schedule (press `e` to toggle schedule if you don't pass `--enable`):
```bash
python AckermanSimulator/Simulator.py -s example_constant.csv --enable
```

- Headless export (runs a deterministic batch simulation and writes images/CSV):
```bash
python AckermanSimulator/Simulator.py -s example_ramp_turn.csv --enable --export myrun
```
This creates these files under `AckermanSimulator/output/` by default:
- `myrun_trajectory.png` (aerial path with start/end heading arrows)
- `myrun_dynamics.png` (speed, steer, yaw_rate, lateral accel, yaw)
- `myrun_history.csv` (time series CSV)

Notes on `--schedule` resolution
- The `--schedule`/`-s` argument accepts:
	1. An exact path you've provided (absolute or relative), or
	2. A basename which will be looked up in `AckermanSimulator/schedules/`, or
	3. The basename with `.csv` appended if needed.

- If the loader cannot find the schedule, the script now prints a clear error and exits without creating any output files. The error shows the absolute paths it attempted, e.g.:
```
ERROR: schedule file not found. Tried the following absolute paths:
	- /full/path/you/passed
	- /full/path/to/AckermanSimulator/schedules/you/passed
	- /full/path/to/AckermanSimulator/schedules/you/passed.csv
```
(the `ERROR` line is printed in red in terminals that support ANSI colors)

Export path behavior
- `--export <name>`: if `<name>` contains no path separators it is treated as a basename and the outputs are written to `AckermanSimulator/output/<name>_...`.
- If you supply a path that begins with `output/`, it will be routed to the script-local `AckermanSimulator/output/` folder for consistency.
- If you pass an absolute path or another directory explicitly, the script will respect it and write outputs there (creating the directory if necessary).
