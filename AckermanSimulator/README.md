# Ackermann Simulator


## Controls:
- Hold `Up` / `Down`: Increase / decrease speed command
- Hold `Left` / `Right`: Steer left / right (degrees)
- `Space`: Pause / Unpause
- `r`: Reset simulation
- `s`: Immediate stop (zero speed)
- `e`: Toggle external schedule on/off

## Using external schedules:
- Sample schedule CSV available at `schedules/sample_schedule.csv`.
- In Python, load it before running the simulator (from the project root):

- Run the simulator and press `e` to enable the schedule. Scheduled entries override keyboard input while active.

Schedule CLI and CSV

- You can load a schedule at startup using the simulator's built-in CLI flags:

```bash
python AckermanSimulator/Simulator.py --schedule schedules/sample_schedule.csv --enable
```

- Flags:
	- `--schedule, -s PATH` : Path to a CSV schedule file to load at startup.
	- `--enable` : If present, the loaded schedule is enabled immediately (otherwise press `e` in the running window to toggle).


### External Inputs
- CSV format: header columns `start,end,speed,steer` (times in seconds, `speed` in m/s, `steer` in degrees). Example row:

```
0.0,3.0,2.0,10.0
```

Parametric schedules (ramps)

You can create compact parametric rows using `mode=ramp` (or `mode=accel`) and either `v0`+`accel` or `v0`+`v1` to specify a linear ramp. The loader will expand these rows into per-`DT` entries automatically.

Example `schedules/ramp_schedule.csv`:

```
start,end,mode,v0,accel,steer
0.0,5.0,ramp,0.0,0.5,0.0
5.0,10.0,ramp,2.5,0.0,10.0
```

- `v0`: initial speed at `start` (m/s)
- `accel`: constant acceleration (m/s^2)
- loader expands ramp rows into `DT`-sized steps so the simulator can match intervals without changing runtime logic.
