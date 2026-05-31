# Reachy Mini Morning Greeter

Simple app for Reachy Mini that:
- detects faces from camera frames,
- greets a new person with a morning intro,
- avoids repeatedly greeting the same tracked person.

## 1) Setup

From the project folder:

```powershell
uv sync
```

Optional: install simulation dependencies

```powershell
uv sync --extra sim
```

## 2) Run

### Real Reachy Mini (recommended with explicit IP)

```powershell
uv run python morning_greeter.py --robot-host <REACHY_IP> --robot-port 8000 --connection-mode network --show-video --debug-detections
```

Example:

```powershell
uv run python morning_greeter.py --robot-host 192.168.1.42 --connection-mode network --show-video --debug-detections
```

### Simulation mode

Terminal A:

```powershell
uv run reachy-mini-daemon --sim
```

Terminal B:

```powershell
uv run python morning_greeter.py --sim --show-video --debug-detections
```

## 3) Useful Flags

- `--save-detection-frames 10` save first 10 detection frames
- `--save-dir debug_detections` output folder for saved debug images
- `--same-person-cooldown 300` seconds before greeting same tracked person again
- `--robot-media-backend default|local|webrtc` choose media backend
- `--fallback-webcam-if-no-robot-video` fallback to webcam if robot stream is unavailable
- `--webcam-backend auto|msmf|dshow|default` webcam backend selection
- `--webcam-index N` webcam device index

## 4) Troubleshooting

If `reachy-mini.local` does not resolve on Windows, use robot IP directly with:

```powershell
--robot-host <REACHY_IP> --connection-mode network
```

If no frames arrive, keep `--debug-detections` enabled and verify connection to daemon docs page:

- local daemon: http://127.0.0.1:8000/docs
- remote robot: http://<REACHY_IP>:8000/docs
