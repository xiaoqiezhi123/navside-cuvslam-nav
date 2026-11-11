# NavSide

This directory is the SRU-only navigation side of `sru_mujoco_sim`.

## What it contains
- `run_nav.py`: thin launcher entry.
- `navside/`: actual implementation package.
  - `navside/runtime.py`: top-level orchestration and app entry for `--sim-control`.
  - `navside/adapter.py`: depth -> feature -> obs -> ONNX -> action -> clamp.
  - `navside/bridge.py`: UDP packet parsing and command transport.
  - `navside/depth.py`: MuJoCo depth rendering helpers.
  - `navside/sim.py`: MuJoCo SRU runner for `--sim-control`.
  - `navside/state.py`: shared SRU robot-state dataclass.
- `config/nav.yaml`: local config with model paths and limits.
- `asset/`: copied ONNX models, robot XML, URDF, meshes, and textures.

## Current contract
- depth feature: 2560
- obs size: 2576
- policy tick: 5 Hz
- action output: 3 floats
- fail-safe: zero cmd on invalid inputs / goal reached

## Sim mode
`run_nav.py` launches the MuJoCo SRU control loop:

```bash
python3 -B run_nav.py --sim-control
```

`--sim-control` opens the MuJoCo scene, renders depth, runs SRU inference,
and sends `vx, vy, wz` to robot-side UDP port `8080`.

## Robot-side state packets
NavSide expects `NavStatePacketV2`: 84 bytes, parsed as `<IHHId16f`

`NavStatePacketV2` contains:

- `magic`: `0x32555253`
- `version`: `2`
- `seq`
- `timestamp_sec`
- `linear_vel_b[3]`
- `angular_vel_b[3]`
- `projected_gravity_b[3]`
- `robot_pos_w[3]`
- `robot_quat_wxyz[4]`

NavSide uses it directly for SRU obs construction.

## Runtime note
This is the navigation core only. Robot-side state publishing and command
transport are expected to be provided by `robotside`.

## Depth camera backend

Real mode selects the camera through `depth.backend` in `config/nav.yaml`:

- `realsense`: RealSense D455F depth stream `1280x720@30 Z16`, converted to
  float32 meters, center-cropped to `1152x720`, then resized to `64x40`.
- `zed`: existing ZED Mini backend, kept for fallback.

The active VAE encoder is:

```text
asset/models/vae_encoder.onnx
```

To test the RealSense backend directly:

```bash
python3 -B scripts/test_realsense_depth_perception.py
```

To inspect the raw RealSense depth in a live window:

```bash
python3 -B scripts/view_realsense_depth.py
```

Real mode can also open the viewer with:

```bash
python3 -B scripts/run_nav.py --real --show-depth --config config/nav.yaml
```

The offline conversion/factory smoke test is:

```bash
python3 -B tests/smoke_test_realsense_depth_perception.py
```
