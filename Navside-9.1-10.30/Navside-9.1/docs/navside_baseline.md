# NavSide Baseline

## Frozen facts
- Encoder model: `models/vae_encoder.onnx`
- Policy model: `models/nav_policy.onnx`
- Depth embedding dim: 2560
- Obs dim: 2576
- Control gate: 5 Hz
- Walk threshold: 0.3
- Temporary navigation clamp:
  - vx: [0.0, 0.45]
  - vy: 0.0
  - wz: [-0.4, 0.4]
- Goal stop: enabled

## Intent
Keep only the SRU navigation core required for sim2sim.
Do not bring VIPlanner legacy paths into NavSide.

## Sim mode
- `--sim-control`: MuJoCo depth + SRU policy + UDP command send to robot-side `8080`.
- Preferred robot-side state is `NavStatePacketV2`, 84 bytes, parsed as `<IHHId16f`.
