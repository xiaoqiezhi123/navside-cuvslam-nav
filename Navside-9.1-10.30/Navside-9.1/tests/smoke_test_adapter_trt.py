#!/usr/bin/env python3
"""Adapter smoke test for the TensorRT backend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

from navside.adapter import SruNavAdapter
from navside.state import SruRobotState


def load_backend_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "nav.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def build_state() -> SruRobotState:
    return SruRobotState(
        linear_vel_b=np.zeros(3, dtype=np.float32),
        angular_vel_b=np.zeros(3, dtype=np.float32),
        projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        robot_pos_w=np.array([0.0, 0.0, 0.695], dtype=np.float32),
        robot_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def main() -> None:
    cfg = load_backend_config()
    config_dir = Path(__file__).resolve().parents[1] / "config"
    model_cfg = cfg.get("models", {})
    control_cfg = cfg.get("control", {})
    depth_cfg = cfg.get("depth", {})

    adapter = SruNavAdapter(
        encoder_path=str((config_dir / model_cfg.get("encoder_path", "../asset/models/vae_pretrain_new.plan")).resolve()),
        policy_path=str((config_dir / model_cfg.get("policy_path", "../asset/models/policy_1.plan")).resolve()),
        dry_run_hz=float(control_cfg.get("dry_run_hz", 5.0)),
        min_depth=float(depth_cfg.get("min_depth", 0.25)),
        max_depth=float(depth_cfg.get("max_depth", 10.0)),
        verbose=False,
        inference_backend=str(model_cfg.get("inference_backend", "tensorrt")),
        encoder_engine_path=str((config_dir / model_cfg.get("encoder_engine_path", model_cfg.get("encoder_path", "../asset/models/vae_pretrain_new.plan"))).resolve()),
        policy_engine_path=str((config_dir / model_cfg.get("policy_engine_path", model_cfg.get("policy_path", "../asset/models/policy_1.plan"))).resolve()),
    )

    assert adapter.inference_backend == "tensorrt"
    assert "onnxruntime" not in sys.modules, "onnxruntime must not be imported"

    depth_tensor = np.random.default_rng(0).standard_normal((1, 1, 40, 64), dtype=np.float32)
    depth_features = adapter.trt_runner.encode_depth(np.ascontiguousarray(depth_tensor, dtype=np.float32))
    assert tuple(depth_features.shape) == (1, 64, 5, 8)

    depth_img = np.linspace(
        0.2,
        8.0,
        num=1200 * 1920,
        dtype=np.float32,
    ).reshape(1200, 1920)
    state = build_state()
    target = np.array([5.0, 1.0, 0.695], dtype=np.float32)

    diag = adapter.step(depth_img=depth_img, state=state, target_pos_w=target, timestamp=0.0)
    assert diag is not None
    assert tuple(diag["depth_feature_shape"].tolist()) == (2560,)
    assert tuple(diag["obs_shape"].tolist()) == (1, 2576)
    assert tuple(diag["raw_action"].shape) == (3,)
    assert tuple(diag["cmd_vel"].shape) == (3,)

    print("adapter backend=tensorrt smoke passed")


if __name__ == "__main__":
    main()
