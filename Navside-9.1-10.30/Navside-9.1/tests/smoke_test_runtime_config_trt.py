#!/usr/bin/env python3
"""Runtime/config smoke test for the TensorRT backend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from navside.runtime import load_nav_config
from navside.adapter import SruNavAdapter


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "nav.yaml"
    cfg = load_nav_config(str(config_path))

    assert cfg.inference_backend == "tensorrt"
    assert cfg.encoder_engine_path.endswith("vae_pretrain_new.plan")
    assert cfg.policy_engine_path.endswith("policy_1.plan")
    assert cfg.encoder_path.endswith(".onnx")
    assert cfg.policy_path.endswith(".onnx")

    adapter = SruNavAdapter(
        encoder_path=cfg.encoder_path,
        policy_path=cfg.policy_path,
        dry_run_hz=cfg.dry_run_hz,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        verbose=False,
        inference_backend=cfg.inference_backend,
        encoder_engine_path=cfg.encoder_engine_path,
        policy_engine_path=cfg.policy_engine_path,
    )

    assert adapter.inference_backend == "tensorrt"
    assert adapter.encoder_engine_path.endswith("vae_pretrain_new.plan")
    assert adapter.policy_engine_path.endswith("policy_1.plan")
    assert adapter.encoder_path.endswith(".onnx")
    assert adapter.policy_path.endswith(".onnx")
    assert "onnxruntime" not in sys.modules, "onnxruntime must not be imported"

    depth = np.random.default_rng(0).standard_normal((1, 1, 40, 64), dtype=np.float32)
    obs = np.random.default_rng(1).standard_normal((1, 2576), dtype=np.float32)
    h_in = np.random.default_rng(2).standard_normal((1, 1, 512), dtype=np.float32)
    c_in = np.random.default_rng(3).standard_normal((1, 1, 512), dtype=np.float32)

    depth_features = adapter.trt_runner.encode_depth(np.ascontiguousarray(depth, dtype=np.float32))
    actions, h_out, c_out = adapter.trt_runner.run_policy(
        np.ascontiguousarray(obs, dtype=np.float32),
        np.ascontiguousarray(h_in, dtype=np.float32),
        np.ascontiguousarray(c_in, dtype=np.float32),
    )

    assert tuple(depth_features.shape) == (1, 64, 5, 8)
    assert tuple(actions.shape) == (1, 3)
    assert tuple(h_out.shape) == (1, 1, 512)
    assert tuple(c_out.shape) == (1, 1, 512)

    print("runtime config tensorrt smoke passed")


if __name__ == "__main__":
    main()
