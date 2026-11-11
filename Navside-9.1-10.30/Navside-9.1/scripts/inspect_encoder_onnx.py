#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def _reexec_into_repo_venv() -> None:
    if os.environ.get("NAVSIDE_SKIP_VENV_REEXEC") == "1":
        return

    script_root = Path(__file__).resolve().parents[1]
    workspace_root = script_root.parent
    candidates = (
        workspace_root / ".venv_navside" / "bin" / "python",
        script_root / ".venv_navside" / "bin" / "python",
        Path("/home/amov/nav_arm_mujoco/.venv_navside/bin/python"),
    )
    venv_python = None
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            venv_python = candidate
            break
    if venv_python is None:
        return

    current_prefix = Path(sys.prefix).resolve()
    venv_root = venv_python.parents[1].resolve()
    if current_prefix == venv_root:
        return

    os.environ["NAVSIDE_SKIP_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_into_repo_venv()

import onnxruntime as ort


def main() -> None:
    model_path = Path(__file__).resolve().parents[1] / "asset" / "models" / (
        "vae_regnet-d455f-pretrain_best.onnx"
    )
    print("model:", model_path)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    for tensor in session.get_inputs():
        print("input:", tensor.name, tensor.shape, tensor.type)
    for tensor in session.get_outputs():
        print("output:", tensor.name, tensor.shape, tensor.type)


if __name__ == "__main__":
    main()
