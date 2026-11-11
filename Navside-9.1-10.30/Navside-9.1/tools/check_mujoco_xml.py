#!/usr/bin/env python3
import argparse
from pathlib import Path

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that a MuJoCo XML loads for NavSide.")
    parser.add_argument("xml", help="Path to MuJoCo XML.")
    parser.add_argument("--camera-name", default="head_camera", help="Required camera name.")
    args = parser.parse_args()

    xml_path = Path(args.xml).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)

    print(f"loaded={xml_path}")
    print(f"nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh} ncam={model.ncam}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"camera_name={args.camera_name} camera_id={cam_id}")

    if cam_id == -1:
        raise SystemExit(f"missing required camera: {args.camera_name}")


if __name__ == "__main__":
    main()
