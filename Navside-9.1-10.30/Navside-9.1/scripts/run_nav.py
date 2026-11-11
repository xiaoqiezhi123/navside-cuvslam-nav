#!/usr/bin/env python3
import os
import time
from pathlib import Path
import sys


def _reexec_into_repo_venv() -> None:
    """Run NavSide under the repo-local venv if it exists.

    This keeps users from accidentally launching the script with a system
    interpreter that has incompatible numpy/scipy wheels.
    """

    if os.environ.get("NAVSIDE_SKIP_VENV_REEXEC") == "1":
        return

    script_root = Path(__file__).resolve().parents[1]
    workspace_root = script_root.parent
    candidates = (
        workspace_root / ".venv_navside" / "bin" / "python",
        script_root / ".venv_navside" / "bin" / "python",
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _timing_enabled() -> bool:
    return os.environ.get("NAVSIDE_TIMING", "") == "1" or os.environ.get("SRU_TIMING", "") == "1"


def _timing_log(stage: str, seconds: float) -> None:
    if _timing_enabled():
        print(f"[NavSide TIMING] {stage}: {seconds:.6f}s")

if _timing_enabled() and "NAVSIDE_RUN_NAV_START_T0" not in os.environ:
    os.environ["NAVSIDE_RUN_NAV_START_T0"] = str(time.perf_counter())

_IMPORT_RUNTIME_T0 = time.perf_counter()
from navside.runtime import main
if _timing_enabled():
    run_nav_start_t0 = float(os.environ.get("NAVSIDE_RUN_NAV_START_T0", str(_IMPORT_RUNTIME_T0)))
    _timing_log("run_nav_entry_to_import_navside_runtime", time.perf_counter() - run_nav_start_t0)
    _timing_log("run_nav_import_navside_runtime", time.perf_counter() - _IMPORT_RUNTIME_T0)


if __name__ == "__main__":
    main()
