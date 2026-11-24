#!/usr/bin/env python3

"""
on macOS, re-runs a process so that torch's OpenMP runtime is the only one loaded
"""

import importlib.util
import os
import pathlib
import sys
import typing

# LightGBM's and XGBoost's macOS wheels link `@rpath/libomp.dylib`, which their
# rpaths resolve to Homebrew's libomp, while torch loads the copy in its own
# `lib`. Without Homebrew's, importing either booster fails; with it, a process
# holds two OpenMP runtimes, and the next thing torch runs on its OpenMP threads
# segfaults, whether or not a booster ever fits. Putting torch's `lib` first on
# `DYLD_LIBRARY_PATH` resolves every `libomp.dylib` to torch's, but macOS's
# dynamic linker reads it only as a process starts, so the process has to run
# again. Only the standard library is imported here: this runs before anything
# that loads a booster can.

# set in the environment of the process `rerun` starts, so that it never re-runs
# again (as it would forever were `DYLD_LIBRARY_PATH` dropped on the way, as macOS
# drops it on the way into the programs it protects); set it yourself to keep
# cotorra from re-running at all
RERUN_FLAG = "COTORRA_OPENMP_RERUN"

# loaded where exec would end a session someone is in: a debugger's (pydevd,
# which VS Code's debugpy and PyCharm both run) or a Jupyter kernel's
IN_SESSION = {"pydevd", "ipykernel"}


def torch_lib() -> pathlib.Path | None:
    """torch's `lib` directory if it holds a `libomp.dylib`, found without
    importing torch, which would load that libomp"""
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    lib = pathlib.Path(spec.origin).resolve().parent / "lib"
    return lib if (lib / "libomp.dylib").is_file() else None


def rerun_env() -> dict[str, str] | None:
    """
    the environment to run this process again in, with torch's `lib` first on
    `DYLD_LIBRARY_PATH`; None off macOS, where it's on that path already, where
    torch brings no libomp, after a re-run or under `RERUN_FLAG`, and in a
    debugger or a Jupyter kernel
    """
    if sys.platform != "darwin" or os.environ.get(RERUN_FLAG):
        return None
    if IN_SESSION & sys.modules.keys() or (lib := torch_lib()) is None:
        return None
    paths = [p for p in os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep) if p]
    if any(pathlib.Path(p).resolve() == lib for p in paths):
        return None
    dyld_path = os.pathsep.join([str(lib), *paths])
    return os.environ | {"DYLD_LIBRARY_PATH": dyld_path, RERUN_FLAG: "1"}


def rerun(env: dict[str, str]) -> typing.NoReturn:
    """replace this process with the same command, run with the same interpreter
    and arguments, in `env`"""
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


def main() -> None:
    """the `cotorra` command: `cotorra.cli.main`, re-run first where need be"""
    if (env := rerun_env()) is not None:
        rerun(env)
    from cotorra.cli import main as cli_main

    cli_main()
