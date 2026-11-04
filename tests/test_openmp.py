#!/usr/bin/env python3

"""tests for cotorra.openmp, which re-runs a macOS process with torch's libomp"""

import importlib.machinery
import os
import pathlib
import subprocess
import sys
import tomllib
import types

import pytest

import cotorra.cli
from cotorra import openmp

ROOT = pathlib.Path(__file__).resolve().parents[1]


class Rerun(Exception):
    """raised by the `os.execve` these tests put in place of the real one"""


@pytest.fixture
def mac(monkeypatch, tmp_path) -> pathlib.Path:
    """
    a macOS process outside any debugger or kernel, its environment holding
    neither the re-run's flag nor a `DYLD_LIBRARY_PATH`, whose torch is a stand-in
    with a `libomp.dylib` in its `lib`, returned
    """
    lib = tmp_path / "site-packages" / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib.parent / "__init__.py").touch()
    (lib / "libomp.dylib").touch()
    spec = importlib.machinery.ModuleSpec(
        "torch", None, origin=str(lib.parent / "__init__.py")
    )
    find_spec = openmp.importlib.util.find_spec
    monkeypatch.setattr(
        openmp.importlib.util,
        "find_spec",
        lambda name, *args: spec if name == "torch" else find_spec(name, *args),
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv(openmp.RERUN_FLAG, raising=False)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    for name in openmp.IN_SESSION:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return lib.resolve()


@pytest.fixture
def execve(monkeypatch) -> list:
    """records each call to `os.execve`, raising `Rerun` in place of the exec"""
    calls = []

    def record(path, args, env):
        calls.append((path, args, env))
        raise Rerun

    monkeypatch.setattr(openmp.os, "execve", record)
    return calls


def test_a_rerun_puts_torchs_lib_first_and_flags_itself(mac, monkeypatch):
    monkeypatch.setenv("DYLD_LIBRARY_PATH", f"/elsewhere{os.pathsep}/further")
    monkeypatch.setenv("KEPT", "as it was")
    env = openmp.rerun_env()
    assert env["DYLD_LIBRARY_PATH"].split(os.pathsep) == [
        str(mac),
        "/elsewhere",
        "/further",
    ]
    assert env[openmp.RERUN_FLAG] == "1"
    assert env["KEPT"] == "as it was"
    assert env.keys() - os.environ.keys() == {openmp.RERUN_FLAG}


def test_no_rerun_off_macos(mac, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert openmp.rerun_env() is None


def test_no_rerun_after_a_rerun_or_when_told_not_to(mac, monkeypatch):
    """the flag a re-run sets, which keeps one that loses `DYLD_LIBRARY_PATH` on
    the way from re-running forever, or a user's"""
    monkeypatch.setenv(openmp.RERUN_FLAG, "1")
    assert openmp.rerun_env() is None


def test_no_rerun_with_torchs_lib_on_the_path_already(mac, monkeypatch, tmp_path):
    """however it's named there"""
    (link := tmp_path / "link").symlink_to(mac)
    monkeypatch.setenv("DYLD_LIBRARY_PATH", f"/elsewhere{os.pathsep}{link}")
    assert openmp.rerun_env() is None


def test_no_rerun_where_torch_brings_no_libomp(mac):
    (mac / "libomp.dylib").unlink()
    assert openmp.torch_lib() is None
    assert openmp.rerun_env() is None


def test_no_rerun_without_torch(mac, monkeypatch):
    monkeypatch.setattr(openmp.importlib.util, "find_spec", lambda name: None)
    assert openmp.rerun_env() is None


@pytest.mark.parametrize("name", sorted(openmp.IN_SESSION))
def test_no_rerun_in_a_debugger_or_a_kernel(mac, monkeypatch, name):
    """whose session the exec would end"""
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    assert openmp.rerun_env() is None


def test_rerun_runs_the_same_command_again(execve, monkeypatch):
    """the same interpreter, its options, and the script or module and its
    arguments, as `sys.orig_argv` gives them"""
    monkeypatch.setattr(sys, "orig_argv", ["python3", "-X", "dev", "cotorra", "-h"])
    with pytest.raises(Rerun):
        openmp.rerun({"A": "B"})
    assert execve == [
        (sys.executable, [sys.executable, "-X", "dev", "cotorra", "-h"], {"A": "B"})
    ]


def test_main_reruns_before_it_runs_the_cli(mac, execve, monkeypatch):
    ran = []
    monkeypatch.setattr(cotorra.cli, "main", lambda: ran.append("cli"))
    with pytest.raises(Rerun):
        openmp.main()
    assert len(execve) == 1 and not ran
    monkeypatch.setenv(openmp.RERUN_FLAG, "1")
    openmp.main()
    assert ran == ["cli"] and len(execve) == 1


def test_the_cotorra_command_is_main():
    """read off pyproject.toml, which an editable install's metadata lags"""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"] == {"cotorra": "cotorra.openmp:main"}


@pytest.mark.skipif(
    sys.platform != "darwin" or openmp.torch_lib() is None,
    reason="a re-run happens only on macOS, with a torch bringing its libomp",
)
def test_a_real_rerun_starts_python_again_with_torchs_lib_first():
    """by exec, so that the command's output and exit status are the re-run's"""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in (openmp.RERUN_FLAG, "DYLD_LIBRARY_PATH")
    } | {"PYTHONPATH": str(ROOT / "src")}  # the working tree, as pytest imports
    script = (
        "import os, sys\n"
        "from cotorra import openmp\n"
        "if (env := openmp.rerun_env()) is not None:\n"
        "    openmp.rerun(env)\n"
        "print(os.environ['DYLD_LIBRARY_PATH'], os.environ[openmp.RERUN_FLAG])\n"
        "sys.exit(3)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert done.returncode == 3, done.stderr
    assert done.stdout.split() == [str(openmp.torch_lib()), "1"]
