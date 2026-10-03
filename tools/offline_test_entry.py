"""Isolated suite entry: disable private dotenv and deny external sockets."""
import os
from pathlib import Path
import runpy
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(PYTHON_DOTENV_DISABLED="1", OPENAI_API_KEY="offline-disabled",
                  MODEL="offline-disabled", OPENAI_BASE_URL="http://127.0.0.1:9/v1")
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
connect, connect_ex = socket.socket.connect, socket.socket.connect_ex


def allowed(address):
    if isinstance(address, tuple) and (
            address[0] not in {"127.0.0.1", "::1", "localhost"} or address[1] == 9):
        raise AssertionError("External/default provider network forbidden in repository verification")


def safe_connect(sock, address):
    allowed(address)
    return connect(sock, address)


def safe_connect_ex(sock, address):
    allowed(address)
    return connect_ex(sock, address)


socket.socket.connect, socket.socket.connect_ex = safe_connect, safe_connect_ex
target = sys.argv[1]
if target == "pytest":
    import pytest
    raise SystemExit(pytest.main(sys.argv[2:]))
sys.argv = [target]
runpy.run_path(target, run_name="__main__")
