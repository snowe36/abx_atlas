#!/usr/bin/env python3
"""Orchestrate a GPU training run on RunPod: create a pod, sync code+data up,
run bootstrap.sh (GNN + pretrained transformer training/HPO), pull results
back into this repo, then stop the pod.

Usage:
    python scripts/runpod/launch_gpu_job.py [options]

Prerequisites:
    - pip install -e ".[gpu]"   (for the `runpod` SDK locally)
    - a RunPod API key in .env as RUNPOD_API_KEY (or the existing API_KEY)
    - an SSH public key added to your RunPod account (Settings > SSH Keys)
      so the pod accepts your local private key automatically

Safety: every pod gets a self-terminating watchdog baked into bootstrap.sh
that fires after --max-runtime-hours (default 4h) regardless of what this
script does — robust to a hung job or a dropped connection. On normal
completion this script also stops the pod immediately (the watchdog is just
the backstop). Pass --keep-alive to leave the pod up for interactive SSH use
(the watchdog still applies), or --terminate to delete it outright instead of
just stopping it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
REMOTE_DIR = "/workspace/abx_atlas"
RSYNC_EXCLUDES = [
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data/raw",
    "*.pyc",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
]


def _load_api_key() -> str:
    for name in ("RUNPOD_API_KEY", "API_KEY"):
        val = os.environ.get(name)
        if val:
            return val
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key in ("RUNPOD_API_KEY", "API_KEY") and val:
                return val
    raise SystemExit(
        "No RunPod API key found. Set RUNPOD_API_KEY (or API_KEY) in .env or the environment."
    )


def _ssh_opts(port: int) -> list[str]:
    opts = [
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
    ]
    # Prefer a dedicated RunPod key if present; otherwise fall back to the
    # agent's default identities (user must have added a pubkey to RunPod).
    for candidate in (
        Path.home() / ".ssh" / "runpod_ed25519",
        Path.home() / ".ssh" / "id_ed25519",
        Path.home() / ".ssh" / "id_rsa",
    ):
        if candidate.exists():
            opts.extend(["-i", str(candidate), "-o", "IdentitiesOnly=yes"])
            break
    return opts


def wait_for_ssh(runpod_mod, pod_id: str, timeout_s: int) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        pod = runpod_mod.get_pod(pod_id)
        ports = (pod.get("runtime") or {}).get("ports") or []
        for p in ports:
            if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("ip"):
                ip, port = p["ip"], int(p["publicPort"])
                probe = subprocess.run(
                    ["ssh", *_ssh_opts(port), f"root@{ip}", "echo ready"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if probe.returncode == 0:
                    return ip, port
                last_err = probe.stderr
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} SSH not reachable after {timeout_s}s (last error: {last_err})")


def rsync_up(ip: str, port: int) -> None:
    subprocess.run(["ssh", *_ssh_opts(port), f"root@{ip}", f"mkdir -p {REMOTE_DIR}"], check=True)
    excludes = []
    for e in RSYNC_EXCLUDES:
        excludes += ["--exclude", e]
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            *excludes,
            "-e",
            f"ssh {' '.join(_ssh_opts(port))}",
            f"{ROOT}/",
            f"root@{ip}:{REMOTE_DIR}/",
        ],
        check=True,
    )


def run_bootstrap(ip: str, port: int, env_vars: dict[str, str]) -> int:
    env_prefix = " ".join(f"{k}={v!r}" for k, v in env_vars.items())
    remote_cmd = (
        f"cd {REMOTE_DIR} && chmod +x scripts/runpod/bootstrap.sh && "
        f"env {env_prefix} bash scripts/runpod/bootstrap.sh"
    )
    proc = subprocess.run(["ssh", *_ssh_opts(port), f"root@{ip}", remote_cmd])
    return proc.returncode


def rsync_down(ip: str, port: int) -> None:
    for sub in ("data/processed", "reports/figures"):
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh {' '.join(_ssh_opts(port))}",
                f"root@{ip}:{REMOTE_DIR}/{sub}/",
                f"{ROOT}/{sub}/",
            ],
            check=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu-type-id", default="NVIDIA RTX A4000", help="RunPod GPU type id")
    parser.add_argument("--cloud-type", default="SECURE", choices=["SECURE", "COMMUNITY", "ALL"])
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--volume-gb", type=int, default=20)
    parser.add_argument("--container-disk-gb", type=int, default=30)
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=4.0,
        help="Hard self-terminate backstop for the pod (default 4h)",
    )
    parser.add_argument("--ssh-wait-s", type=int, default=600)
    parser.add_argument("--with-gnn", action="store_true", default=True)
    parser.add_argument("--no-gnn", dest="with_gnn", action="store_false")
    parser.add_argument("--with-pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="with_pretrained", action="store_false")
    parser.add_argument("--gnn-epochs", type=int, default=60)
    parser.add_argument("--gnn-hpo-trials", type=int, default=20)
    parser.add_argument("--pretrained-model", default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--pretrained-epochs", type=int, default=3)
    parser.add_argument("--pretrained-hpo-trials", type=int, default=6)
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help=(
            "Don't stop the pod after the run; print an SSH command for interactive use "
            "(the watchdog still fires after --max-runtime-hours)."
        ),
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Terminate (delete) the pod on completion instead of just stopping it.",
    )
    parser.add_argument("--pod-name", default="abx-atlas-gpu")
    args = parser.parse_args(argv)

    import runpod

    api_key = _load_api_key()
    runpod.api_key = api_key

    env_vars = {
        "MAX_RUNTIME_SECONDS": str(int(args.max_runtime_hours * 3600)),
        "RUNPOD_API_KEY": api_key,
        "WITH_GNN": "1" if args.with_gnn else "0",
        "WITH_PRETRAINED": "1" if args.with_pretrained else "0",
        "GNN_EPOCHS": str(args.gnn_epochs),
        "GNN_HPO_TRIALS": str(args.gnn_hpo_trials),
        "PRETRAINED_MODEL": args.pretrained_model,
        "PRETRAINED_EPOCHS": str(args.pretrained_epochs),
        "PRETRAINED_HPO_TRIALS": str(args.pretrained_hpo_trials),
    }

    print(f"[launch] Creating pod ({args.gpu_type_id}, {args.cloud_type})...")
    pod = runpod.create_pod(
        name=args.pod_name,
        image_name=args.image,
        gpu_type_id=args.gpu_type_id,
        cloud_type=args.cloud_type,
        gpu_count=1,
        volume_in_gb=args.volume_gb,
        container_disk_in_gb=args.container_disk_gb,
        ports="22/tcp",
        support_public_ip=True,
        start_ssh=True,
        env={
            "RUNPOD_API_KEY": api_key,
            "MAX_RUNTIME_SECONDS": env_vars["MAX_RUNTIME_SECONDS"],
        },
    )
    pod_id = pod["id"]
    print(f"[launch] Pod created: {pod_id}")

    status = 1
    ip: str | None = None
    port: int | None = None
    try:
        print(f"[launch] Waiting for SSH (up to {args.ssh_wait_s}s)...")
        ip, port = wait_for_ssh(runpod, pod_id, args.ssh_wait_s)
        print(f"[launch] SSH ready at {ip}:{port}")

        print("[launch] Syncing repo + data up to the pod...")
        rsync_up(ip, port)

        print("[launch] Running bootstrap.sh on the pod (streaming below)...")
        # Pass pod id explicitly so the watchdog can self-terminate even if the
        # platform-injected RUNPOD_POD_ID env var is missing on some images.
        status = run_bootstrap(ip, port, {**env_vars, "RUNPOD_POD_ID": pod_id})
        print(f"[launch] Remote job exited with status {status}")

        print("[launch] Syncing results back...")
        rsync_down(ip, port)
        print("[launch] Done. Updated CSVs/figures are in data/processed/ and reports/figures/.")
    finally:
        if args.keep_alive:
            ssh_hint = f"ssh {' '.join(_ssh_opts(port))} root@{ip}" if ip else "(pod never became reachable)"
            print(
                f"[launch] --keep-alive set: leaving pod {pod_id} running.\n"
                f"  SSH:  {ssh_hint}\n"
                f"  Note: the pod-side watchdog still self-terminates after "
                f"{args.max_runtime_hours}h regardless."
            )
        elif args.terminate:
            print(f"[launch] Terminating pod {pod_id}...")
            runpod.terminate_pod(pod_id)
        else:
            print(f"[launch] Stopping pod {pod_id} (disk kept; GPU billing stops)...")
            runpod.stop_pod(pod_id)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
