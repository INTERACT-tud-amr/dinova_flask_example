#!/usr/bin/env python3
"""
Dummy HTTP velocity service for dummy_client.py.

Answers POST /velocity_action with joint velocities for the Dinova base and arm.
No ROS needed, only flask + numpy.

Run (on the machine whose IP is set as the client's server_url):
    python3 dummy_velocity_service.py                    # small sinusoidal motion
    python3 dummy_velocity_service.py --mode zero        # always zero velocity
    python3 dummy_velocity_service.py --port 4000 --arm-amp 0.1 --base-amp 0.05

Request (JSON, sent by dummy_client.py):
    {"time": float, "first_call": bool, "joint_names": [...], "q": [...], "dq": [...]}
    q = [x, y, theta, q1..q6, gripper]

Response (JSON):
    {"base_velocity": [vx, vy, wz], "arm_velocity": [dq1..dq6]}   # m/s, rad/s

Sine mode: v(t) = A * sin(2*pi*f*(t - t0)), where t0 is reset on every
first_call. Velocity starts at zero, and each joint moves at most 2*A / (2*pi*f)
away from where it started (about 0.32 rad for the arm with the defaults).
"""

import argparse
import math
import threading

import numpy as np
from flask import Flask, jsonify, request

N_BASE = 3
N_ARM = 6

app = Flask(__name__)
_lock = threading.Lock()
_t0 = None
args = None


def compute_velocities(t, q):
    """Replace this with your own controller. q = [x, y, theta, q1..q6, ...]."""
    if args.mode == "zero":
        return np.zeros(N_BASE), np.zeros(N_ARM)

    s = math.sin(2.0 * math.pi * args.freq * t)
    base = np.array([args.base_amp * s, 0.0, args.base_ang_amp * s])
    arm = np.full(N_ARM, args.arm_amp * s * 0)
    return base, arm


@app.route("/velocity_action", methods=["POST"])
def velocity_action():
    global _t0
    req = request.get_json(force=True, silent=True)
    if not req or "time" not in req:
        return jsonify({"error": "expected JSON with 'time'"}), 400

    q = np.asarray(req.get("q", []), dtype=float)
    if q.size < N_BASE + N_ARM:
        return jsonify({"error": f"expected at least {N_BASE + N_ARM} joint positions in 'q', got {q.size}"}), 400

    with _lock:
        if _t0 is None or req.get("first_call", False):
            _t0 = float(req["time"])
            app.logger.info("first call, resetting t0")
        t = float(req["time"]) - _t0

    base, arm = compute_velocities(t, q)
    return jsonify({"base_velocity": base.tolist(), "arm_velocity": arm.tolist()})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": args.mode})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=4000)
    p.add_argument("--mode", choices=["sine", "zero"], default="sine")
    p.add_argument("--freq", type=float, default=0.1, help="sine frequency [Hz]")
    p.add_argument("--arm-amp", type=float, default=0.1, help="arm joint velocity amplitude [rad/s]")
    p.add_argument("--base-amp", type=float, default=0.0, help="base forward velocity amplitude [m/s]")
    p.add_argument("--base-ang-amp", type=float, default=0.0, help="base yaw rate amplitude [rad/s]")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Serving /velocity_action on {args.host}:{args.port} (mode={args.mode})")
    app.run(host=args.host, port=args.port, threaded=True)
