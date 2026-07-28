# -*- coding: utf-8 -*-
"""Run guard/zero-time-delay control-plane NS3 probes.

This script starts the WSL NS3 bridge for each case, sends synthetic SGCP
control-plane requests from the local OpenCDA environment, and parses the NS3
log for application callback counts and timestamps.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = ROOT / "docs" / "doc_workspace" / "SGCP" / "artifacts" / "control_plane_ns3_20260728" / "guard_zero_sweep_20260729"
PROBE = ROOT / "docs" / "doc_workspace" / "SGCP" / "artifacts" / "control_plane_ns3_20260728" / "control_plane_probe.py"
NS3_DIR = "/home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", choices=["all", "smoke"], default="all")
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser.parse_args()


def run(cmd, cwd=None, timeout=None, stdout=None, stderr=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        stdout=stdout if stdout is not None else subprocess.PIPE,
        stderr=stderr if stderr is not None else subprocess.PIPE,
        text=True,
        check=False,
    )


def start_ns3(case, out_log, err_log):
    cast_flag = " --enableControlBroadcast=true" if case["cast"] == "broadcast" else ""
    ns3_cmd = (
        f"cd {NS3_DIR} && timeout 90s stdbuf -oL -eL ./ns3 run "
        f"'scratch/vanet/main.cc --simTime=3.0 --enableTimeSync=true "
        f"--carlaHost=auto --targetSubchannels=10 --slMcs=28 "
        f"--slSymbolsPerSlot=12 --slPscchRbs=10 --slRriMs=5 "
        f"--slBearerActivationGuardMs={case['guard_ms']} "
        f"--nrSlZeroTimeSendDelayMs={case['zero_delay_ms']}"
        f"{cast_flag}'"
    )
    out_fh = open(out_log, "w", encoding="utf-8", errors="replace")
    err_fh = open(err_log, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        ["wsl.exe", "-d", "Ubuntu-22.04", "-u", "sakakibara", "--", "bash", "-lc", ns3_cmd],
        stdout=out_fh,
        stderr=err_fh,
        text=True,
    )
    return proc, out_fh, err_fh


def wait_for_port(timeout_s=20.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = run(
            [
                "wsl.exe",
                "-d",
                "Ubuntu-22.04",
                "-u",
                "sakakibara",
                "--",
                "bash",
                "-lc",
                "ss -ltnp | grep -E ':5556' || true",
            ],
            timeout=5,
        )
        if ":5556" in result.stdout:
            return True
        time.sleep(0.5)
    return False


def kill_ns3():
    run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "-u",
            "sakakibara",
            "--",
            "bash",
            "-lc",
            "pkill -f 'scratch/vanet/main.cc --simTime=3.0' || true",
        ],
        timeout=10,
    )


def run_probe(case, plan_path):
    pre_sync_ms = 0.0
    if case["timing"] == "after_activation":
        pre_sync_ms = 1.0 + float(case["guard_ms"])
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "opencda",
        "python",
        str(PROBE),
        "--vehicles",
        "20",
        "--packet-size",
        "400",
        "--subchannels",
        "10",
        "--profile",
        "aggregated",
        "--cast-type",
        case["cast"],
        "--batch-size",
        "10",
        "--batch-step-ms",
        "2",
        "--pre-send-sync-ms",
        str(pre_sync_ms),
        "--drain-seconds",
        "1.0",
        "--sync-timeout",
        "20",
        "--upload-plan-output",
        str(plan_path),
    ]
    if case["cast"] == "unicast":
        cmd.append("--endpoint-disjoint")
    return run(cmd, cwd=str(ROOT), timeout=60)


def parse_log(path, cast):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    callbacks = []
    for match in re.finditer(r'SendMsgToCarla: (\{"type":"cam_received".*?\}), send_to_carla_fd', text):
        callbacks.append(json.loads(match.group(1)))
    unique_requests = len({row["request_id"] for row in callbacks})
    scheduled = text.count("CamSenderNR: scheduled CAM")
    rlc_rx = text.count("RLC_RX")
    manual = text.count("MANUAL_RESOURCE_APPLY")
    waits = text.count("MANUAL_CMD_WAIT")
    rejects = text.count("MANUAL_CMD_REJECT")
    request_count = scheduled
    expected = request_count if cast == "unicast" else request_count * 10
    recv_times = [row["receive_timestamp"] for row in callbacks]
    send_times = [row["send_timestamp"] for row in callbacks]
    delays = [row["receive_timestamp"] - row["send_timestamp"] for row in callbacks]
    return {
        "scheduled_requests": scheduled,
        "expected_callbacks": expected,
        "callbacks": len(callbacks),
        "unique_requests": unique_requests,
        "rlc_rx": rlc_rx,
        "manual_resources": manual,
        "manual_waits": waits,
        "manual_rejects": rejects,
        "send_min_ms": min(send_times) if send_times else "",
        "send_max_ms": max(send_times) if send_times else "",
        "recv_min_ms": min(recv_times) if recv_times else "",
        "recv_max_ms": max(recv_times) if recv_times else "",
        "delay_mean_ms": sum(delays) / len(delays) if delays else "",
        "delay_max_ms": max(delays) if delays else "",
    }


def build_cases(kind):
    cases = []
    casts = ["unicast", "broadcast"]
    timings = ["at_zero", "after_activation"]
    guards = [1, 0]
    zeros = [1, 0]
    for cast in casts:
        for timing in timings:
            for guard in guards:
                for zero_delay in zeros:
                    cases.append({
                        "cast": cast,
                        "timing": timing,
                        "guard_ms": guard,
                        "zero_delay_ms": zero_delay,
                    })
    if kind == "smoke":
        return cases[:2]
    return cases


def main():
    args = parse_args()
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.cases)
    rows = []
    for index, case in enumerate(cases, start=1):
        case_name = (
            f"{index:02d}_{case['cast']}_{case['timing']}_"
            f"guard{case['guard_ms']}_zero{case['zero_delay_ms']}"
        )
        out_log = ARTIFACT / f"{case_name}.ns3.out.log"
        err_log = ARTIFACT / f"{case_name}.ns3.err.log"
        plan_path = ARTIFACT / f"{case_name}.plan.csv"
        print(f"[CASE] {case_name}", flush=True)
        kill_ns3()
        proc, out_fh, err_fh = start_ns3(case, out_log, err_log)
        try:
            if not wait_for_port():
                raise RuntimeError("NS3 did not open port 5556")
            probe_result = run_probe(case, plan_path)
            time.sleep(1.0)
        finally:
            kill_ns3()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            out_fh.close()
            err_fh.close()
        parsed = parse_log(out_log, case["cast"])
        row = {
            "case": case_name,
            **case,
            **parsed,
            "probe_returncode": probe_result.returncode,
            "probe_stdout": probe_result.stdout.strip().replace("\n", " | "),
            "probe_stderr": probe_result.stderr.strip().replace("\n", " | "),
            "ns3_out_log": str(out_log),
            "ns3_err_log": str(err_log),
            "plan_csv": str(plan_path),
        }
        rows.append(row)
        print(
            f"  callbacks={row['callbacks']}/{row['expected_callbacks']} "
            f"unique={row['unique_requests']} recv_max={row['recv_max_ms']}",
            flush=True,
        )
    result_csv = ARTIFACT / "guard_zero_sweep_results.csv"
    with open(result_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] {result_csv}")


if __name__ == "__main__":
    sys.exit(main())
