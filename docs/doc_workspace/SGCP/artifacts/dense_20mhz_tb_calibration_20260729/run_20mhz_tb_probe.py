# -*- coding: utf-8 -*-
"""Calibrate 20 MHz / 10 target-subchannel NS3 manual-scheduler TB size.

The script starts the CARLA-NS3 bridge directly through the local Windows path,
runs a deterministic OpenCDA link probe, then parses MANUAL_CMD_CONSUME events.
It avoids shell background-job quoting so it can be run from PowerShell/Codex.
"""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

from opencda.core.networking.ns3_co_simulation.bridge.carla_ns3_bridge import (
    CarlaNs3Bridge,
)


REPO = pathlib.Path(r"C:\Workspace\OpenCDA")
NS3_ROOT = pathlib.Path(r"C:\Workspace\carla-ns3-co-simulation\ns-3-dev")
WSL_NS3_ROOT = "/home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev"
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/dense_20mhz_tb_calibration_20260729"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-subchannels", type=int, default=10)
    parser.add_argument("--sl-bandwidth-in-100khz", type=int, default=200)
    parser.add_argument("--sl-subchannel-size", type=int, default=5)
    parser.add_argument("--probe-case", default=None,
                        choices=["success", "edge_success", "conflict", "out_of_band"])
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def run_probe():
    args = parse_args()
    label = args.label or (
        "%dmhz_%dch_%dprb" % (
            args.sl_bandwidth_in_100khz // 10,
            args.target_subchannels,
            args.sl_subchannel_size,
        )
    )
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    ns3_stdout = ARTIFACT / ("ns3_%s_stdout.log" % label)
    ns3_stderr = ARTIFACT / ("ns3_%s_stderr.log" % label)
    probe_stdout = ARTIFACT / ("probe_%s_stdout.log" % label)
    upload_plan = ARTIFACT / ("upload_plan_%s.csv" % label)
    summary_json = ARTIFACT / ("%s_tb_summary.json" % label)
    for path in [ns3_stdout, ns3_stderr, probe_stdout, upload_plan, summary_json]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    ns3_args = (
        "scratch/vanet/main.cc "
        "--simTime=3.0 "
        "--enableTimeSync=true "
        "--carlaHost=auto "
        "--targetSubchannels=%d "
        "--slBandwidthIn100kHz=%d "
        "--slSubchannelSize=%d "
        "--slMcs=28 "
        "--slSymbolsPerSlot=12 "
        "--slPscchRbs=10 "
        "--slRriMs=5 "
        "--slBearerActivationGuardMs=1 "
        "--nrSlZeroTimeSendDelayMs=0"
    ) % (
        args.target_subchannels,
        args.sl_bandwidth_in_100khz,
        args.sl_subchannel_size,
    )
    # The current ns-3 CMake cache was generated under WSL/Linux paths.
    # Launch through WSL to avoid CMake source-dir mismatch while keeping the
    # OpenCDA probe on Windows. Windows localhost can connect to the WSL bridge.
    ns3_cmd = [
        "wsl", "bash", "-lc",
        "cd %s && ./ns3 run '%s'" % (WSL_NS3_ROOT, ns3_args),
    ]
    with ns3_stdout.open("wb") as out, ns3_stderr.open("wb") as err:
        ns3_proc = subprocess.Popen(
            ns3_cmd,
            cwd=str(REPO),
            stdout=out,
            stderr=err,
        )

    probe_return = None
    try:
        time.sleep(5)
        probe_case = args.probe_case
        if probe_case is None:
            probe_case = "edge_success" if args.target_subchannels >= 10 else "success"
        with probe_stdout.open("w", encoding="utf-8") as out:
            probe_return = run_direct_bridge_probe(
                upload_plan, probe_case, args.target_subchannels, out)
        try:
            ns3_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            ns3_proc.terminate()
            try:
                ns3_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ns3_proc.kill()
                ns3_proc.wait(timeout=5)
    finally:
        if ns3_proc.poll() is None:
            ns3_proc.kill()
            ns3_proc.wait(timeout=5)

    text = ns3_stdout.read_text(encoding="utf-8", errors="replace")
    allocated = [int(match.group(1))
                 for match in re.finditer(r"allocated=(\d+)", text)]
    callbacks = len(re.findall(r"\[CAM_RX_CALLBACK\]", text))
    manual_add = len(re.findall(r"\[MANUAL_CMD_ADD\]", text))
    manual_reject = len(re.findall(r"\[MANUAL_CMD_REJECT\]", text))
    manual_consume = len(re.findall(r"\[MANUAL_CMD_CONSUME\]", text))
    summary = {
        "ns3_command": " ".join(ns3_cmd),
        "windows_ns3_root": str(NS3_ROOT),
        "wsl_ns3_root": WSL_NS3_ROOT,
        "probe_returncode": probe_return,
        "ns3_returncode": ns3_proc.returncode,
        "manual_add": manual_add,
        "manual_reject": manual_reject,
        "manual_consume": manual_consume,
        "callbacks": callbacks,
        "allocated_events": len(allocated),
        "allocated_min_bytes": min(allocated) if allocated else None,
        "allocated_mean_bytes": round(statistics.mean(allocated), 2) if allocated else None,
        "allocated_median_bytes": round(statistics.median(allocated), 2) if allocated else None,
        "allocated_max_bytes": max(allocated) if allocated else None,
        "recommended_tb_size_bytes": int(round(statistics.median(allocated))) if allocated else None,
        "target_subchannels": args.target_subchannels,
        "sl_bandwidth_in_100khz": args.sl_bandwidth_in_100khz,
        "sl_subchannel_size_prb": args.sl_subchannel_size,
        "notes": (
            "Recommended estimator TB size is the median "
            "observed MANUAL_CMD_CONSUME allocated bytes for one subchannel "
            "grant under MCS28 and 12 PSSCH symbols."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_vehicles():
    vehicles = []
    for index, carla_id in enumerate([1, 2, 3, 4]):
        vehicles.append({
            "id": index,
            "carla_id": carla_id,
            "position": {"x": float(index * 5), "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "heading": 0.0,
            "speed": 0.0,
        })
    return vehicles


def build_requests(case_name, target_subchannels):
    if case_name == "edge_success" and target_subchannels >= 10:
        return [{
            "source": 1,
            "target": 2,
            "size": 20000,
            "pkt_id": 1,
            "sc_start": 9,
            "sc_num": 1,
            "case": case_name,
        }]
    return [
        {
            "source": 1,
            "target": 2,
            "size": 20000,
            "pkt_id": 1,
            "sc_start": 0,
            "sc_num": 1,
            "case": case_name,
        },
        {
            "source": 3,
            "target": 4,
            "size": 20000,
            "pkt_id": 2,
            "sc_start": min(1, max(0, target_subchannels - 1)),
            "sc_num": 1,
            "case": case_name,
        },
    ]


def write_upload_plan(path, requests):
    import csv
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "timestamp", "area_id", "source_id", "target_id", "bytes",
            "upload_type", "pkt_id",
        ])
        writer.writeheader()
        for request in requests:
            writer.writerow({
                "timestamp": "probe000",
                "area_id": "",
                "source_id": request["source"],
                "target_id": request["target"],
                "bytes": request["size"],
                "upload_type": "tb_probe_" + request.get("case", ""),
                "pkt_id": request["pkt_id"],
            })


def run_direct_bridge_probe(upload_plan, case_name, target_subchannels, out):
    vehicles = build_vehicles()
    requests = build_requests(case_name, target_subchannels)
    write_upload_plan(upload_plan, requests)
    out.write("case=%s vehicles=%d requests=%d bytes=%d\n" % (
        case_name, len(vehicles), len(requests),
        sum(request["size"] for request in requests)))
    for request in requests:
        out.write("request pkt_id={pkt_id} {source}->{target} "
                  "sc={sc_start}:{sc_num} bytes={size}\n".format(**request))
    out.flush()

    bridge = CarlaNs3Bridge()
    bridge.sync_timeout = 20.0
    bridge.enable_time_sync(True)
    bridge.start()
    try:
        bridge.send_vehicles_num(len(vehicles))
        bridge.send_vehicles_position(vehicles)
        if not bridge.sync_with_ns3(0.0):
            raise RuntimeError("initial NS3 sync failed")
        # The NS3 script activates sidelink bearers after
        # 1 ms + slBearerActivationGuardMs.  The probe uses zero application
        # send delay, so inject transfer requests only after activation.
        if not bridge.sync_with_ns3(0.005):
            raise RuntimeError("pre-send activation sync failed")
        bridge.send_transfer_requests(requests)
        time.sleep(0.2)
        for target_time in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5]:
            if not bridge.sync_with_ns3(target_time):
                raise RuntimeError("NS3 sync failed at %.3fs" % target_time)
            time.sleep(0.03)
        return 0
    except Exception as exc:
        out.write("probe_error=%s\n" % exc)
        out.flush()
        return 1
    finally:
        bridge.stop()


if __name__ == "__main__":
    run_probe()
