# -*- coding: utf-8 -*-
"""Replay the dense SGCP 93.75 Mbps frame through NS3.

This helper launches the WSL ns-3 VANET bridge, sends one precomputed
OpenCDA upload plan to it, and evaluates application/RLC/PHY delivery.
It is intentionally kept under the artifact directory so the experiment is
reproducible without touching the production OpenCDA algorithms.
"""

import argparse
import csv
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.networking.ns3_co_simulation.bridge.carla_ns3_bridge import (
    CarlaNs3Bridge,
)
from opencda.tools.offline_ns3_replay import pose_to_vehicle_state


REPO = pathlib.Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/dense_120mbps_rho_20260729"
WSL_NS3_ROOT = "/home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev"
DEFAULT_DATASET_ROOT = pathlib.Path(r"D:\Data\Carla")
DEFAULT_SCENARIO_ID = "2026_07_29_02_32_08"
DEFAULT_TIMESTAMP = "000060"
DEFAULT_PLAN = ARTIFACT / "ns3_frame000060_rho5_deadline200/upload_plan.csv"

CAM_PATTERN = re.compile(r"SendMsgToCarla: (\{.*?\}), send_to_carla_fd")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    parser.add_argument("--upload-plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--output-dir", default=str(
        ARTIFACT / "ns3_frame000060_rho5_deadline200" / "guard1_zero0_40mhz10ch"))
    parser.add_argument("--guard-ms", type=float, default=1.0)
    parser.add_argument("--zero-delay-ms", type=float, default=0.0)
    parser.add_argument("--sim-time", type=float, default=8.0)
    parser.add_argument("--target-subchannels", type=int, default=10)
    parser.add_argument("--sl-bandwidth-in-100khz", type=int, default=400)
    parser.add_argument("--sl-subchannel-size", type=int, default=10)
    parser.add_argument("--sl-mcs", type=int, default=28)
    parser.add_argument("--sl-symbols-per-slot", type=int, default=12)
    parser.add_argument("--sl-pscch-rbs", type=int, default=10)
    parser.add_argument("--sl-rri-ms", type=int, default=5)
    parser.add_argument("--sync-timeout", type=float, default=30.0)
    parser.add_argument("--pre-send-sync-seconds", type=float, default=0.005)
    return parser.parse_args()


def load_vehicles(dataset_root, scenario_id, timestamp):
    dataset = OPV2VFrameDataset(dataset_root)
    frame = dataset.load_frame(scenario_id, timestamp, add_transformation=False)
    vehicles = []
    for index, vehicle_id in enumerate(sorted(frame.keys(), key=lambda value: int(value))):
        vehicles.append(pose_to_vehicle_state(index, vehicle_id, frame[vehicle_id]))
    return vehicles


def read_requests(upload_plan, timestamp):
    requests = []
    total_bytes = 0
    links = set()
    with open(upload_plan, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("timestamp") != timestamp:
                continue
            request = {
                "source": int(float(row["source_id"])),
                "target": int(float(row["target_id"])),
                "size": int(float(row["bytes"])),
                "pkt_id": int(float(row.get("pkt_id") or len(requests) + 1)),
                "sc_start": int(float(row.get("sc_start") or 0)),
                "sc_num": int(float(row.get("sc_num") or 1)),
            }
            if row.get("upload_type"):
                request["upload_type"] = row["upload_type"]
            requests.append(request)
            total_bytes += request["size"]
            links.add((request["source"], request["target"]))
    if not requests:
        raise ValueError("No requests found for timestamp %s in %s" % (timestamp, upload_plan))
    return requests, total_bytes, links


def ns3_command(args):
    ns3_args = (
        "scratch/vanet/main.cc "
        "--simTime={sim_time} "
        "--enableTimeSync=true "
        "--carlaHost=auto "
        "--targetSubchannels={target_subchannels} "
        "--slBandwidthIn100kHz={bandwidth} "
        "--slSubchannelSize={subchannel_size} "
        "--slMcs={mcs} "
        "--slSymbolsPerSlot={symbols} "
        "--slPscchRbs={pscch_rbs} "
        "--slRriMs={rri_ms} "
        "--slBearerActivationGuardMs={guard_ms:g} "
        "--nrSlZeroTimeSendDelayMs={zero_delay_ms:g}"
    ).format(
        sim_time=args.sim_time,
        target_subchannels=args.target_subchannels,
        bandwidth=args.sl_bandwidth_in_100khz,
        subchannel_size=args.sl_subchannel_size,
        mcs=args.sl_mcs,
        symbols=args.sl_symbols_per_slot,
        pscch_rbs=args.sl_pscch_rbs,
        rri_ms=args.sl_rri_ms,
        guard_ms=args.guard_ms,
        zero_delay_ms=args.zero_delay_ms,
    )
    return [
        "wsl", "bash", "-lc",
        "cd %s && ./ns3 run '%s'" % (WSL_NS3_ROOT, ns3_args),
    ]


def parse_cam_records(path):
    records = []
    with open(path, encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            match = CAM_PATTERN.search(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "cam_received":
                continue
            send_ms = float(payload.get("send_timestamp", 0.0))
            receive_ms = float(payload.get("receive_timestamp", 0.0))
            records.append({
                "request_id": int(float(payload.get("request_id", 0))),
                "source": int(float(payload.get("sender_id", 0))),
                "target": int(float(payload.get("receiver_id", 0))),
                "bytes": int(float(payload.get("packet_size", 0))),
                "send_ms": send_ms,
                "receive_ms": receive_ms,
                "delay_ms": receive_ms - send_ms,
            })
    return records


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    index = int(math.ceil((pct / 100.0) * len(values))) - 1
    return values[max(0, min(index, len(values) - 1))]


def run_bridge(args, stdout_path, probe_stdout_path):
    vehicles = load_vehicles(args.dataset_root, args.scenario_id, args.timestamp)
    requests, total_bytes, links = read_requests(args.upload_plan, args.timestamp)
    with open(probe_stdout_path, "w", encoding="utf-8") as out:
        out.write("vehicles=%d requests=%d links=%d bytes=%d\n" % (
            len(vehicles), len(requests), len(links), total_bytes))
        out.write("guard_ms=%s zero_delay_ms=%s bandwidth_100khz=%s target_subchannels=%s\n" % (
            args.guard_ms, args.zero_delay_ms,
            args.sl_bandwidth_in_100khz, args.target_subchannels))
        out.flush()

        bridge = CarlaNs3Bridge()
        bridge.sync_timeout = args.sync_timeout
        bridge.enable_time_sync(True)
        bridge.start()
        try:
            bridge.send_vehicles_num(len(vehicles))
            bridge.send_vehicles_position(vehicles)
            if not bridge.sync_with_ns3(0.0):
                raise RuntimeError("initial NS3 sync failed")
            if args.pre_send_sync_seconds > 0:
                if not bridge.sync_with_ns3(args.pre_send_sync_seconds):
                    raise RuntimeError("pre-send activation sync failed")
            bridge.send_transfer_requests(requests)
            # Advance NS3 time densely around the expected deadline and then
            # drain long enough to observe delayed RLC/application callbacks.
            sync_times = [
                0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20,
                0.30, 0.50, 1.00, 2.50, min(args.sim_time - 0.1, 6.0),
            ]
            seen = set()
            for target_time in sync_times:
                if target_time <= args.pre_send_sync_seconds or target_time in seen:
                    continue
                seen.add(target_time)
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


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ns3_stdout = output_dir / "ns3_stdout.log"
    ns3_stderr = output_dir / "ns3_stderr.log"
    probe_stdout = output_dir / "probe_stdout.log"
    summary_json = output_dir / "summary.json"

    for path in [ns3_stdout, ns3_stderr, probe_stdout, summary_json]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    cmd = ns3_command(args)
    with ns3_stdout.open("wb") as out, ns3_stderr.open("wb") as err:
        ns3_proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=err)

    probe_return = None
    try:
        time.sleep(5)
        probe_return = run_bridge(args, ns3_stdout, probe_stdout)
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

    requests, total_bytes, links = read_requests(args.upload_plan, args.timestamp)
    cam_records = parse_cam_records(ns3_stdout)
    delays = [item["delay_ms"] for item in cam_records]
    requested_ids = {request["pkt_id"] for request in requests}
    callback_ids = {record["request_id"] for record in cam_records}
    stdout_text = ns3_stdout.read_text(encoding="utf-8", errors="replace")
    summary = {
        "ns3_command": " ".join(cmd),
        "scenario_id": args.scenario_id,
        "timestamp": args.timestamp,
        "guard_ms": args.guard_ms,
        "zero_delay_ms": args.zero_delay_ms,
        "sl_bandwidth_in_100khz": args.sl_bandwidth_in_100khz,
        "target_subchannels": args.target_subchannels,
        "planned_requests": len(requests),
        "planned_links": len(links),
        "planned_bytes": total_bytes,
        "application_callbacks": len(cam_records),
        "unique_callback_requests": len(callback_ids),
        "missing_callback_requests": len(requested_ids - callback_ids),
        "callback_delivery_ratio": (
            len(callback_ids) / len(requested_ids) if requested_ids else 0.0
        ),
        "delay_mean_ms": round(statistics.mean(delays), 3) if delays else 0.0,
        "delay_p95_ms": round(percentile(delays, 95), 3),
        "delay_max_ms": round(max(delays), 3) if delays else 0.0,
        "manual_add": len(re.findall(r"\[MANUAL_CMD_ADD\]", stdout_text)),
        "manual_reject": len(re.findall(r"\[MANUAL_CMD_REJECT\]", stdout_text)),
        "manual_consume": len(re.findall(r"\[MANUAL_CMD_CONSUME\]", stdout_text)),
        "rlc_tx": len(re.findall(r"\[NRSL_RLC_TX\]", stdout_text)),
        "rlc_rx": len(re.findall(r"\[NRSL_RLC_RX\]", stdout_text)),
        "rlc_drop": len(re.findall(r"\[NRSL_RLC_DROP\]", stdout_text)),
        "pssch_fail": len(re.findall(r"\[PSSCH_DECODE_FAIL\]", stdout_text)),
        "probe_returncode": probe_return,
        "ns3_returncode": ns3_proc.returncode,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
