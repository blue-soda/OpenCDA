"""Windowed 41-frame NS3 replay for the SGCP hybrid main-table schedule.

The generic offline replay sends transfers after the frame-start sync and then
does not explicitly check the 60 ms data-plane window. This runner replays the
fixed SGCP main-table upload plan and advances NS3 to frame_start + 60 ms after
each frame, which directly tests whether the scheduled raw-LiDAR payload fits
the reserved communication window.
"""

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
import subprocess
import time
from collections import defaultdict

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.networking.ns3_co_simulation.bridge.carla_ns3_bridge import (
    CarlaNs3Bridge,
)
from opencda.tools.offline_ns3_replay import pose_to_vehicle_state


REPO = pathlib.Path(r"C:\Workspace\OpenCDA")
HERE = pathlib.Path(__file__).resolve().parent
WSL_NS3_ROOT = "/home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev"
CAM_PATTERN = re.compile(r"SendMsgToCarla: (\{.*?\}), send_to_carla_fd")
RLC_PATTERN = re.compile(
    r"\[NRSL_RLC_(TX|RX|DROP)\].*?timeMs=([0-9.eE+-]+).*?"
    r"request_id=([0-9]+).*?size=([0-9]+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=r"D:\Data\Carla")
    parser.add_argument("--scenario-id", default="2026_07_29_02_32_08")
    parser.add_argument("--upload-plan", default=str(HERE / "sgcp_hybrid_ns3_upload_plan.csv"))
    parser.add_argument("--output-dir", default=str(HERE / "ns3_replay_41f_windowed_guard1_zero0"))
    parser.add_argument("--frame-interval-ms", type=float, default=100.0)
    parser.add_argument("--start-offset-ms", type=float, default=0.0,
                        help="NS3 warm-up time before the first replayed frame.")
    parser.add_argument("--data-window-ms", type=float, default=60.0)
    parser.add_argument("--receiver-stagger-ms", type=float, default=0.0,
                        help="If positive, requests in the same frame are "
                             "sent in micro-batches so that a receiver gets "
                             "at most one request per batch. The next batch "
                             "is delayed by this many milliseconds.")
    parser.add_argument("--max-frames", type=int, default=41)
    parser.add_argument("--sim-time", type=float, default=8.0)
    parser.add_argument("--guard-ms", type=float, default=1.0)
    parser.add_argument("--zero-delay-ms", type=float, default=0.0)
    parser.add_argument("--sl-error-model-enabled", action="store_true",
                        help="Enable stochastic NR sidelink PSCCH/PSSCH error "
                             "models. The default deterministic replay checks "
                             "scheduled capacity and latency without random "
                             "decode drops.")
    parser.add_argument("--sync-timeout", type=float, default=30.0)
    return parser.parse_args()


def ns3_command(args):
    ns3_args = (
        "scratch/vanet/main.cc "
        "--simTime={sim_time} "
        "--enableTimeSync=true "
        "--carlaHost=auto "
        "--targetSubchannels=10 "
        "--slBandwidthIn100kHz=400 "
        "--slSubchannelSize=10 "
        "--slMcs=28 "
        "--slSymbolsPerSlot=12 "
        "--slPscchRbs=10 "
        "--slRriMs=5 "
        "--slBearerActivationGuardMs={guard:g} "
        "--nrSlZeroTimeSendDelayMs={zero:g} "
        "--slErrorModelEnabled={error_model}"
    ).format(
        sim_time=args.sim_time,
        guard=args.guard_ms,
        zero=args.zero_delay_ms,
        error_model=str(bool(args.sl_error_model_enabled)).lower(),
    )
    return [
        "wsl",
        "bash",
        "-lc",
        "cd {root} && ./ns3 run '{ns3_args}'".format(
            root=WSL_NS3_ROOT, ns3_args=ns3_args),
    ]


def read_upload_plan(path, max_frames):
    by_timestamp = defaultdict(list)
    timestamps = []
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamp = row["timestamp"]
            if timestamp not in by_timestamp:
                timestamps.append(timestamp)
            request = {
                "source": int(float(row["source_id"])),
                "target": int(float(row["target_id"])),
                "size": int(float(row["bytes"])),
                "pkt_id": int(float(row["pkt_id"])),
                "sc_start": int(float(row.get("sc_start") or 0)),
                "sc_num": int(float(row.get("sc_num") or 1)),
                "upload_type": row.get("upload_type", "sgcp_hybrid_table1_trace"),
            }
            by_timestamp[timestamp].append(request)
    timestamps = timestamps[:max_frames]
    return timestamps, by_timestamp


def load_vehicle_data(dataset, scenario_id, timestamp):
    frame = dataset.load_frame(scenario_id, timestamp, add_transformation=False)
    vehicles = []
    for index, vehicle_id in enumerate(sorted(frame.keys(), key=lambda value: int(value))):
        vehicles.append(pose_to_vehicle_state(index, vehicle_id, frame[vehicle_id]))
    return vehicles


def group_requests_by_receiver_slot(requests):
    slots = []
    for request in requests:
        placed = False
        for slot in slots:
            if all(item["target"] != request["target"] for item in slot):
                slot.append(request)
                placed = True
                break
        if not placed:
            slots.append([request])
    return slots


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


def parse_rlc_records(path):
    records = []
    with open(path, encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            match = RLC_PATTERN.search(line)
            if not match:
                continue
            event, time_ms, request_id, size = match.groups()
            records.append({
                "event": event,
                "time_ms": float(time_ms),
                "request_id": int(request_id),
                "size": int(size),
            })
    return records


def summarize_rlc_delivery(rlc_records, timestamps, requests_by_ts):
    planned = {}
    for timestamp in timestamps:
        for request in requests_by_ts[timestamp]:
            planned[request["pkt_id"]] = {
                "timestamp": timestamp,
                "payload_bytes": request["size"],
            }
    tx_start = {}
    rx_bytes = defaultdict(int)
    rx_complete_time = {}
    drop_ids = set()
    for record in sorted(rlc_records, key=lambda item: item["time_ms"]):
        request_id = record["request_id"]
        if request_id not in planned:
            continue
        if record["event"] == "TX":
            tx_start.setdefault(request_id, record["time_ms"])
        elif record["event"] == "RX":
            rx_bytes[request_id] += record["size"]
            if (request_id not in rx_complete_time and
                    rx_bytes[request_id] >= planned[request_id]["payload_bytes"]):
                rx_complete_time[request_id] = record["time_ms"]
        elif record["event"] == "DROP":
            drop_ids.add(request_id)

    complete_ids = sorted(
        request_id for request_id in planned
        if request_id in tx_start and request_id in rx_complete_time)
    delays = [
        rx_complete_time[request_id] - tx_start[request_id]
        for request_id in complete_ids
    ]
    any_rx_ids = {request_id for request_id, size in rx_bytes.items() if size > 0}
    incomplete_ids = sorted(set(planned) - set(complete_ids))
    return {
        "planned_requests": len(planned),
        "rlc_any_rx_requests": len(any_rx_ids),
        "rlc_complete_requests": len(complete_ids),
        "rlc_incomplete_requests": len(incomplete_ids),
        "rlc_complete_delivery_ratio": (
            len(complete_ids) / len(planned) if planned else 0.0),
        "rlc_delay_mean_ms": round(statistics.mean(delays), 2) if delays else 0.0,
        "rlc_delay_p95_ms": round(percentile(delays, 95), 2),
        "rlc_delay_max_ms": round(max(delays), 2) if delays else 0.0,
        "rlc_drop_requests": len(drop_ids),
        "rlc_incomplete_request_ids": incomplete_ids[:50],
    }


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    index = int(math.ceil((pct / 100.0) * len(values))) - 1
    return values[max(0, min(index, len(values) - 1))]


def write_plan_copy(path, timestamps, requests_by_ts):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "timestamp", "area_id", "source_id", "target_id", "bytes",
            "upload_type", "pkt_id", "sc_start", "sc_num",
        ])
        writer.writeheader()
        for timestamp in timestamps:
            for request in requests_by_ts[timestamp]:
                writer.writerow({
                    "timestamp": timestamp,
                    "area_id": "",
                    "source_id": request["source"],
                    "target_id": request["target"],
                    "bytes": request["size"],
                    "upload_type": request.get("upload_type", ""),
                    "pkt_id": request["pkt_id"],
                    "sc_start": request["sc_start"],
                    "sc_num": request["sc_num"],
                })


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ns3_stdout = output_dir / "ns3_stdout.log"
    ns3_stderr = output_dir / "ns3_stderr.log"
    replay_stdout = output_dir / "windowed_replay_stdout.log"
    upload_plan_copy = output_dir / "upload_plan_windowed.csv"
    summary_path = output_dir / "summary.json"
    for path in [ns3_stdout, ns3_stderr, replay_stdout, upload_plan_copy, summary_path]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    timestamps, requests_by_ts = read_upload_plan(args.upload_plan, args.max_frames)
    write_plan_copy(upload_plan_copy, timestamps, requests_by_ts)
    dataset = OPV2VFrameDataset(args.dataset_root)

    with ns3_stdout.open("wb") as out, ns3_stderr.open("wb") as err:
        ns3_proc = subprocess.Popen(
            ns3_command(args),
            cwd=str(REPO),
            stdout=out,
            stderr=err)

    replay_error = None
    try:
        time.sleep(5.0)
        bridge = CarlaNs3Bridge()
        bridge.sync_timeout = args.sync_timeout
        bridge.enable_time_sync(True)
        bridge.start()
        try:
            first_vehicles = load_vehicle_data(dataset, args.scenario_id, timestamps[0])
            bridge.send_vehicles_num(len(first_vehicles))
            with replay_stdout.open("w", encoding="utf-8") as replay_log:
                for frame_index, timestamp in enumerate(timestamps):
                    frame_start_s = (
                        args.start_offset_ms + frame_index * args.frame_interval_ms) / 1000.0
                    frame_window_s = frame_start_s + args.data_window_ms / 1000.0
                    frame_end_s = frame_start_s + args.frame_interval_ms / 1000.0
                    vehicles = load_vehicle_data(dataset, args.scenario_id, timestamp)
                    requests = requests_by_ts[timestamp]
                    bridge.send_vehicles_position(vehicles)
                    if not bridge.sync_with_ns3(frame_start_s):
                        raise RuntimeError("sync frame start failed %s" % timestamp)
                    # Ensure the first frame is sent after bearer activation.
                    if frame_index == 0:
                        activation_s = max(frame_start_s + 0.005, 0.005)
                        if not bridge.sync_with_ns3(activation_s):
                            raise RuntimeError("activation sync failed")
                    if args.receiver_stagger_ms > 0:
                        request_slots = group_requests_by_receiver_slot(requests)
                        for slot_index, slot_requests in enumerate(request_slots):
                            slot_time_s = frame_start_s + (
                                slot_index * args.receiver_stagger_ms / 1000.0)
                            if slot_time_s > frame_start_s:
                                if not bridge.sync_with_ns3(slot_time_s):
                                    raise RuntimeError(
                                        "sync stagger slot failed %s slot %d" %
                                        (timestamp, slot_index))
                            bridge.send_transfer_requests(slot_requests)
                    else:
                        request_slots = [requests]
                        bridge.send_transfer_requests(requests)
                    if not bridge.sync_with_ns3(frame_window_s):
                        raise RuntimeError("sync data window failed %s" % timestamp)
                    if frame_end_s > frame_window_s:
                        if not bridge.sync_with_ns3(frame_end_s):
                            raise RuntimeError("sync frame end failed %s" % timestamp)
                    replay_log.write(
                        "frame=%d timestamp=%s requests=%d slots=%d bytes=%d start=%.3f window=%.3f end=%.3f\n" %
                        (frame_index + 1, timestamp, len(requests),
                         len(request_slots),
                         sum(item["size"] for item in requests),
                         frame_start_s, frame_window_s, frame_end_s))
                    replay_log.flush()
                final_time = (
                    args.start_offset_ms + len(timestamps) * args.frame_interval_ms) / 1000.0 + 1.0
                bridge.sync_with_ns3(final_time)
        finally:
            bridge.stop()
        try:
            ns3_proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            ns3_proc.terminate()
            try:
                ns3_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                ns3_proc.kill()
                ns3_proc.wait(timeout=10)
    except Exception as exc:
        replay_error = str(exc)
    finally:
        if ns3_proc.poll() is None:
            ns3_proc.terminate()
            try:
                ns3_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ns3_proc.kill()
                ns3_proc.wait(timeout=5)

    cam_records = parse_cam_records(ns3_stdout)
    rlc_records = parse_rlc_records(ns3_stdout)
    rlc_summary = summarize_rlc_delivery(rlc_records, timestamps, requests_by_ts)
    delays = [record["delay_ms"] for record in cam_records]
    requested_ids = {
        request["pkt_id"]
        for timestamp in timestamps
        for request in requests_by_ts[timestamp]
    }
    callback_ids = {record["request_id"] for record in cam_records}
    stdout_text = ns3_stdout.read_text(encoding="utf-8", errors="ignore")
    summary = {
        "output_dir": str(output_dir),
        "frames": len(timestamps),
        "start_offset_ms": args.start_offset_ms,
        "sl_error_model_enabled": args.sl_error_model_enabled,
        "planned_requests": len(requested_ids),
        "planned_bytes": sum(
            request["size"]
            for timestamp in timestamps
            for request in requests_by_ts[timestamp]),
        "application_callbacks": len(cam_records),
        "unique_callback_requests": len(callback_ids),
        "missing_callback_requests": len(requested_ids - callback_ids),
        "callback_delivery_ratio": (
            len(callback_ids) / len(requested_ids) if requested_ids else 0.0),
        "delay_mean_ms": round(statistics.mean(delays), 2) if delays else 0.0,
        "delay_p95_ms": round(percentile(delays, 95), 2),
        "delay_max_ms": round(max(delays), 2) if delays else 0.0,
        "manual_add": len(re.findall(r"\[MANUAL_CMD_ADD\]", stdout_text)),
        "manual_reject": len(re.findall(r"\[MANUAL_CMD_REJECT\]", stdout_text)),
        "manual_consume": len(re.findall(r"\[MANUAL_CMD_CONSUME\]", stdout_text)),
        "rlc_tx": len(re.findall(r"\[NRSL_RLC_TX\]", stdout_text)),
        "rlc_rx": len(re.findall(r"\[NRSL_RLC_RX\]", stdout_text)),
        "rlc_drop": len(re.findall(r"\[NRSL_RLC_DROP\]", stdout_text)),
        "pssch_fail": len(re.findall(r"\[PSSCH_DECODE_FAIL\]", stdout_text)),
        "ns3_returncode": ns3_proc.returncode,
        "replay_error": replay_error,
        "ns3_stdout": str(ns3_stdout),
        "ns3_stderr": str(ns3_stderr),
        "upload_plan": str(upload_plan_copy),
    }
    summary.update(rlc_summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if replay_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
