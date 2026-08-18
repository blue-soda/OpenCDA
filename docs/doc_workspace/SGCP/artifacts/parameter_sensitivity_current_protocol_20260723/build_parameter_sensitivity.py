# -*- coding: utf-8 -*-
"""Run SGCP parameter sensitivity around the clean-package operating point."""

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / "docs/doc_workspace/SGCP/artifacts/parameter_sensitivity_current_protocol_20260723"

BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64
FRAME_INTERVAL_S = 0.1

BASE_ARGS = [
    "-m", "opencda.tools.offline_inference",
    "--dataset-root", r"D:\Data\Carla",
    "--scenario-id", "2026_07_15_01_26_56",
    "--ego-cav-id", "1",
    "--max-frames", "41",
    "--fusion-method", "early",
    "--coperception-yaml",
    r"docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml",
    "--sgcp-constrained",
    "--clustering", "coalition_game",
    "--sgcp-receiver-policy", "all-cluster-heads",
    "--sgcp-upload-mode", "grid",
    "--resource-allocation", "perception_aware_potential_game",
    "--sgcp-inter-cluster-late-fusion",
    "--sgcp-grid-selection-mode", "utility",
    "--sgcp-grid-score-mode", "utility",
    "--n-max", "4",
    "--rho-th", "3",
    "--head-rb-budget", "2",
    "--num-channels", "10",
    "--bandwidth-mhz", "40",
    "--communication-deadline-ms", "200",
    "--channel-estimator", "ns3",
    "--ns3-tb-size-bytes", "899",
    "--ns3-slot-duration-ms", "0.5",
    "--ns3-subchannel-prbs", "10",
    "--ns3-symbols-per-slot", "12",
    "--ns3-mcs", "28",
]

RUNS = [
    {"group": "rho_th", "setting": "0.01", "stem": "rho0p01", "extra": ["--rho-th", "0.01"]},
    {"group": "rho_th", "setting": "0.03", "stem": "rho0p03", "extra": ["--rho-th", "0.03"]},
    {"group": "rho_th", "setting": "0.05", "stem": "rho0p05", "extra": ["--rho-th", "0.05"]},
    {"group": "rho_th", "setting": "0.1", "stem": "rho0p1", "extra": ["--rho-th", "0.1"]},
    {"group": "rho_th", "setting": "0.3", "stem": "rho0p3", "extra": ["--rho-th", "0.3"]},
    {"group": "rho_th", "setting": "0.5", "stem": "rho0p5", "extra": ["--rho-th", "0.5"]},
    {"group": "rho_th", "setting": "1", "stem": "rho1", "extra": ["--rho-th", "1"]},
    {"group": "rho_th", "setting": "2", "stem": "rho2", "extra": ["--rho-th", "2"]},
    {"group": "rho_th", "setting": "3", "stem": "base", "extra": []},
    {"group": "N_max", "setting": "2", "stem": "nmax2", "extra": ["--n-max", "2"]},
    {"group": "N_max", "setting": "3", "stem": "nmax3", "extra": ["--n-max", "3"]},
    {"group": "N_max", "setting": "4", "stem": "base", "extra": [], "reuse": True},
    {"group": "N_max", "setting": "5", "stem": "nmax5", "extra": ["--n-max", "5"]},
    {"group": "N_max", "setting": "6", "stem": "nmax6", "extra": ["--n-max", "6"]},
    {"group": "target_subchannels", "setting": "5", "stem": "ch5", "extra": ["--num-channels", "5"]},
    {"group": "target_subchannels", "setting": "10", "stem": "base", "extra": [], "reuse": True},
    {"group": "target_subchannels", "setting": "20", "stem": "ch20", "extra": ["--num-channels", "20"]},
    {"group": "communication_budget_ms", "setting": "40", "stem": "budget40", "extra": ["--communication-deadline-ms", "40"]},
    {"group": "communication_budget_ms", "setting": "60", "stem": "budget60", "extra": ["--communication-deadline-ms", "60"]},
    {"group": "communication_budget_ms", "setting": "100", "stem": "budget100", "extra": ["--communication-deadline-ms", "100"]},
    {"group": "communication_budget_ms", "setting": "200", "stem": "base", "extra": [], "reuse": True},
    {"group": "communication_budget_ms", "setting": "300", "stem": "budget300", "extra": ["--communication-deadline-ms", "300"]},
]

AP_PATTERN = re.compile(
    r"The Average Precision at IOU 0\.3 is\s+([0-9.]+),\s+"
    r"The Average Precision at IOU 0\.5 is\s+([0-9.]+),\s+"
    r"The Average Precision at IOU 0\.7 is\s+([0-9.]+)"
)
SUMMARY_PATTERN = re.compile(
    r"sgcp_summary\s+frames=(?P<trace_rows>\d+)\s+"
    r"avg_comm_bytes=(?P<avg_comm>[0-9.]+)\s+"
    r"total_comm_bytes=(?P<total_comm>\d+)\s+"
    r"avg_source_cavs=(?P<avg_sources>[0-9.]+)\s+"
    r"avg_selected_grids=(?P<avg_grids>[0-9.]+)"
)


def replace_arg(args, name, value):
    updated = list(args)
    if name in updated:
        idx = updated.index(name)
        updated[idx + 1] = str(value)
    else:
        updated.extend([name, str(value)])
    return updated


def safe_int(value):
    try:
        if value in ("", None):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def safe_float(value):
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def most_common(values):
    values = [value for value in values if value not in ("", None)]
    if not values:
        return ""
    return max(set(values), key=values.count)


def command_for(run):
    args = list(BASE_ARGS)
    for idx in range(0, len(run.get("extra", [])), 2):
        args = replace_arg(args, run["extra"][idx], run["extra"][idx + 1])
    args.extend(["--sgcp-trace-output", str(ARTIFACT / ("%s_trace.csv" % run["stem"]))])
    args.extend(["--eval-stats-output", str(ARTIFACT / ("%s_eval_stats.csv" % run["stem"]))])
    return ["conda", "run", "--no-capture-output", "-n", "opencda", "python"] + args


def run_experiments(force=False):
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run_by_stem = {}
    for run in RUNS:
        run_by_stem.setdefault(run["stem"], run)
    for run in run_by_stem.values():
        log_path = ARTIFACT / ("%s.log" % run["stem"])
        trace_path = ARTIFACT / ("%s_trace.csv" % run["stem"])
        if not force and log_path.exists() and trace_path.exists():
            print("skip existing %s" % run["stem"])
            continue
        cmd = command_for(run)
        print("running %s=%s stem=%s" % (run["group"], run["setting"], run["stem"]))
        print(" ".join('"%s"' % p if " " in p else p for p in cmd))
        with log_path.open("w", encoding="utf-8", errors="replace") as stream:
            proc = subprocess.run(
                cmd, cwd=str(REPO), env=env, stdout=stream,
                stderr=subprocess.STDOUT, text=True
            )
        if proc.returncode != 0:
            raise RuntimeError("%s failed with exit code %s; see %s" % (
                run["stem"], proc.returncode, log_path))


def parse_log(path):
    text = path.read_text(errors="replace")
    ap_matches = AP_PATTERN.findall(text)
    summary_matches = SUMMARY_PATTERN.findall(text)
    if not ap_matches or not summary_matches:
        raise RuntimeError("Missing AP or sgcp_summary in %s" % path)
    ap_03, ap_05, ap_07 = ap_matches[-1]
    trace_rows, avg_comm, total_comm, avg_sources, avg_grids = summary_matches[-1]
    return {
        "ap_03": safe_float(ap_03),
        "ap_05": safe_float(ap_05),
        "ap_07": safe_float(ap_07),
        "trace_rows": safe_int(trace_rows),
        "avg_comm_bytes_per_trace_row": safe_float(avg_comm),
        "payload_bytes": safe_int(total_comm),
        "avg_source_cavs": safe_float(avg_sources),
        "avg_selected_grids": safe_float(avg_grids),
    }


def parse_trace(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Empty trace: %s" % path)
    timestamps = sorted({row.get("timestamp", "") for row in rows if row.get("timestamp", "")})
    by_sample = {}
    for row in rows:
        timestamp = row.get("timestamp", "")
        receiver = row.get("receiver_id", "")
        if not timestamp or not receiver:
            continue
        key = (timestamp, receiver)
        by_sample[key] = max(by_sample.get(key, 0), safe_int(row.get("pred_boxes")))
    box_bytes = sum(
        MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
        for boxes in by_sample.values()
        if boxes > 0
    )
    duration_s = max(len(timestamps) * FRAME_INTERVAL_S, 1e-9)
    return {
        "unique_timestamps": len(timestamps),
        "receiver_policy": most_common([row.get("receiver_policy", "") for row in rows]),
        "late_fusion": "inter_cluster_nms",
        "clustering": most_common([row.get("clustering", "") for row in rows]),
        "resource_allocation": most_common([row.get("resource_allocation", "") for row in rows]),
        "upload_mode": most_common([row.get("upload_mode", "") for row in rows]),
        "num_channels": most_common([row.get("num_channels", "") for row in rows]),
        "bandwidth_mhz": most_common([row.get("bandwidth_mhz", "") for row in rows]),
        "communication_deadline_ms": most_common(
            [row.get("communication_deadline_ms", "") for row in rows]),
        "box_bytes": box_bytes,
        "box_mbps": box_bytes * 8.0 / duration_s / 1e6,
    }


def build_rows():
    rows = []
    for run in RUNS:
        log = parse_log(ARTIFACT / ("%s.log" % run["stem"]))
        trace = parse_trace(ARTIFACT / ("%s_trace.csv" % run["stem"]))
        duration_s = trace["unique_timestamps"] * FRAME_INTERVAL_S
        raw_mbps = log["payload_bytes"] * 8.0 / max(duration_s, 1e-9) / 1e6
        total_mbps = raw_mbps + trace["box_mbps"]
        rows.append({
            "label": "%s=%s" % (run["group"], run["setting"]),
            "parameter": run["group"],
            "setting": run["setting"],
            "ap_03": "%.2f" % log["ap_03"],
            "ap_05": "%.2f" % log["ap_05"],
            "ap_07": "%.2f" % log["ap_07"],
            "raw_lidar_mbps": "%.2f" % raw_mbps,
            "box_mbps": "%.2f" % trace["box_mbps"],
            "total_mbps": "%.2f" % total_mbps,
            "avg_source_cavs": "%.2f" % log["avg_source_cavs"],
            "avg_selected_grids": "%.2f" % log["avg_selected_grids"],
            "evaluated_frames": str(trace["unique_timestamps"]),
            "trace_rows": str(log["trace_rows"]),
            "receiver_policy": trace["receiver_policy"],
            "late_fusion": trace["late_fusion"],
            "resource_allocation": trace["resource_allocation"],
            "clustering": trace["clustering"],
            "upload_mode": trace["upload_mode"],
            "num_channels": trace["num_channels"],
            "bandwidth_mhz": trace["bandwidth_mhz"],
            "communication_budget_ms": trace["communication_deadline_ms"],
            "checkpoint": "attentive",
            "artifact_stem": run["stem"],
            "trace_path": str((ARTIFACT / ("%s_trace.csv" % run["stem"])).relative_to(REPO)),
            "log_path": str((ARTIFACT / ("%s.log" % run["stem"])).relative_to(REPO)),
        })
    return rows


def write_csv(rows):
    path = ARTIFACT / "parameter_sensitivity_current_protocol_20260723.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(path)
    return path


def run_compute_profile(metrics_csv):
    methods = []
    seen = set()
    for run in RUNS:
        if run["stem"] in seen:
            continue
        seen.add(run["stem"])
        methods.extend([
            "--method",
            "%s=%s" % (run["stem"], ARTIFACT / ("%s_trace.csv" % run["stem"])),
        ])
    output_csv = ARTIFACT / "parameter_sensitivity_compute_profile_20260723.csv"
    summary_md = ARTIFACT / "parameter_sensitivity_compute_profile_20260723.md"
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "opencda", "python",
        "-m", "opencda.tools.sgcp_compute_profile",
    ] + methods + [
        "--metrics-csv", str(metrics_csv),
        "--calibration-json",
        str(REPO / "docs/doc_workspace/SGCP/artifacts/compute_profile_20260722/attentive_singleton_forward_flops.json"),
        "--dense-calibration-json",
        str(REPO / "docs/doc_workspace/SGCP/artifacts/compute_profile_20260722/attentive_full20_forward_flops.json"),
        "--output-csv", str(output_csv),
        "--summary-md", str(summary_md),
    ]
    log_path = ARTIFACT / "compute_profile.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=stream, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError("compute profile failed; see %s" % log_path)
    print(output_csv)
    print(summary_md)
    return output_csv


def merge_compute(rows, compute_csv):
    by_stem = {}
    with compute_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_stem[row["label"]] = row
    for row in rows:
        comp = by_stem.get(row["artifact_stem"], {})
        row["input_adjusted_gflops_per_frame"] = (
            "" if not comp else "%.2f" % safe_float(
                comp.get("input_adjusted_detector_gflops_per_frame")))
        row["detector_calls_per_frame"] = (
            "" if not comp else comp.get("detector_calls_per_frame", ""))
    path = ARTIFACT / "parameter_sensitivity_current_protocol_with_gflops_20260723.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(path)
    return path


def write_markdown(rows):
    groups = [
        ("rho_th", "rho_th"),
        ("N_max", "N_max"),
        ("target_subchannels", "Target Subchannels"),
        ("communication_budget_ms", "Communication Budget"),
    ]
    lines = [
        "# SGCP Parameter Sensitivity",
        "",
        "Protocol: attentive detector, v2xp_cluster_carla, 41 frames, 20 CAVs, 40 MHz total bandwidth, NS3-calibrated estimator, PAPG scheduler, coalition-game clustering, all cluster heads as receivers, grid upload, inter-cluster box NMS. Unless varied, N_max=4, rho_th=3, head_rb_budget=2, target subchannels=10, and scheduler communication budget=200 ms. The headline point is retained because exact NS3 replay measured sub-60 ms delivery for the selected payload.",
        "",
        "Box-level communication for inter-cluster NMS is included in total Mbps.",
        "",
    ]
    for group, title in groups:
        lines.extend([
            "## %s" % title,
            "",
            "| Setting | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in rows:
            if row["parameter"] != group:
                continue
            lines.append(
                "| {setting} | {ap_03} | {ap_05} | {ap_07} | {raw_lidar_mbps} | {box_mbps} | {total_mbps} | {input_adjusted_gflops_per_frame} | {avg_source_cavs} | {avg_selected_grids} |".format(**row)
            )
        lines.append("")
    output = ARTIFACT / "parameter_sensitivity_current_protocol_20260723.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    if not args.skip_run:
        run_experiments(force=args.force)
    rows = build_rows()
    metrics_csv = write_csv(rows)
    compute_csv = run_compute_profile(metrics_csv)
    merge_compute(rows, compute_csv)
    write_markdown(rows)
    for row in rows:
        print(
            "{label}: {ap_03}/{ap_05}/{ap_07}, {total_mbps} Mbps, {input_adjusted_gflops_per_frame} GFLOPs, grids={avg_selected_grids}".format(**row)
        )


if __name__ == "__main__":
    main()
