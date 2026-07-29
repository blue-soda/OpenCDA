# -*- coding: utf-8 -*-
"""Run dense SGCP dynamic-C/V + density-capped sensitivity experiments."""

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


REPO = Path(r"C:\Workspace\OpenCDA")
ARTIFACT = REPO / (
    "docs/doc_workspace/SGCP/artifacts/"
    "dense_dynamic_cv_sensitivity_20260729"
)

BOX_BYTES = 80
MESSAGE_OVERHEAD_BYTES = 64
FRAME_INTERVAL_S = 0.1

SCENARIO_ID = "2026_07_29_02_32_08"
DATASET_ROOT = r"D:\Data\Carla"
COPERCEPTION_YAML = (
    r"docs\doc_workspace\SGCP\artifacts"
    r"\early_from_late_checkpoint_20260719"
    r"\enable_coperception_early_from_attentive.yaml"
)

BASE_ARGS = [
    "-m", "opencda.tools.offline_inference",
    "--dataset-root", DATASET_ROOT,
    "--scenario-id", SCENARIO_ID,
    "--ego-cav-id", "1",
    "--max-frames", "41",
    "--fusion-method", "early",
    "--coperception-yaml", COPERCEPTION_YAML,
    "--sgcp-constrained",
    "--clustering", "potential_verified_cov_coalition_game",
    "--sgcp-receiver-policy", "all-cluster-heads",
    "--sgcp-upload-mode", "grid",
    "--resource-allocation", "dynamic_cv",
    "--sgcp-inter-cluster-late-fusion",
    "--sgcp-grid-selection-mode", "utility",
    "--sgcp-grid-score-mode", "utility",
    "--head-rb-budget", "2",
    "--num-channels", "10",
    "--bandwidth-mhz", "40",
    "--communication-deadline-ms", "60",
    "--channel-estimator", "ns3",
    "--ns3-tb-size-bytes", "899",
    "--ns3-slot-duration-ms", "0.5",
    "--ns3-subchannel-prbs", "10",
    "--ns3-symbols-per-slot", "12",
    "--ns3-mcs", "28",
]

RHO_RUNS = [
    ("rho0p1", "0.1"),
    ("rho0p2", "0.2"),
    ("rho0p5", "0.5"),
    ("rho1", "1"),
    ("rho2", "2"),
    ("rho5", "5"),
    ("rho10", "10"),
]

BUDGET_VALUES = ["1", "5", "10", "20", "40", "60", "68", "84"]

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


def safe_float(value):
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value):
    try:
        if value in ("", None):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def most_common(values):
    values = [value for value in values if value not in ("", None)]
    if not values:
        return ""
    return max(set(values), key=values.count)


def replace_arg(args, name, value):
    updated = list(args)
    if name in updated:
        idx = updated.index(name)
        updated[idx + 1] = str(value)
    else:
        updated.extend([name, str(value)])
    return updated


def make_command(run):
    args = list(BASE_ARGS)
    args = replace_arg(args, "--rho-th", run["rho_th"])
    args = replace_arg(args, "--upload-density-cap-rho", run["rho_th"])
    args = replace_arg(args, "--sgcp-frame-mbps-budget", run["raw_budget"])
    args.extend(["--sgcp-trace-output", str(run["trace_path"])])
    args.extend(["--eval-stats-output", str(run["eval_path"])])
    return ["conda", "run", "--no-capture-output", "-n", "opencda", "python"] + args


def all_runs(phase):
    runs = []
    if phase in ("rho", "all"):
        for stem, rho_th in RHO_RUNS:
            runs.append({
                "group": "rho_th",
                "setting": rho_th,
                "stem": "rho_sweep_%s" % stem,
                "rho_th": rho_th,
                "raw_budget": "68",
            })
    if phase == "budget":
        raise ValueError("Use --budget-rho for budget-only runs.")
    return runs


def budget_runs(rho_values):
    runs = []
    for rho_th in rho_values:
        rho_stem = str(rho_th).replace(".", "p")
        for budget in BUDGET_VALUES:
            runs.append({
                "group": "raw_mbps_budget",
                "setting": budget,
                "stem": "budget_rho%s_mbps%s" % (rho_stem, budget),
                "rho_th": str(rho_th),
                "raw_budget": budget,
            })
    return runs


def prepare_runs(runs):
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run["run_dir"] = ARTIFACT / run["stem"]
        run["run_dir"].mkdir(parents=True, exist_ok=True)
        run["trace_path"] = run["run_dir"] / "trace.csv"
        run["eval_path"] = run["run_dir"] / "eval_stats.csv"
        run["log_path"] = run["run_dir"] / "run.out"
        run["command_path"] = run["run_dir"] / "command.txt"
    return runs


def quote_command(cmd):
    return " ".join('"%s"' % item if " " in item else item for item in cmd)


def run_experiments(runs, force=False):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for run in runs:
        if (not force and run["log_path"].exists() and
                run["trace_path"].exists()):
            print("skip existing %s" % run["stem"], flush=True)
            continue
        cmd = make_command(run)
        run["command_path"].write_text(quote_command(cmd) + "\n",
                                       encoding="utf-8")
        print("running %s" % run["stem"], flush=True)
        with run["log_path"].open("w", encoding="utf-8",
                                  errors="replace") as stream:
            proc = subprocess.run(cmd, cwd=str(REPO), env=env,
                                  stdout=stream, stderr=subprocess.STDOUT,
                                  text=True)
        if proc.returncode != 0:
            raise RuntimeError("%s failed with exit code %s; see %s" % (
                run["stem"], proc.returncode, run["log_path"]))


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    ap_matches = AP_PATTERN.findall(text)
    summary_matches = SUMMARY_PATTERN.findall(text)
    if not ap_matches or not summary_matches:
        raise RuntimeError("Missing AP or sgcp_summary in %s" % path)
    ap_03, ap_05, ap_07 = ap_matches[-1]
    trace_rows, avg_comm, total_comm, avg_sources, avg_grids = (
        summary_matches[-1]
    )
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

    timestamps = sorted({
        row.get("timestamp", "") for row in rows if row.get("timestamp", "")
    })
    by_sample = {}
    for row in rows:
        timestamp = row.get("timestamp", "")
        receiver = row.get("receiver_id", "")
        if not timestamp or not receiver:
            continue
        key = (timestamp, receiver)
        by_sample[key] = max(by_sample.get(key, 0),
                             safe_int(row.get("pred_boxes")))
    box_bytes = sum(
        MESSAGE_OVERHEAD_BYTES + boxes * BOX_BYTES
        for boxes in by_sample.values()
        if boxes > 0
    )
    duration_s = max(len(timestamps) * FRAME_INTERVAL_S, 1e-9)
    return {
        "unique_timestamps": len(timestamps),
        "box_bytes": box_bytes,
        "box_mbps": box_bytes * 8.0 / duration_s / 1e6,
        "receiver_policy": most_common(
            [row.get("receiver_policy", "") for row in rows]),
        "clustering": most_common([row.get("clustering", "") for row in rows]),
        "resource_allocation": most_common(
            [row.get("resource_allocation", "") for row in rows]),
        "upload_mode": most_common([row.get("upload_mode", "") for row in rows]),
        "num_channels": most_common([row.get("num_channels", "") for row in rows]),
        "bandwidth_mhz": most_common([row.get("bandwidth_mhz", "") for row in rows]),
        "communication_deadline_ms": most_common(
            [row.get("communication_deadline_ms", "") for row in rows]),
        "raw_mbps_budget": most_common(
            [row.get("sgcp_frame_mbps_budget", "") for row in rows]),
        "rho_th": most_common([row.get("rho_th", "") for row in rows]),
        "upload_density_cap_rho": most_common(
            [row.get("upload_density_cap_rho", "") for row in rows]),
    }


def build_rows(runs):
    output = []
    for run in runs:
        log = parse_log(run["log_path"])
        trace = parse_trace(run["trace_path"])
        duration_s = trace["unique_timestamps"] * FRAME_INTERVAL_S
        raw_mbps = log["payload_bytes"] * 8.0 / max(duration_s, 1e-9) / 1e6
        total_mbps = raw_mbps + trace["box_mbps"]
        output.append({
            "label": "%s=%s" % (run["group"], run["setting"]),
            "parameter": run["group"],
            "setting": run["setting"],
            "rho_th": run["rho_th"],
            "raw_budget_mbps": run["raw_budget"],
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
            "late_fusion": "inter_cluster_nms",
            "resource_allocation": trace["resource_allocation"],
            "clustering": trace["clustering"],
            "upload_mode": trace["upload_mode"],
            "num_channels": trace["num_channels"],
            "bandwidth_mhz": trace["bandwidth_mhz"],
            "communication_deadline_ms": trace["communication_deadline_ms"],
            "checkpoint": "attentive",
            "artifact_stem": run["stem"],
            "trace_path": str(run["trace_path"].relative_to(REPO)),
            "log_path": str(run["log_path"].relative_to(REPO)),
        })
    return output


def write_csv(rows, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(path, flush=True)
    return path


def run_compute_profile(rows, suffix):
    methods = []
    for row in rows:
        methods.extend([
            "--method",
            "%s=%s" % (row["artifact_stem"], REPO / row["trace_path"]),
        ])
    metrics_csv = ARTIFACT / ("metrics_%s.csv" % suffix)
    write_csv(rows, metrics_csv)
    output_csv = ARTIFACT / ("compute_%s.csv" % suffix)
    summary_md = ARTIFACT / ("compute_%s.md" % suffix)
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "opencda", "python",
        "-m", "opencda.tools.sgcp_compute_profile",
    ] + methods + [
        "--metrics-csv", str(metrics_csv),
        "--calibration-json",
        str(REPO / (
            "docs/doc_workspace/SGCP/artifacts/compute_profile_20260722/"
            "attentive_singleton_forward_flops.json")),
        "--dense-calibration-json",
        str(REPO / (
            "docs/doc_workspace/SGCP/artifacts/compute_profile_20260722/"
            "attentive_full20_forward_flops.json")),
        "--output-csv", str(output_csv),
        "--summary-md", str(summary_md),
    ]
    log_path = ARTIFACT / ("compute_%s.log" % suffix)
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=stream,
                              stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError("compute profile failed; see %s" % log_path)
    by_label = {}
    with output_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_label[row["label"]] = row
    for row in rows:
        comp = by_label.get(row["artifact_stem"], {})
        row["input_adjusted_gflops_per_frame"] = (
            "" if not comp else "%.2f" % safe_float(
                comp.get("input_adjusted_detector_gflops_per_frame")))
        row["detector_calls_per_frame"] = (
            "" if not comp else comp.get("detector_calls_per_frame", ""))
    return output_csv, summary_md


def markdown_table(rows, group):
    lines = [
        "| Setting | rho_th | Raw budget (Mbps) | AP@0.3 | AP@0.5 | AP@0.7 | Raw LiDAR Mbps | Box Mbps | Total Mbps | GFLOPs/frame | Avg source CAVs | Avg selected grids |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["parameter"] != group:
            continue
        lines.append(
            "| {setting} | {rho_th} | {raw_budget_mbps} | {ap_03} | {ap_05} | {ap_07} | {raw_lidar_mbps} | {box_mbps} | {total_mbps} | {input_adjusted_gflops_per_frame} | {avg_source_cavs} | {avg_selected_grids} |".format(**row)
        )
    return lines


def write_summary(rows, suffix):
    lines = [
        "# Dense SGCP Dynamic C/V Parameter Sensitivity",
        "",
        "Protocol: dense `v2xp_cluster_carla` dump `%s`, 20 CAVs, 41 frames, attentive checkpoint, `40 MHz / 10 ch`, NS3-calibrated estimator (`tb_size=899`, `slot=0.5 ms`, `symbols=12`, `MCS=28`), `potential_verified_cov_coalition_game` clustering, `dynamic_cv` resource allocation, all cluster heads as receivers, grid upload, inter-cluster box NMS. `rho_th` and upload density cap use the same value; receiver-side residual density is updated after each admitted grid upload." % SCENARIO_ID,
        "",
        "No explicit `N_max` is passed; for this protocol it follows `N_max = ceil(N / floor(K / B_h)) = ceil(20 / floor(10 / 2)) = 4`.",
        "",
    ]
    if any(row["parameter"] == "rho_th" for row in rows):
        lines.extend(["## rho_th Sweep", ""])
        lines.extend(markdown_table(rows, "rho_th"))
        lines.append("")
    if any(row["parameter"] == "raw_mbps_budget" for row in rows):
        lines.extend(["## Raw LiDAR Mbps Budget", ""])
        lines.extend(markdown_table(rows, "raw_mbps_budget"))
        lines.append("")
    output = ARTIFACT / ("summary_%s.md" % suffix)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output, flush=True)
    return output


def write_result_bundle(runs, suffix):
    rows = build_rows(runs)
    run_compute_profile(rows, suffix)
    rows_with_compute = ARTIFACT / ("results_%s_with_gflops.csv" % suffix)
    write_csv(rows, rows_with_compute)
    write_summary(rows, suffix)
    for row in rows:
        print(
            "{label}: AP={ap_03}/{ap_05}/{ap_07}, total={total_mbps} Mbps, "
            "GFLOPs={input_adjusted_gflops_per_frame}, grids={avg_selected_grids}".format(**row),
            flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["rho", "budget", "all"],
                        default="rho")
    parser.add_argument("--budget-rho", action="append", default=[],
                        help="rho_th value for budget sweep; repeatable.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    if args.phase == "budget":
        if not args.budget_rho:
            raise ValueError("--phase budget requires --budget-rho")
        runs = budget_runs(args.budget_rho)
    else:
        runs = all_runs(args.phase)
        if args.phase == "all":
            budget_rho = args.budget_rho or ["5"]
            runs.extend(budget_runs(budget_rho))
    runs = prepare_runs(runs)
    if not args.skip_run:
        run_experiments(runs, force=args.force)
    write_result_bundle(runs, args.phase)


if __name__ == "__main__":
    main()
