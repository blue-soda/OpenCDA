import argparse
import json
import os
import pathlib
import re
import subprocess
import time


REPO = pathlib.Path(r"C:\Workspace\OpenCDA")
HERE = pathlib.Path(__file__).resolve().parent
WSL_NS3_ROOT = "/home/sakakibara/workspace/carla-ns3-co-simulation/ns-3-dev"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(HERE / "ns3_replay_41f_guard1_zero0"))
    parser.add_argument("--upload-plan", default=str(HERE / "sgcp_hybrid_ns3_upload_plan.csv"))
    parser.add_argument("--sim-time", type=float, default=8.0)
    parser.add_argument("--drain-seconds", type=float, default=2.0)
    parser.add_argument("--sync-timeout", type=float, default=30.0)
    parser.add_argument("--guard-ms", type=float, default=1.0)
    parser.add_argument("--zero-delay-ms", type=float, default=0.0)
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
        "--nrSlZeroTimeSendDelayMs={zero:g}"
    ).format(sim_time=args.sim_time, guard=args.guard_ms, zero=args.zero_delay_ms)
    return [
        "wsl",
        "bash",
        "-lc",
        "cd {root} && ./ns3 run '{ns3_args}'".format(
            root=WSL_NS3_ROOT, ns3_args=ns3_args),
    ]


def replay_command(args, replayed_plan):
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "opencda",
        "python",
        "-m",
        "opencda.tools.offline_ns3_replay",
        "--dataset-root",
        r"D:\Data\Carla",
        "--scenario-id",
        "2026_07_29_02_32_08",
        "--ego-cav-id",
        "1",
        "--max-frames",
        "41",
        "--lgcp-upload-plan",
        str(pathlib.Path(args.upload_plan)),
        "--upload-plan-output",
        str(replayed_plan),
        "--drain-seconds",
        str(args.drain_seconds),
        "--sync-timeout",
        str(args.sync_timeout),
    ]


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ns3_stdout = output_dir / "ns3_stdout.log"
    ns3_stderr = output_dir / "ns3_stderr.log"
    replay_stdout = output_dir / "replay_stdout.log"
    replay_stderr = output_dir / "replay_stderr.log"
    replayed_plan = output_dir / "upload_plan_replayed.csv"
    summary_path = output_dir / "process_summary.json"
    for path in [ns3_stdout, ns3_stderr, replay_stdout, replay_stderr,
                 replayed_plan, summary_path]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    with ns3_stdout.open("wb") as out, ns3_stderr.open("wb") as err:
        ns3_proc = subprocess.Popen(
            ns3_command(args),
            cwd=str(REPO),
            stdout=out,
            stderr=err)

    replay_code = None
    try:
        time.sleep(5.0)
        with replay_stdout.open("wb") as out, replay_stderr.open("wb") as err:
            replay_proc = subprocess.Popen(
                replay_command(args, replayed_plan),
                cwd=str(REPO),
                stdout=out,
                stderr=err)
            replay_code = replay_proc.wait(timeout=900)
        try:
            ns3_proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            ns3_proc.terminate()
            try:
                ns3_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                ns3_proc.kill()
                ns3_proc.wait(timeout=10)
    finally:
        if ns3_proc.poll() is None:
            ns3_proc.terminate()
            try:
                ns3_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ns3_proc.kill()
                ns3_proc.wait(timeout=5)

    stdout_text = ns3_stdout.read_text(encoding="utf-8", errors="ignore")
    summary = {
        "output_dir": str(output_dir),
        "ns3_returncode": ns3_proc.returncode,
        "replay_returncode": replay_code,
        "manual_add": len(re.findall(r"\\[MANUAL_CMD_ADD\\]", stdout_text)),
        "manual_reject": len(re.findall(r"\\[MANUAL_CMD_REJECT\\]", stdout_text)),
        "manual_consume": len(re.findall(r"\\[MANUAL_CMD_CONSUME\\]", stdout_text)),
        "cam_received": len(re.findall(r"cam_received", stdout_text)),
        "rlc_tx": len(re.findall(r"\\[NRSL_RLC_TX\\]", stdout_text)),
        "rlc_rx": len(re.findall(r"\\[NRSL_RLC_RX\\]", stdout_text)),
        "rlc_drop": len(re.findall(r"\\[NRSL_RLC_DROP\\]", stdout_text)),
        "pssch_fail": len(re.findall(r"\\[PSSCH_DECODE_FAIL\\]", stdout_text)),
        "ns3_stdout": str(ns3_stdout),
        "ns3_stderr": str(ns3_stderr),
        "replay_stdout": str(replay_stdout),
        "replay_stderr": str(replay_stderr),
        "upload_plan_replayed": str(replayed_plan),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if replay_code != 0:
        raise SystemExit(replay_code)


if __name__ == "__main__":
    main()
