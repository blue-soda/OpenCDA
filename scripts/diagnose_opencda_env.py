#!/usr/bin/env python3
import argparse
import importlib
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_CARLA_DIR = REPO_ROOT / "cache" / "carla"
CACHE_CARLA_COMPAT_DIR = CACHE_CARLA_DIR / "compat"
TORCH_LIB_PATH = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "torch" / "lib" / "libtorch_cpu.so"
KNOWN_LFS_MODELS = [
    REPO_ROOT / "opencood" / "logs" / "pointpillar_early_fusion" / "latest.pth",
    REPO_ROOT / "opencood" / "logs" / "pointpillar_attentive_fusion" / "latest.pth",
    REPO_ROOT / "opencood" / "logs" / "pointpillar_v2xvit_fusion" / "net_epoch60.pth",
    REPO_ROOT / "opencood" / "logs" / "pointpillar_cobevt_fusion" / "net_epoch91.pth",
    REPO_ROOT / "opencood" / "logs" / "pointpillar_late_fusion" / "net_epoch30.pth",
]
KNOWN_COMPAT_LIBS = ("libjpeg.so.8", "libtiff.so.5", "libwebp.so.6")
SPCONV_PACKAGE_BY_CUDA = {
    "10.2": "spconv-cu102",
    "11.1": "spconv-cu111",
    "11.3": "spconv-cu113",
    "11.4": "spconv-cu114",
    "11.7": "spconv-cu117",
    "11.8": "spconv-cu118",
    "12.0": "spconv-cu120",
    "12.1": "spconv-cu121",
    "12.2": "spconv-cu122",
    "12.3": "spconv-cu123",
    "12.4": "spconv-cu124",
}
SPCONV_PACKAGES = sorted(set(SPCONV_PACKAGE_BY_CUDA.values()) | {"spconv"})


class Diagnosis:
    def __init__(self, auto_fix: bool, verbose: bool):
        self.auto_fix = auto_fix
        self.verbose = verbose
        self.issues: List[str] = []
        self.fixed: List[str] = []
        self.notes: List[str] = []

    def info(self, message: str) -> None:
        print(f"[info] {message}")

    def warn(self, message: str) -> None:
        print(f"[warn] {message}")
        self.issues.append(message)

    def ok(self, message: str) -> None:
        print(f"[ok] {message}")

    def fix(self, message: str) -> None:
        print(f"[fix] {message}")
        self.fixed.append(message)

    def note(self, message: str) -> None:
        print(f"[note] {message}")
        self.notes.append(message)


def run(cmd: Sequence[str], cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def pip_install(diag: Diagnosis, packages: Sequence[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    proc = run(cmd, cwd=REPO_ROOT)
    if proc.returncode == 0:
        diag.fix("Installed packages: %s" % " ".join(packages))
        return True
    diag.warn("pip install failed for %s\n%s%s" % (
        " ".join(packages),
        proc.stdout,
        proc.stderr,
    ))
    return False


def pip_uninstall(diag: Diagnosis, packages: Sequence[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
    proc = run(cmd, cwd=REPO_ROOT)
    if proc.returncode == 0:
        diag.fix("Uninstalled conflicting packages: %s" % " ".join(packages))
        return True
    diag.note("pip uninstall returned non-zero for %s" % " ".join(packages))
    return False


def check_python(diag: Diagnosis) -> None:
    version = sys.version_info
    diag.info(f"Python: {version.major}.{version.minor}.{version.micro} ({sys.executable})")
    if (version.major, version.minor) != (3, 7):
        diag.warn("OpenCDA is developed around Python 3.7. Current Python is %d.%d." % (version.major, version.minor))
    else:
        diag.ok("Python version matches the expected 3.7 runtime.")


def parse_ldd_missing(binary_path: Path) -> List[str]:
    proc = run(["ldd", str(binary_path)])
    missing = []
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        match = re.search(r"(\S+)\s*=>\s*not found", line)
        if match:
            missing.append(match.group(1))
    return missing


def find_library_candidates(libname: str) -> List[Path]:
    candidates: List[Path] = []
    search_roots = [
        Path.home() / ".local" / "share" / "Steam",
        Path("/var/lib/flatpak/app"),
    ]
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for match in root.rglob(libname):
            if match.is_file():
                key = str(match)
                if key not in seen:
                    candidates.append(match)
                    seen.add(key)
    return candidates


def ensure_carla_compat(diag: Diagnosis) -> None:
    binary = CACHE_CARLA_DIR / "libcarla.cpython-37m-x86_64-linux-gnu.so"
    if not binary.exists():
        diag.note("CARLA Python binary not found under cache/. Skipping compat-library checks.")
        return

    missing = parse_ldd_missing(binary)
    if not missing:
        diag.ok("CARLA binary shared-library dependencies are resolvable by the current process.")
        return

    unresolved = [name for name in missing if name in KNOWN_COMPAT_LIBS]
    compat_present = [name for name in KNOWN_COMPAT_LIBS if (CACHE_CARLA_COMPAT_DIR / name).exists()]
    if unresolved and set(unresolved).issubset(set(compat_present)):
        try:
            importlib.import_module("carla")
        except Exception:
            pass
        else:
            diag.ok("CARLA compatibility libraries are staged in cache/carla/compat and `import carla` succeeds.")
            return

    if not unresolved:
        diag.warn("CARLA binary has unresolved libraries: %s" % ", ".join(missing))
        return

    diag.warn("CARLA binary is missing compatibility libraries: %s" % ", ".join(unresolved))
    if not diag.auto_fix:
        diag.note("Run this script with --auto-fix to copy compatible shared libraries into cache/carla/compat.")
        return

    CACHE_CARLA_COMPAT_DIR.mkdir(parents=True, exist_ok=True)
    unresolved_after_copy = []
    for libname in unresolved:
        target = CACHE_CARLA_COMPAT_DIR / libname
        if target.exists():
            continue
        candidates = find_library_candidates(libname)
        if not candidates:
            unresolved_after_copy.append(libname)
            continue
        shutil.copy2(candidates[0], target)
        diag.fix(f"Copied {libname} from {candidates[0]} to {target}")

    missing_after = parse_ldd_missing(binary)
    if any(name in KNOWN_COMPAT_LIBS for name in missing_after):
        diag.warn("CARLA binary still has unresolved compat libraries after attempted fix: %s" % ", ".join(missing_after))
    else:
        diag.ok("CARLA compatibility libraries are present.")


def ensure_libtorch_stack(diag: Diagnosis) -> None:
    if not TORCH_LIB_PATH.exists():
        diag.note("libtorch_cpu.so not found in current Python environment. Skipping GNU_STACK check.")
        return

    with TORCH_LIB_PATH.open("rb") as f:
        ident = f.read(64)
        if ident[:4] != b"\x7fELF" or ident[4] != 2 or ident[5] != 1:
            diag.note("libtorch_cpu.so is not a little-endian ELF64 file. Skipping GNU_STACK patch.")
            return
        e_phoff = struct.unpack_from("<Q", ident, 32)[0]
        e_phentsize = struct.unpack_from("<H", ident, 54)[0]
        e_phnum = struct.unpack_from("<H", ident, 56)[0]

    with TORCH_LIB_PATH.open("rb") as f:
        f.seek(e_phoff)
        program_headers = [f.read(e_phentsize) for _ in range(e_phnum)]

    exec_stack_index = None
    exec_stack_flags = None
    for index, ph in enumerate(program_headers):
        p_type, p_flags = struct.unpack_from("<II", ph, 0)
        if p_type == 0x6474E551:
            exec_stack_index = index
            exec_stack_flags = p_flags
            break

    if exec_stack_index is None:
        diag.note("PT_GNU_STACK not found in libtorch_cpu.so.")
        return

    if exec_stack_flags is not None and not (exec_stack_flags & 0x1):
        diag.ok("libtorch_cpu.so already uses a non-executable stack.")
        return

    diag.warn("libtorch_cpu.so requests an executable stack, which breaks import on hardened Linux systems.")
    if not diag.auto_fix:
        diag.note("Run this script with --auto-fix to patch PT_GNU_STACK from RWE to RW.")
        return

    backup = TORCH_LIB_PATH.with_name(TORCH_LIB_PATH.name + ".gnu_stack.bak")
    if not backup.exists():
        shutil.copy2(TORCH_LIB_PATH, backup)
        diag.fix(f"Created backup at {backup}")

    with TORCH_LIB_PATH.open("r+b") as f:
        off = e_phoff + exec_stack_index * e_phentsize
        f.seek(off)
        ph = bytearray(f.read(e_phentsize))
        _, p_flags = struct.unpack_from("<II", ph, 0)
        struct.pack_into("<I", ph, 4, p_flags & ~0x1)
        f.seek(off)
        f.write(ph)
    diag.fix("Patched libtorch_cpu.so PT_GNU_STACK flags to remove execute permission.")


def detect_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except OSError:
        return False


def ensure_model_checkpoints(diag: Diagnosis) -> None:
    pointer_paths = [path for path in KNOWN_LFS_MODELS if path.exists() and detect_lfs_pointer(path)]
    if not pointer_paths:
        diag.ok("OpenCOOD checkpoints are present as binary files.")
        return

    diag.warn("OpenCOOD checkpoints are still Git LFS pointer files: %s" % ", ".join(str(p.relative_to(REPO_ROOT)) for p in pointer_paths))
    git_lfs = shutil.which("git-lfs") or shutil.which("git")
    if not diag.auto_fix:
        diag.note("Install git-lfs and run `git lfs pull` for the listed checkpoint files.")
        return
    if not shutil.which("git-lfs"):
        diag.warn("git-lfs is not installed. Automatic checkpoint fetch is unavailable.")
        return

    include = ",".join(str(path.relative_to(REPO_ROOT)) for path in pointer_paths)
    run(["git", "lfs", "install"], cwd=REPO_ROOT)
    proc = run(["git", "lfs", "pull", f"--include={include}"], cwd=REPO_ROOT)
    if proc.returncode == 0:
        remaining = [path for path in pointer_paths if path.exists() and detect_lfs_pointer(path)]
        if remaining:
            diag.warn("git lfs pull completed but some checkpoints are still pointer files: %s" % ", ".join(str(p.relative_to(REPO_ROOT)) for p in remaining))
        else:
            diag.fix("Fetched OpenCOOD checkpoints through Git LFS.")
    else:
        diag.warn("git lfs pull failed.\n%s%s" % (proc.stdout, proc.stderr))


def load_torch_version() -> Tuple[Optional[str], Optional[str]]:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return None, None
    return getattr(torch, "__version__", None), getattr(torch.version, "cuda", None)


def try_import_spconv_api() -> Tuple[bool, str]:
    try:
        from spconv.utils import Point2VoxelCPU3d  # noqa: F401
        return True, "spconv Point2VoxelCPU3d import succeeded."
    except Exception as exc:
        return False, repr(exc)


def ensure_spconv(diag: Diagnosis) -> None:
    torch_version, torch_cuda = load_torch_version()
    if not torch_version:
        diag.warn("PyTorch is not importable. Install torch before validating spconv.")
        return

    diag.info(f"PyTorch version: {torch_version}, CUDA: {torch_cuda}")
    if not torch_cuda:
        diag.note("CPU-only PyTorch detected. spconv auto-resolution is skipped.")
        return

    target_package = SPCONV_PACKAGE_BY_CUDA.get(torch_cuda)
    if not target_package:
        diag.warn(f"No automatic spconv package mapping is defined for CUDA {torch_cuda}.")
        return

    try:
        spconv = importlib.import_module("spconv")
        spconv_version = getattr(spconv, "__version__", "unknown")
        diag.info(f"spconv version: {spconv_version}")
    except Exception as exc:
        diag.warn(f"spconv import failed: {exc!r}")
        spconv = None

    ok, detail = try_import_spconv_api()
    if ok:
        diag.ok(detail)
        return

    diag.warn("spconv API import failed: %s" % detail)
    if not diag.auto_fix:
        diag.note(f"Install a CUDA-matched spconv package such as `{target_package}`.")
        return

    pip_uninstall(diag, SPCONV_PACKAGES)
    if pip_install(diag, ["--force-reinstall", target_package]):
        ok, detail = try_import_spconv_api()
        if ok:
            diag.fix(f"spconv repaired by installing {target_package}.")
        else:
            diag.warn("spconv still fails after reinstall: %s" % detail)


def ensure_carla_import(diag: Diagnosis) -> None:
    try:
        importlib.import_module("carla")
    except Exception as exc:
        diag.warn(f"Importing carla still fails: {exc!r}")
    else:
        diag.ok("`import carla` succeeded.")


def detect_open_ports(host: str, ports: Sequence[int]) -> List[int]:
    open_ports = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            if sock.connect_ex((host, port)) == 0:
                open_ports.append(port)
        finally:
            sock.close()
    return open_ports


def check_carla_runtime(diag: Diagnosis, host: str, port: int) -> None:
    open_ports = detect_open_ports(host, [port, port + 1])
    if port not in open_ports:
        diag.note(f"CARLA RPC port {host}:{port} is not listening. Runtime checks skipped.")
        return

    try:
        carla = importlib.import_module("carla")
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        current_map = client.get_world().get_map().name
        diag.ok(f"CARLA runtime is reachable at {host}:{port}; current map: {current_map}")
    except Exception as exc:
        diag.warn(f"CARLA runtime port is open but the client handshake failed: {exc!r}")


def summarize(diag: Diagnosis) -> int:
    print("\nSummary")
    print(f"- Fixed: {len(diag.fixed)}")
    print(f"- Issues: {len(diag.issues)}")
    print(f"- Notes: {len(diag.notes)}")
    if diag.issues:
        print("\nOutstanding diagnostics:")
        for issue in diag.issues:
            print(f"- {issue}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose and optionally repair a local OpenCDA environment.")
    parser.add_argument("--auto-fix", action="store_true", help="Apply safe automated fixes when a known issue is detected.")
    parser.add_argument("--verbose", action="store_true", help="Reserved for future detailed logging.")
    parser.add_argument("--carla-host", default="localhost", help="CARLA host to probe when checking runtime availability.")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA RPC port to probe when checking runtime availability.")
    args = parser.parse_args()

    diag = Diagnosis(auto_fix=args.auto_fix, verbose=args.verbose)
    os.chdir(REPO_ROOT)

    check_python(diag)
    ensure_carla_compat(diag)
    ensure_libtorch_stack(diag)
    ensure_model_checkpoints(diag)
    ensure_spconv(diag)
    ensure_carla_import(diag)
    check_carla_runtime(diag, args.carla_host, args.carla_port)

    return summarize(diag)


if __name__ == "__main__":
    sys.exit(main())
