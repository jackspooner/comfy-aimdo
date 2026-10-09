"""Check local prerequisites and build from an explicitly provisioned archive."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "d49d682f5d2386c173b6e006967ecfee37c45fd1eef3e9d4232992ccdaca4ef5"


def main():
    for command in ("gcc", "cmake", "make", "bash"):
        if not shutil.which(command):
            raise SystemExit(f"Missing {command}; provision the build tools before setup.")
    archive = Path(os.environ.get("AIMDO_FUNCHOOK_ARCHIVE", ROOT / "build/funchook-1.1.3.tar.gz"))
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit("Provision verified funchook 1.1.3 via Download Manager 2 and set AIMDO_FUNCHOOK_ARCHIVE.")
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    if not (build / "funchook-1.1.3/CMakeLists.txt").exists():
        with tarfile.open(archive) as source:
            source.extractall(build, filter="data")
    subprocess.run(["bash", "scripts/build-linux-aimdo.sh"], cwd=ROOT, check=True)
    print("Native libraries built locally; no GPU workload or service was started.")


if __name__ == "__main__":
    main()
