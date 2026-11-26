"""Verify the archived inputs and restore the three short-trace archives."""
import hashlib
import shutil
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "reproduction" / "archives"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    for row in (ARCHIVES / "SHA256SUMS").read_text().splitlines():
        digest, name = row.split()
        if sha256(ARCHIVES / name) != digest:
            raise SystemExit(f"Archive hash mismatch: {name}. Check git lfs pull.")
        print(f"Verified {name}")
    output = ROOT / "data" / "short"
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVES / "KV_Cache_Agent_Handoff.zip") as bundle:
        for name in (
            "h2-development-20261120T045605Z.zip",
            "heldout-20261120T043814Z.zip",
            "gpu-layer-smoke-20261122T044930617682Z.zip",
        ):
            with bundle.open(f"KV_Cache_Agent_Handoff/data/short/{name}") as source:
                dest = output / name
                if dest.exists():
                    if sha256(dest) != hashlib.file_digest(source, "sha256").hexdigest():
                        raise SystemExit(f"Existing file differs: {dest}")
                else:
                    with dest.open("xb") as stream:
                        shutil.copyfileobj(source, stream)
            print(f"Ready {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
