"""Audit the submission archive against the working repository."""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREFIX = "GOGOGO/"
FIGURES = (
    "edfig1_instrument.png",
    "edfig2_fss_collapse.png",
    "fig2.png",
    "fig3.png",
    "fig4.png",
    "fig5.png",
    "fig6.png",
)


def run(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        args, cwd=cwd, check=True, text=True, capture_output=True
    )
    return proc.stdout


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_files(archive: Path) -> tuple[set[str], list[tarfile.TarInfo]]:
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
    files: set[str] = set()
    for member in members:
        name = member.name
        if name.startswith("/") or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe archive path: {name}")
        if member.isdir() and name.rstrip("/") == PREFIX.rstrip("/"):
            continue
        if not name.startswith(PREFIX):
            raise RuntimeError(f"path outside {PREFIX}: {name}")
        if member.isfile():
            files.add(name[len(PREFIX):])
    return files, members


def release_files() -> set[str]:
    output = run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], ROOT
    )
    return {line for line in output.splitlines() if line}


def check_latex_log(path: Path) -> None:
    text = path.read_text(errors="replace")
    bad = re.findall(
        r"Overfull|undefined citations?|multiply defined|LaTeX Error",
        text,
        flags=re.IGNORECASE,
    )
    if bad:
        raise RuntimeError(f"{path.name} contains LaTeX diagnostics: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=ROOT.parent / "gogogo_submission_package.tar.gz",
    )
    args = parser.parse_args()
    archive = args.archive.resolve()

    packaged, members = archive_files(archive)
    released = release_files()
    missing = sorted(released - packaged)
    extra = sorted(packaged - released)
    if missing or extra:
        raise RuntimeError(
            f"archive manifest mismatch; missing={missing}, extra={extra}"
        )

    with tempfile.TemporaryDirectory(prefix="gogogo-package-audit-") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp_path, members=members, filter="data")
        work = tmp_path / "GOGOGO"

        audit = run(["python3", "verify_paper_numbers.py"], work)
        m = re.search(r"(\d+)/(\d+) checks passed, 0 failed", audit)
        if not m or m.group(1) != m.group(2):
            raise RuntimeError("paper-number audit did not pass cleanly")
        n_checks = m.group(0)

        before = {name: digest(work / "figures" / name) for name in FIGURES}
        run(["python3", "make_figures.py"], work)
        after = {name: digest(work / "figures" / name) for name in FIGURES}
        changed = [name for name in FIGURES if before[name] != after[name]]
        if changed:
            raise RuntimeError(f"regenerated figures differ: {changed}")

        run(["latexmk", "-pdf", "-gg", "-silent", "main.tex"], work)
        run(["latexmk", "-pdf", "-gg", "-silent", "si.tex"], work)
        check_latex_log(work / "main.log")
        check_latex_log(work / "si.log")

    print(
        f"PASS: {len(packaged)} files; {n_checks}; "
        f"{len(FIGURES)}/{len(FIGURES)} figures reproduced; PDFs rebuilt"
    )


if __name__ == "__main__":
    main()
