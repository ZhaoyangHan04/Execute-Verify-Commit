"""Explicitly download a public benchmark; excluded from the repository/archive."""

import argparse
import subprocess

from .config import ROOT, load_config


def main():
    benchmarks = load_config()["benchmarks"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=benchmarks)
    args = parser.parse_args()
    spec = benchmarks[args.benchmark]
    destination = ROOT / spec["directory"]
    if destination.exists():
        parser.error("Destination exists; it will not be overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            spec["source"],
            str(destination),
        ],
        check=True,
    )

    def git(*args):
        subprocess.run(["git", "-C", str(destination), *args], check=True)

    git("checkout", "--detach", spec["revision"])
    if spec.get("patch"):
        git("apply", "--check", str(ROOT / spec["patch"]))
        git("apply", str(ROOT / spec["patch"]))
        git("add", "--update")
        # A fresh anonymous patch commit keeps the upstream integrity checks active.
        git(
            "-c",
            "user.name=Anonymous",
            "-c",
            "user.email=anonymous@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "Apply benchmark correctness fix",
        )
    print(
        f"Install the benchmark dependencies: python -m pip install -e {spec['directory']}"
    )


if __name__ == "__main__":
    main()
