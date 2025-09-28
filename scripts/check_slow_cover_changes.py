import argparse
import subprocess
import sys
from pathlib import Path
from typing import Set

import coverage
import re


def get_changed_files(base_sha: str, src_dir: str = "src") -> list[str]:
    """Get list of changed Python files in src directory."""
    try:
        if base_sha == "HEAD":
            # Compare staged changes (index) vs HEAD
            cmd = ["git", "diff", "--name-only", "--cached"]
        elif base_sha == "WORKING":
            # Compare working directory vs index (unstaged changes)
            cmd = ["git", "diff", "--name-only"]
        else:
            # Compare specific commit/branch vs HEAD
            cmd = ["git", "diff", "--name-only", base_sha, "HEAD"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        all_changed = result.stdout.strip().split("\n") if result.stdout.strip() else []

        # Filter for Python files in the src directory
        python_files = []
        for file in all_changed:
            if file and file.startswith(src_dir) and file.endswith(".py") and Path(file).exists():
                python_files.append(file)

        return python_files
    except subprocess.CalledProcessError:
        return []


def get_changed_lines(filename: str, base_sha: str) -> Set[int]:
    """Get line numbers that were changed in a file using git diff."""
    try:
        # Use -U0 for no context, making parsing simpler
        if base_sha == "HEAD":
            # Compare staged changes (index) vs HEAD
            cmd = ["git", "diff", "-U0", "--cached", "--", filename]
        elif base_sha == "WORKING":
            # Compare working directory vs index (unstaged changes)
            cmd = ["git", "diff", "-U0", "--", filename]
        else:
            # Compare specific commit/branch vs HEAD
            cmd = ["git", "diff", "-U0", base_sha, "HEAD", "--", filename]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        changed_lines: set = set()
        for line in result.stdout.split("\n"):
            if line.startswith("@@"):
                # Parse both old and new line ranges
                # @@ -old_start,old_count +new_start,new_count @@
                match = re.search(r"@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@", line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1
                    changed_lines.update(range(old_start, old_start + old_count))
                    changed_lines.update(range(new_start, new_start + new_count))
        return changed_lines
    except subprocess.CalledProcessError:
        return set()


def get_slow_cover_excluded_lines(filename: str) -> Set[int]:
    """Get line numbers excluded by 'pragma: slow-cover' using coverage."""
    cov = coverage.Coverage()
    # Clear exclusions and add only slow-cover
    cov.exclude(r"pragma: slow-cover", which="exclude")
    # Analyze the file to get excluded lines
    _, _, excluded_lines, _, _ = cov.analysis2(filename)
    return set(excluded_lines)


def should_run_slow_tests(base_sha: str, src_dir: str = "src") -> bool:
    """
    Determine if slow tests should be run based on changes to marked slow-cover sections.
    """

    if base_sha == "HEAD":
        print(f"Checking for staged changes in {src_dir}/")
    elif base_sha == "WORKING":
        print(f"Checking for unstaged changes in {src_dir}/")
    else:
        print(f"Checking for changes between {base_sha} and HEAD in {src_dir}/")

    changed_files = get_changed_files(base_sha, src_dir)
    print(f"Found {len(changed_files)} changed Python files: {changed_files}")

    if not changed_files:
        print("No Python files changed")
        return False

    for filename in changed_files:
        print(f"Analyzing changes in {filename}...")

        # Get lines that were changed
        changed_lines = get_changed_lines(filename, base_sha)
        print(f"  Changed lines: {sorted(changed_lines) if changed_lines else 'No changed lines'}")

        if not changed_lines:
            continue

        # Get lines excluded by slow-cover pragma
        slow_cover_lines = get_slow_cover_excluded_lines(filename)
        print(f"  Slow-cover lines: {sorted(slow_cover_lines) if slow_cover_lines else 'Not excluded lines'}")

        if not slow_cover_lines:
            continue

        # Check if any changed lines intersect with slow-cover lines
        intersection = changed_lines & slow_cover_lines
        if intersection:
            print(f"Changes in {filename} affect slow-cover lines: {sorted(intersection)}")
            return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_sha",
        help="Base SHA to compare against. Use 'HEAD' for staged changes, 'WORKING' for unstaged changes, or a commit SHA like 'HEAD~1'",
    )
    parser.add_argument("--src-dir", default="src", help="Source directory to check (default: src)")

    args = parser.parse_args()

    if should_run_slow_tests(args.base_sha, args.src_dir):
        print("Will run all tests (including slow ones)")
        sys.exit(0)  # Run slow tests
    else:
        print("Will skip slow tests")
        sys.exit(1)  # Skip slow tests


if __name__ == "__main__":
    main()
