import hashlib
import subprocess
from pathlib import Path

from pydantic import TypeAdapter

from assistant.config import PROJECT_ROOT
from evals.run_cases import CaseRun

CACHE_DIR = PROJECT_ROOT / "evals" / "results" / "cache"
CASE_RUNS = TypeAdapter(list[CaseRun])
# Eval outputs change on every run and must not make the code look edited.
GIT_PATHS_OUTSIDE_EVAL_OUTPUT = [".", ":(exclude)evals/results"]


def git_output(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, check=True
    ).stdout


def code_fingerprint() -> str:
    """The commit, plus a hash of uncommitted and untracked changes when the tree is dirty."""
    commit = git_output("rev-parse", "--short=12", "HEAD").decode().strip()
    tracked_changes = git_output("diff", "HEAD", "--", *GIT_PATHS_OUTSIDE_EVAL_OUTPUT)
    untracked_paths = git_output(
        "ls-files", "--others", "--exclude-standard", "--", *GIT_PATHS_OUTSIDE_EVAL_OUTPUT
    ).split()
    untracked_contents = b"".join(
        path + (PROJECT_ROOT / path.decode()).read_bytes() for path in sorted(untracked_paths)
    )
    if not tracked_changes and not untracked_contents:
        return commit
    # Uncommitted edits are part of the key, so editing code never reuses outputs of older code.
    changes_hash = hashlib.sha256(tracked_changes + untracked_contents).hexdigest()[:12]
    return f"{commit}-dirty-{changes_hash}"


def cache_path(golden_set_fingerprint: str, code_version: str) -> Path:
    return CACHE_DIR / f"{code_version}-golden-{golden_set_fingerprint[:12]}.json"


def load_cached_runs(path: Path) -> dict[str, CaseRun]:
    if not path.exists():
        return {}
    return {run.case.id: run for run in CASE_RUNS.validate_json(path.read_bytes())}


def save_cached_runs(path: Path, runs: dict[str, CaseRun]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(CASE_RUNS.dump_json(list(runs.values()), indent=2))
