"""Create a submission ZIP without raw data or large model artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import zipfile
from pathlib import Path


EXCLUDE_PATTERNS = (
    "Agents.md",
    ".git/*",
    "__pycache__/*",
    "*.pyc",
    ".pytest_cache/*",
    ".venv/*",
    "venv/*",
    "env/*",
    "data/*",
    "datasets/*",
    "raw_data/*",
    "checkpoints/*",
    "models/*",
    "outputs/*",
    "runs/*",
    "logs/*",
    "*.keras",
    "*.h5",
    "*.ckpt",
    "*.pt",
    "*.pth",
    "*.onnx",
    "dist/*",
    "*.zip",
    ".DS_Store",
    "Thumbs.db",
    ".vscode/*",
    ".idea/*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a course project submission ZIP.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument(
        "--output",
        default="dist/chestxray8-keras-course-project.zip",
        help="Output ZIP path.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra file or directory to include, e.g. a final report. Can be repeated.",
    )
    return parser.parse_args()


def normalize(path: Path) -> str:
    value = path.as_posix()
    if value.startswith("./"):
        return value[2:]
    return value


def is_excluded(relative_path: str) -> bool:
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in EXCLUDE_PATTERNS)


def git_tracked_files(project_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return [project_root / line.strip() for line in result.stdout.splitlines() if line.strip()]


def collect_extra_files(project_root: Path, includes: list[str]) -> list[Path]:
    files: list[Path] = []
    for include in includes:
        path = (project_root / include).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Extra include path does not exist: {include}")
        if path.is_file():
            files.append(path)
        else:
            files.extend(child for child in path.rglob("*") if child.is_file())
    return files


def collect_submission_files(project_root: Path, includes: list[str] | None = None) -> list[Path]:
    includes = includes or []
    candidates = git_tracked_files(project_root) + collect_extra_files(project_root, includes)

    unique: dict[str, Path] = {}
    for path in candidates:
        relative = normalize(path.resolve().relative_to(project_root.resolve()))
        if not is_excluded(relative):
            unique[relative] = path
    return [unique[key] for key in sorted(unique)]


def build_zip(project_root: Path, output_path: Path, includes: list[str] | None = None) -> list[str]:
    files = collect_submission_files(project_root, includes)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    archived: list[str] = []
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            relative = normalize(path.resolve().relative_to(project_root.resolve()))
            zf.write(path, relative)
            archived.append(relative)
        zf.writestr("SUBMISSION_MANIFEST.txt", "\n".join(archived) + "\n")
        archived.append("SUBMISSION_MANIFEST.txt")
    return archived


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = (project_root / args.output).resolve()
    archived = build_zip(project_root, output_path, args.include)
    print(f"created: {output_path}")
    print(f"files: {len(archived)}")


if __name__ == "__main__":
    main()
