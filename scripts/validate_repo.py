"""Validate the research scaffold without downloading data or model artifacts."""

from __future__ import annotations

import sys
import re
from pathlib import Path
from urllib.parse import unquote

import yaml


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = (
    ".github/workflows",
    ".github/ISSUE_TEMPLATE",
    "configs/baseline",
    "configs/evaluation",
    "data/raw/FMB",
    "data/processed",
    "data/manifests",
    "data/audit",
    "data/examples",
    "external",
    "checkpoints/pretrained",
    "checkpoints/experiments",
    "notebooks/quan",
    "notebooks/man",
    "notebooks/loc_be",
    "notebooks/loc_lon",
    "src/data",
    "src/baseline",
    "src/evaluation",
    "src/analysis",
    "src/methods",
    "runs",
    "outputs",
    "docs/problem_cards",
    "docs/meetings",
    "reports/weekly",
    "reports/experiments",
    "reports/tables",
    "reports/figures",
    "reports/final",
    "reports/slides",
)

REQUIRED_FILES = (
    "README.md",
    ".gitignore",
    "requirements.txt",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/task.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/experiment.md",
    "configs/baseline/fmb_student.yaml",
    "configs/evaluation/fusion_metrics.yaml",
    "docs/PROJECT_OVERVIEW.md",
    "docs/TEAM_RULES.md",
    "docs/GIT_WORKFLOW.md",
    "docs/OUTPUT_CONTRACT.md",
    "docs/DATA_POLICY.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/RESEARCH_GAPS.md",
    "docs/TEAM_TASKS.md",
    "docs/MEETING_SCHEDULE.md",
    "docs/problem_cards/TEMPLATE.md",
    "docs/problem_cards/quan.md",
    "docs/problem_cards/man.md",
    "docs/problem_cards/loc_be.md",
    "docs/problem_cards/loc_lon.md",
    "docs/meetings/TEMPLATE.md",
)

YAML_FILES = (
    "configs/baseline/fmb_student.yaml",
    "configs/evaluation/fusion_metrics.yaml",
    ".github/workflows/ci.yml",
)

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def validate_relative_markdown_links(errors: list[str]) -> None:
    """Catch broken local links while ignoring URLs, anchors, and template placeholders."""
    for markdown_file in ROOT.rglob("*.md"):
        text = markdown_file.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_without_anchor = unquote(target.split("#", maxsplit=1)[0])
            if not target_without_anchor or "PLACEHOLDER" in target_without_anchor:
                continue
            resolved = (markdown_file.parent / target_without_anchor).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(
                    f"Markdown link escapes repository: "
                    f"{markdown_file.relative_to(ROOT)} -> {target}"
                )
                continue
            if not resolved.exists():
                errors.append(
                    f"Broken relative Markdown link: "
                    f"{markdown_file.relative_to(ROOT)} -> {target}"
                )


def main() -> int:
    errors: list[str] = []

    for relative in REQUIRED_DIRS:
        path = ROOT / relative
        if not path.is_dir():
            errors.append(f"Missing directory: {relative}")

    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing file: {relative}")

    for relative in YAML_FILES:
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                errors.append(f"YAML root must be a mapping: {relative}")
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            errors.append(f"Invalid YAML {relative}: {exc}")

    for relative in (*REQUIRED_DIRS, *REQUIRED_FILES, *YAML_FILES):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"Unsafe relative path in validator: {relative}")

    validate_relative_markdown_links(errors)

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Repository validation passed: "
        f"{len(REQUIRED_DIRS)} directories, {len(REQUIRED_FILES)} files, "
        f"{len(YAML_FILES)} YAML files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
