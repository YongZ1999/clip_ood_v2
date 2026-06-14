#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


FORBIDDEN_PATTERNS = (
    r"\bstate-of-the-art\b",
    r"\bstate of the art\b",
    r"\bSOTA\b",
    r"\bBayes-optimal\b",
    r"\bBayes[- ]?Optimal(?:ity)?\b",
    r"\badaptive router\b",
    r"\bnear-zero (?:OOD )?confidence\b",
    r"\bBayesian Ensemble Classifiers\b",
    r"\bco-optimizes?\b",
    r"\bco-optimization\b",
    r"\bholistic framework\b",
    r"\bis a universal replacement for LADA\b",
    r"(?<!not )as a universal replacement for LADA",
    r"\bsurpass official LADA DPT\b",
)

REQUIRED_PATTERNS = (
    r"classifier\\_feature\\_transform=test",
    r"not as a universal replacement for LADA",
    r"LADA remains stronger",
    r"not as a claim that compact GMM classifier rebuilding is equivalent to official LADA DPT",
    r"not the final classifier parameters",
)

README_REQUIRED_PATTERNS = (
    r"--classifier_feature_transform test",
    r"not a universal replacement for LADA",
    r"LADA is stronger",
    r"not official LADA DPT reproductions",
    r"excluding final classifier parameters",
)

NEGATED_CLAIM_CONTEXT = (
    "do not claim",
    "should not say",
    "should not claim",
    "not as a claim",
    "not a universal",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        type=Path,
        default=Path("paper_writing/paper-template/paper_draft.tex"),
        nargs="*",
    )
    return parser.parse_args()


def visible_lines(lines):
    hidden_depth = 0
    visible = []
    hidden = []
    structure_errors = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith(r"\iffalse"):
            hidden_depth += 1
            hidden.append((lineno, line))
            continue
        if hidden_depth:
            hidden.append((lineno, line))
            if stripped.startswith(r"\fi"):
                hidden_depth -= 1
            continue
        if stripped.startswith(r"\fi"):
            structure_errors.append((lineno, line.rstrip()))
        visible.append((lineno, line))
    if hidden_depth:
        structure_errors.append((len(lines), f"unclosed \\\\iffalse block depth: {hidden_depth}"))
    return visible, hidden, structure_errors


def uncomment(line):
    if line.lstrip().startswith("%"):
        return ""
    return line.split("%", 1)[0]


def find_pattern(lines, pattern, flags=re.IGNORECASE):
    regex = re.compile(pattern, flags)
    return [(lineno, line.rstrip()) for lineno, line in lines if regex.search(uncomment(line))]


def required_patterns_for(path):
    if path.suffix == ".md":
        return README_REQUIRED_PATTERNS
    return REQUIRED_PATTERNS


def is_negated_claim_context(line):
    lower = line.lower()
    return any(marker in lower for marker in NEGATED_CLAIM_CONTEXT)


def find_forbidden_pattern(lines, pattern, flags=re.IGNORECASE):
    regex = re.compile(pattern, flags)
    matches = []
    for lineno, line in lines:
        visible_line = uncomment(line)
        if regex.search(visible_line) and not is_negated_claim_context(visible_line):
            matches.append((lineno, line.rstrip()))
    return matches


def audit_path(path):
    lines = path.read_text().splitlines()
    visible, hidden, structure_errors = visible_lines(lines)
    visible_text = "\n".join(uncomment(line) for _, line in visible)

    failures = []

    if structure_errors:
        failures.append(("malformed hidden block structure", structure_errors))

    visible_todos = find_pattern(visible, r"\\todo\{")
    if visible_todos:
        failures.append(("visible TODO commands", visible_todos))

    for pattern in FORBIDDEN_PATTERNS:
        matches = find_forbidden_pattern(visible, pattern)
        if matches:
            failures.append((f"forbidden pattern: {pattern}", matches))

    missing_required = []
    for pattern in required_patterns_for(path):
        if not re.search(pattern, visible_text, flags=re.IGNORECASE):
            missing_required.append(pattern)
    if missing_required:
        failures.append(("missing required boundary text", [(0, item) for item in missing_required]))

    hidden_todos = find_pattern(hidden, r"\\todo\{")
    print(f"audited: {path}")
    print(f"visible lines: {len(visible)}")
    print(f"hidden lines ignored: {len(hidden)}")
    print(f"hidden TODO commands ignored: {len(hidden_todos)}")

    if failures:
        print("CLAIM AUDIT FAILED")
        for title, matches in failures:
            print(f"\n[{title}]")
            for lineno, line in matches:
                location = f"line {lineno}" if lineno else "required"
                print(f"- {location}: {line}")
        return False

    print("CLAIM AUDIT PASSED")
    return True


def main():
    args = parse_args()
    paths = args.paths or [Path("paper_writing/paper-template/paper_draft.tex")]
    ok = True
    for path in paths:
        ok = audit_path(path) and ok
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
