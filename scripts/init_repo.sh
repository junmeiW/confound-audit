#!/usr/bin/env bash
#
# init_repo.sh — initialise the confound-audit repository and make the first commit.
#
# Safe by design:
#   * refuses to run if a .git already exists
#   * never pushes and never invents credentials
#   * prints exactly what will be committed and checks for build junk first
#   * supports --dry-run to review without changing anything
#
# Usage:
#   ./scripts/init_repo.sh --dry-run
#   ./scripts/init_repo.sh
#
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# --- locate the package root (this script lives in <root>/scripts) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "package root: $ROOT"
echo

# --- preflight -------------------------------------------------------------
for f in pyproject.toml setup.cfg README.md LICENSE; do
  if [ ! -f "$f" ]; then
    echo "ERROR: expected $f — is this the package root?" >&2
    exit 1
  fi
done

# Repository may already exist but be empty (e.g. a previous run stopped at the
# identity check). Proceed in that case; only refuse if it already has commits.
if [ -e .git ] && git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "ERROR: this repository already has commits — refusing to re-initialise." >&2
  echo "       Delete .git manually if you really mean to start over." >&2
  exit 1
fi

# Initialise first, so that a repository-LOCAL identity can work as well as a
# global one. (Checking identity before 'git init' would make local config
# impossible to use — a chicken-and-egg problem.)
CREATED_GIT=0
if [ ! -e .git ]; then
  git init -q -b main
  CREATED_GIT=1
  echo "initialised empty repository"
fi

# Identity: a placeholder email makes the commit unattributable, and GitHub
# will not link it to an account. Environment variables are accepted too.
NAME="${GIT_AUTHOR_NAME:-$(git config user.name  || true)}"
EMAIL="${GIT_AUTHOR_EMAIL:-$(git config user.email || true)}"
if [ -z "$NAME" ] || [ -z "$EMAIL" ] || [ "$EMAIL" = "yourgitemail@example.com" ]; then
  cat >&2 <<EOF

ERROR: git identity is not set (or is still the placeholder).

  current user.name : ${NAME:-(unset)}
  current user.email: ${EMAIL:-(unset)}

The repository has been initialised but nothing was committed.
Set an identity and re-run this script:

  git config user.name  "Your Name"     # this repository only
  git config user.email "you@example.com"

  # ...or globally, for all repositories:
  git config --global user.name  "Your Name"
  git config --global user.email "you@example.com"

EOF
  exit 1
fi
echo "author: $NAME <$EMAIL>"
echo

# --- stage ----------------------------------------------------------------
git add -A

echo "=== files staged for the first commit ==="
git diff --cached --name-only | sort
COUNT="$(git diff --cached --name-only | wc -l | tr -d ' ')"
echo "  -> $COUNT files"
echo

# --- guard: nothing unwanted should be staged ------------------------------
PROBLEMS=0
check_absent() {
  if git diff --cached --name-only | grep -qE "$1"; then
    echo "ERROR: staged file matching '$1' — these must not be committed:" >&2
    git diff --cached --name-only | grep -E "$1" >&2
    PROBLEMS=1
  fi
}
check_absent '^dist/|^build/'
check_absent '__pycache__|\.pyc$'
check_absent '\.egg-info/'
check_absent '\.pytest_cache/'
check_absent '\.env$|\.env\.'
check_absent '\.(pt|bin|safetensors|npz|zip)$'

if [ "$PROBLEMS" -ne 0 ]; then
  echo >&2
  echo "Fix .gitignore (or unstage with: git rm -r --cached <path>) and retry." >&2
  exit 1
fi
echo "guard: no build artifacts, caches, secrets or large binaries staged"
echo

# --- sanity: the tests should pass before the first commit -----------------
if command -v python3 >/dev/null 2>&1; then
  echo "=== running the test suite before committing ==="
  if python3 -m pytest -q 2>&1 | tail -3; then
    echo "tests passed"
  else
    echo "ERROR: tests failed — fix them before committing." >&2
    exit 1
  fi
  echo
fi

# --- commit ---------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: stopping before the commit; nothing was written."
  if [ "$CREATED_GIT" -eq 1 ]; then
    rm -rf .git
    echo "         (the temporary .git created for this dry run was removed)"
  fi
  exit 0
fi

git commit -q -m "Initial release of confound-audit 0.1.0

Detect input-composition confounds in mutation-induced embedding analyses
before interpreting them as model findings.

Implements the three checks recommended by the accompanying analysis:
  1. zero-parameter baselines (one-hot, replacement-pair, random lookup)
  2. stratified permutation nulls (within assay / position / wild-type)
  3. strict substitution-type matching, with a guard against reporting a
     constructive zero as a finding
plus protein-clustered inference.

No GPU, no model weights, no particular dataset: it consumes displacement
vectors the user already has.

70 tests, including regression checks that reproduce the published
reference statistic (+0.2170)."

echo "=== commit created ==="
git log --stat --oneline -1 | head -35
echo
echo "Next: create the GitHub repository, then push (see SETUP_GITHUB.md)."
