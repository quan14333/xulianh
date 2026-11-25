#!/usr/bin/env bash
set -euo pipefail
REPO="${1:?Usage: bash scripts/PUSH_LOCBE.sh /path/to/clone-of-xulianh}"
ORIGIN=$(git -C "$REPO" remote get-url origin)
[[ "$ORIGIN" == *"quan14333/xulianh"* ]] || { echo "Wrong remote: $ORIGIN" >&2; exit 1; }
git -C "$REPO" fetch origin main
git -C "$REPO" switch main
git -C "$REPO" pull --ff-only origin main
if git -C "$REPO" show-ref --verify --quiet refs/heads/LocBe; then
 git -C "$REPO" switch LocBe
else
 git -C "$REPO" switch -c LocBe
fi
mkdir -p "$REPO/research/LocBe"
cp -R "$(cd "$(dirname "$0")/.." && pwd)/". "$REPO/research/LocBe/"
git -C "$REPO" add research/LocBe
git -C "$REPO" commit -m 'docs: chot phuong phap va ban giao LocBe DCTR-C3T' || true
git -C "$REPO" push -u origin LocBe
printf 'PUSHED: https://github.com/quan14333/xulianh/tree/LocBe\n'
