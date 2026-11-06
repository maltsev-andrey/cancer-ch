#!/usr/bin/env bash
# Gate before making the repository public: no internal paths, hosts, addresses or users.
# Usage: scripts/sanitize_check.sh [repo-root]      exit 1 if anything is found.
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

PATTERNS=(
  '/nfs/'                       # NFS mounts
  'srv-[a-z]+-[0-9]+'           # internal host names
  '\b(10|170|172|192)\.[0-9]+\.[0-9]+\.[0-9]+\b'   # private / internal IPv4
  'ansible@'                    # user@host prompts
  '/home/[a-z]+/'               # home directories
  'miniforge3/etc'              # local tool paths
  'wiki\.[a-z]+/books'          # internal wiki
  'BEGIN (RSA|OPENSSH) PRIVATE' # keys
  'password|passwd|secret_key|api_key|token='
)

EXCLUDE='--exclude-dir=.git --exclude-dir=data --exclude-dir=wiki_bookstack --exclude=sanitize_check.sh'
fail=0
for p in "${PATTERNS[@]}"; do
  if hits=$(grep -rnIiE $EXCLUDE "$p" . 2>/dev/null); then
    echo "== pattern: $p"; echo "$hits" | head -20; fail=1
  fi
done

# files that must not be committed at all
for f in $(git ls-files 2>/dev/null | grep -E '\.(parquet|nc|log|xlsx|csv)$|^data/' || true); do
  echo "== tracked data file: $f"; fail=1
done

if [ "$fail" -eq 0 ]; then echo "sanitize: clean"; else echo "sanitize: FAILED"; fi
exit $fail
