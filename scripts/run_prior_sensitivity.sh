#!/usr/bin/env bash
# Prior-sensitivity fits for s10: 4 key fits x 3 prior variants = 12 fits, ~5 min at -P 4.
# Baseline fits must already exist (main sweep). Run from the repo root inside the cancer-ch env.
set -euo pipefail
cd "$(dirname "$0")/.."
P="${P:-4}"

FITS=(
  "Lunge, Bronchien, Luftröhre|F|incidence"
  "Hautmelanom|F|incidence"
  "Prostata|M|incidence"
  "Alle Krebsarten (total)|M|incidence"
)
# tag|drift_sd|rw_sd|eps_sd   (baseline is 0.5 / 0.5 / 0.1)
VARIANTS=(
  "tight|0.15|0.15|0.03"
  "wide|1.5|1.5|0.3"
  "curvy|0.15|1.5|0.1"
)

jobs=()
for f in "${FITS[@]}"; do
  IFS='|' read -r site sex ind <<< "$f"
  for v in "${VARIANTS[@]}"; do
    IFS='|' read -r tag dsd rsd esd <<< "$v"
    jobs+=("python pipeline/s06_apc.py --site $(printf %q "$site") --sex $sex --indicator $ind --future 2 --tag $tag --drift-sd $dsd --rw-sd $rsd --eps-sd $esd 2>&1 | grep -E '^\[.*(OK|CHECK|FAILED)'")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -d '\n' -P "$P" -I{} sh -c '{}'
python pipeline/s10_prior_sensitivity.py
