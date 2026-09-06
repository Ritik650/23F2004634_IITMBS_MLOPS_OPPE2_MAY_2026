#!/usr/bin/env bash
# stress_test.sh
# Deliverable 6: Performance monitoring & stress testing with wrk.
#
# Requires `wrk` installed (apt-get install wrk, or build from source at
# https://github.com/wg/wrk). Run from a machine that can reach the
# service's external IP (e.g. the Cloud Shell / Compute Engine VM you
# used to hit kubectl).
#
# Usage:
#   ./stress_test.sh http://<EXTERNAL_IP> [duration] [threads] [connections]
#
# Example (>2000 concurrent connections as required by the deliverable):
#   ./stress_test.sh http://34.123.45.67 30s 12 2500

set -euo pipefail

HOST="${1:?Usage: $0 <host, e.g. http://EXTERNAL_IP> [duration] [threads] [connections]}"
DURATION="${2:-30s}"
THREADS="${3:-12}"
CONNECTIONS="${4:-2500}"

OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="${OUT_DIR}/wrk_${STAMP}.txt"

echo "Running: wrk -t${THREADS} -c${CONNECTIONS} -d${DURATION} -s post.lua --timeout 30s ${HOST}/predict"
wrk -t"${THREADS}" -c"${CONNECTIONS}" -d"${DURATION}" \
    -s "$(dirname "$0")/post.lua" \
    --timeout 30s \
    "${HOST}/predict" | tee "${OUT_FILE}"

echo ""
echo "Results saved to: ${OUT_FILE}"
echo "Capture throughput (Req/Sec), latency distribution, and any Socket"
echo "errors / timeouts from the output above into the README's stress"
echo "test section, then (optionally) also grab GKE HPA scaling events:"
echo "  kubectl get hpa heart-disease-api-hpa --watch"
echo "  kubectl get pods -l app=heart-disease-api --watch"
