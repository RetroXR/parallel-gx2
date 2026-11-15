#!/bin/sh
# The fibers' test. FIBER=unix or libco picks the backend (MinGW: Windows'
# own); RUNNER prefixes the test (wine for a MinGW build).
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 300 $RUN"
$RUN ./fiber_test${EXE:-}
