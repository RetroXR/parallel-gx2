#!/bin/sh
# Objects shared by key, made once by whichever thread asks first. RUNNER prefixes the test (wine for a MinGW build).
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 120 $RUN"
$RUN ./shared_by_key_test${EXE:-}
