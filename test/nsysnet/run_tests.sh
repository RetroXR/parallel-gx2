#!/bin/sh
# The guest's sockets: handles made and closed in place, a full table refused. RUNNER prefixes the test (wine for a MinGW build).
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 120 $RUN"
$RUN ./nsysnet_test${EXE:-}
