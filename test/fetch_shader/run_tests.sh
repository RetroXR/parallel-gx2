#!/bin/sh
# Fetch shaders parsed: groups and attributes in place, none allocated. RUNNER prefixes the test (wine for a MinGW build).
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 300 $RUN"
$RUN ./fetch_shader_test${EXE:-}
