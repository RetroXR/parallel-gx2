#!/bin/sh
# NFC version 0 tags read and written back the same. RUNNER prefixes the test (wine for a MinGW build).
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 120 $RUN"
$RUN ./nfc_tag_test${EXE:-}
