#!/bin/sh
# The SPIR-V writer. RUNNER prefixes the test (wine for a MinGW build); the
# module it writes goes through spirv-val where that is installed.
set -e
cd "$(dirname "$0")"
make -s
RUN=${RUNNER:-}
command -v timeout >/dev/null 2>&1 && RUN="timeout 300 $RUN"
$RUN ./spirv_writer_test${EXE:-} module.spv
if command -v spirv-val >/dev/null 2>&1; then
	spirv-val --target-env vulkan1.0 module.spv
	echo "spirv-val: module.spv valid"
fi
rm -f module.spv
