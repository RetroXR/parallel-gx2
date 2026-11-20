// What FetchShader.cpp takes from the rest of the core
#include <string_view>

#include "Cemu/Logging/CemuLogging.h"
#include "Cafe/HW/Latte/Core/Latte.h"

uint64 s_loggingFlagMask = 0;

bool cemuLog_log(LogType type, std::string_view text)
{
	return false;
}

void LatteSHRC_RemoveShaderStatesUsing(const void* shader)
{
}

// the fetch shader for the GPU's state (FindByGPUState) is not looked up here
LatteGPUState_t LatteGPUState = {};

uint8* memory_getPointerFromPhysicalOffset(uint32 physicalOffset)
{
	return nullptr;
}

// lrc_hash.c's file hashing and SIMD dispatch, not reached from here
#include <streams/file_stream.h>
#include <features/features_cpu.h>
extern "C" {
RFILE* filestream_open(const char*, unsigned, unsigned) { return nullptr; }
int64_t filestream_read(RFILE*, void*, int64_t) { return -1; }
int64_t filestream_view(RFILE*, int64_t, const uint8_t**) { return -1; }
int filestream_close(RFILE*) { return -1; }
uint64_t cpu_features_get(void) { return 0; }
}

#include "../test_renderer.h"
