#pragma once
#include "Cafe/HW/Latte/Common/RegisterSerializer.h"
#include "util/helpers/Serializer.h"

namespace Latte
{
	// the dictionary the entries are compressed with; before the first
	// (de)serialize, once
	void InitShaderSerializer();
	void SerializeShaderProgram(void* shaderProg, uint32 size, MemStreamWriter& memWriter);
	bool DeserializeShaderProgram(std::vector<uint8>& progData, MemStreamReader& memReader);
	// The calling thread's own buffers to inflate an entry's programs into
	// (slot 0 to 2): kept, so that once they are as big as the biggest
	// program, reading the next entry allocates nothing for them
	std::vector<uint8>& ShaderProgramScratch(int slot);

	// A separable shader's cache entry as the GPU thread takes it when a new
	// shader is made (LatteShaderCache_write*): what it knows, the registers
	// compacted, then its programs as they are, programCount of them, after
	// this header. BuildShaderCacheEntry serializes and compresses it - the
	// registers at zstd level 9, the programs at 6 - into the entry the
	// transferable cache holds, malloc'd, on the cache's writer thread
	// (file_cache_add_built), not the GPU thread. The type in the top four
	// bits of versionAndType (1 geometry: vsRingParameterCount, the others:
	// usesGeometryShader).
	struct ShaderCacheEntryInputs
	{
		uint8 versionAndType;
		uint8 usesGeometryShader;
		uint16 vsRingParameterCount;
		uint32 programCount;
		uint32 programSize[2];
		uint64 baseHash;
		uint64 auxHash;
		GPUCompactedRegisterState regs;
	};
	void* BuildShaderCacheEntry(const void* inputs, size_t size, size_t* outSize);
};