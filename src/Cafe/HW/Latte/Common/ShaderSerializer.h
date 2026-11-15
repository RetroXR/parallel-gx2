#pragma once
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
};