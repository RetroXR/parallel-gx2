#pragma once

#include <cstdint>

// The keys a draw's pixel and geometry shaders are found by
// (LatteSHRC_UpdateActiveShaders) and decompiled ahead under
// (LatteShaderAhead.cpp), and how a draw finds those a shader cache holds
// under the keys of before: a geometry shader's took the ring of the vertex
// shader active before the draw's, a pixel shader's after a geometry shader
// the program of the geometry shader hashed before the draw's.

// a pixel shader's, after a geometry shader that of its program's first hash
inline uint64_t LatteSHRC_PixelBaseHash(uint64_t psHash1, uint64_t psHash2, uint64_t psImportKey, bool usesGeometryShader, uint64_t gsHash1)
{
	return psHash1 + psHash2 + psImportKey + (usesGeometryShader ? gsHash1 : 0ULL);
}

// a geometry shader's, with the ring its vertex shader writes
inline uint64_t LatteSHRC_GeometryBaseHash(uint64_t gsHash1, uint64_t gsHash2, uint32_t vsRingParameterCount, bool streamout)
{
	return gsHash1 + gsHash2 + (uint64_t)vsRingParameterCount + (streamout ? 21 : 0);
}

// The draw's pixel shader by its key; else, after a geometry shader, by the
// key that took the one hashed before (the same shader: no geometry shader
// reaches a pixel shader's decompile). find(baseHash): the shader, or null.
template<typename Shader, typename Find>
Shader* LatteSHRC_FindPixelShaderFor(Find&& find, uint64_t psHash1, uint64_t psHash2, uint64_t psImportKey, bool usesGeometryShader,
	uint64_t gsHash1, uint64_t gsHash1Before)
{
	Shader* shader = find(LatteSHRC_PixelBaseHash(psHash1, psHash2, psImportKey, usesGeometryShader, gsHash1));
	if (!shader && usesGeometryShader && gsHash1Before != gsHash1)
		shader = find(LatteSHRC_PixelBaseHash(psHash1, psHash2, psImportKey, true, gsHash1Before));
	return shader;
}

// The draw's geometry shader by its key; else by the key that took the ring
// of the vertex shader active before (previousRing, none: negative), where
// it was made for this ring (its ringParameterCountFromPrevStage)
template<typename Shader, typename Find>
Shader* LatteSHRC_FindGeometryShaderFor(Find&& find, uint64_t gsHash1, uint64_t gsHash2, uint32_t vsRingParameterCount, int64_t previousRing, bool streamout)
{
	Shader* shader = find(LatteSHRC_GeometryBaseHash(gsHash1, gsHash2, vsRingParameterCount, streamout));
	if (!shader && previousRing >= 0 && (uint32_t)previousRing != vsRingParameterCount)
	{
		Shader* before = find(LatteSHRC_GeometryBaseHash(gsHash1, gsHash2, (uint32_t)previousRing, streamout));
		if (before && before->ringParameterCountFromPrevStage == vsRingParameterCount)
			shader = before;
	}
	return shader;
}
