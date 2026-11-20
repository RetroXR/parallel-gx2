#pragma once
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "util/helpers/PoolNew.h"
#include <boost/container/static_vector.hpp>

struct LatteParsedFetchShaderAttribute
{
	uint8								attributeBufferIndex;
	uint8								semanticId;
	Latte::E_HWFMT						format;
	LatteConst::VertexFetchType2		fetchType;
	uint8								nfa;
	uint8								isSigned;
	LatteConst::VertexFetchEndianMode	endianSwap;
	uint8								ds[4]; // destination component select
	sint32								aluDivisor;
	uint32								offset;
};

struct LatteParsedFetchShaderBufferGroup
{
	uint8 attributeBufferIndex{}; // index of buffer (0 to 15 are valid)
	sint8 attribCount{};
	bool hasVtxIndexAccess : 1;
	bool hasInstanceIndexAccess : 1;
	uint32 minOffset{};
	uint32 totalAttribRangeSize{}; // max attribOffset+attribSize
	LatteParsedFetchShaderAttribute* attrib{}; // attributes for this buffer, in its fetch shader's m_attribs

	uint32 getCurrentBufferStride(uint32* contextRegister) const;
};

// from a pool (PoolNew.h), its groups and attributes in place: a fetch
// shader first seen while a game plays allocates nothing
struct LatteFetchShader : PoolNew<LatteFetchShader>
{
	using CacheHash = uint64;

	// attributes a fetch shader may fetch, of every buffer together
	static constexpr size_t kMaxAttribs = 128;
	// groups of each kind: one a vertex buffer (the invalid ones' indices
	// past the 16 valid, as many again)
	static constexpr size_t kMaxGroups = 16;

	LatteFetchShader() = default;
	// its groups point into its own m_attribs
	LatteFetchShader(const LatteFetchShader&) = delete;
	LatteFetchShader& operator=(const LatteFetchShader&) = delete;
	~LatteFetchShader();

	boost::container::static_vector<LatteParsedFetchShaderBufferGroup, kMaxGroups> bufferGroups;
	boost::container::static_vector<LatteParsedFetchShaderBufferGroup, kMaxGroups> bufferGroupsInvalid; // groups with buffer index not being a valid buffer (dst components of these can affect shader code, but no actual vertex imports are done)
	// the groups' attributes, each group's together (FinishAttribs)
	LatteParsedFetchShaderAttribute m_attribs[kMaxAttribs];
	uint32 m_attribCount{};
	// while parsing, each attribute's group: its index, the invalid ones' after
	// the valid ones' (kMaxGroups on)
	uint8 m_attribGroup[kMaxAttribs];

	// an attribute for the group, in parse order (nullptr past kMaxAttribs)
	LatteParsedFetchShaderAttribute* AddAttrib(LatteParsedFetchShaderBufferGroup* group);
	// the attributes put in each group's order and its attrib pointed at them
	void FinishAttribs();

	uint64 key{};
	uint32 attributeBufferMask{}; // mask of buffers sourced by this fetch shader

	// Vulkan
	uint64 vkPipelineHashFragment{}; // hash of all fetch shader state that influences the Vulkan graphics pipeline

	// Metal
	bool mtlFetchVertexManually{};

	// cache info
	CacheHash m_cacheHash{};
	bool m_isRegistered{}; // if true, fetch shader is referenced by cache (RegisterInCache() succeeded)

	void CalculateFetchShaderVkHash();

#ifdef ENABLE_METAL
	void CheckIfVerticesNeedManualFetchMtl(uint32* contextRegister);
#endif

	uint64 getVkPipelineHashFragment() const { return vkPipelineHashFragment; };

	static bool isValidBufferIndex(const uint32 index) { return index < 0x10; };


	// fetch shader cache, the GPU thread's alone (FetchShaderRegistry.h)
	LatteFetchShader* RegisterInCache(CacheHash fsHash); // fails if another fetch shader object is already registered with the same fsHash. Returns the previously registered fetch shader, null when this one is registered, or this one when the cache is full (not registered)
	void UnregisterInCache();
	static CacheHash CalculateCacheHash(void* programCode, uint32 programSize);
	static LatteFetchShader* FindInCacheByHash(CacheHash fsHash);
	static LatteFetchShader* FindByGPUState();
};

LatteFetchShader* LatteShaderRecompiler_createFetchShader(LatteFetchShader::CacheHash fsHash, uint32* contextRegister, uint32* fsProgramCode, uint32 fsProgramSize);
