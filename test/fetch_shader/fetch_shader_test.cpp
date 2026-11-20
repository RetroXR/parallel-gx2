// Fetch shaders (FetchShader.cpp) parsed from programs whose attributes are
// fetched from several buffers in turn: each buffer's group with its own
// attributes together, in the order they were fetched, as the key, the
// pipeline and the decompiler read them; more buffers or attributes than
// there is room for left out, not written past; and a fetch shader made and
// deleted again and again taking nothing from the allocator.
#include "Cafe/HW/Latte/Core/FetchShader.h"
#include "Cafe/HW/Latte/ISA/LatteReg.h"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define COUNT_ALLOCS 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define COUNT_ALLOCS 0
#endif
#endif
#ifndef COUNT_ALLOCS
#define COUNT_ALLOCS 1
#endif
static unsigned long long s_allocs = 0;
#if COUNT_ALLOCS
void* operator new(size_t size)
{
	s_allocs++;
	if (void* p = std::malloc(size ? size : 1))
		return p;
	throw std::bad_alloc();
}
void* operator new[](size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
#endif

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

struct Fetch
{
	uint32 buffer;   // 0..31: past 15 an invalid one
	uint32 semantic;
	uint32 offset;
};

// a fetch shader program without a CF program: one VTX_INST_SEMANTIC each
static std::vector<uint32> Program(const std::vector<Fetch>& fetches)
{
	std::vector<uint32> words;
	for (const Fetch& f : fetches)
	{
		words.push_back(1u | (0xA0u + f.buffer) << 8); // vertex data, SRC_SEL_X: X
		words.push_back(f.semantic | (uint32)Latte::E_HWFMT::HWFMT_8_8_8_8 << 22);
		words.push_back(f.offset);
		words.push_back(0);
	}
	return words;
}

static uint32 s_regs[0x10000];

static LatteFetchShader* Make(const std::vector<Fetch>& fetches)
{
	std::vector<uint32> program = Program(fetches);
	const auto hash = LatteFetchShader::CalculateCacheHash(program.data(), (uint32)program.size() * 4);
	return LatteShaderRecompiler_createFetchShader(hash, s_regs, program.data(), (uint32)program.size() * 4);
}

// each fetch in its buffer's group, in order, a group's together
static void CheckGroups(const LatteFetchShader* fs, const std::vector<Fetch>& fetches)
{
	std::vector<uint32> order; // buffers as first fetched
	for (const Fetch& f : fetches)
	{
		bool seen = false;
		for (uint32 b : order)
			seen |= b == f.buffer;
		if (!seen)
			order.push_back(f.buffer);
	}
	size_t valid = 0, invalid = 0;
	for (uint32 b : order)
	{
		const bool isValid = b < 16;
		const auto& groups = isValid ? fs->bufferGroups : fs->bufferGroupsInvalid;
		const size_t g = isValid ? valid++ : invalid++;
		CHECK(g < groups.size());
		if (g >= groups.size())
			continue;
		const LatteParsedFetchShaderBufferGroup& group = groups[g];
		CHECK(group.attributeBufferIndex == b);
		int k = 0;
		for (const Fetch& f : fetches)
		{
			if (f.buffer != b)
				continue;
			CHECK(k < group.attribCount);
			if (k < group.attribCount)
				CHECK(group.attrib[k].semanticId == f.semantic && group.attrib[k].offset == f.offset && group.attrib[k].attributeBufferIndex == b);
			k++;
		}
		CHECK(k == group.attribCount);
		// in the fetch shader's own storage
		CHECK(group.attrib >= fs->m_attribs && group.attrib + group.attribCount <= fs->m_attribs + LatteFetchShader::kMaxAttribs);
	}
	CHECK(valid == fs->bufferGroups.size() && invalid == fs->bufferGroupsInvalid.size());
}

int main()
{
	std::mt19937 rng(9);
	for (int round = 0; round < 500; round++)
	{
		std::vector<Fetch> fetches;
		const int n = 1 + rng() % 24;
		const uint32 buffers = round % 3 == 0 ? 20 : 6; // some past the 16 valid
		for (int i = 0; i < n; i++)
			fetches.push_back({ (uint32)(rng() % buffers), (uint32)(rng() % 256), (uint32)(rng() % 64) * 4 });
		LatteFetchShader* fs = Make(fetches);
		CheckGroups(fs, fetches);
		delete fs;
	}
	// more attributes than there is room for, and more buffers of a kind:
	// what fits kept, the rest left out
	{
		std::vector<Fetch> fetches;
		for (uint32 i = 0; i < LatteFetchShader::kMaxAttribs + 40; i++)
			fetches.push_back({ i % 16, i & 0xFF, i * 4 });
		LatteFetchShader* fs = Make(fetches);
		size_t total = 0;
		for (auto& g : fs->bufferGroups)
			total += g.attribCount;
		CHECK(total == LatteFetchShader::kMaxAttribs && fs->m_attribCount == LatteFetchShader::kMaxAttribs);
		std::vector<Fetch> kept(fetches.begin(), fetches.begin() + LatteFetchShader::kMaxAttribs);
		CheckGroups(fs, kept);
		delete fs;
		fetches.clear();
		for (uint32 i = 0; i < 40; i++)
			fetches.push_back({ 16 + i % 20, i, 0 }); // 20 invalid buffers
		fs = Make(fetches);
		CHECK(fs->bufferGroupsInvalid.size() == LatteFetchShader::kMaxGroups);
		delete fs;
	}
	if (COUNT_ALLOCS)
	{
		// a fetch shader made and deleted: from the pool, nothing allocated
		std::vector<Fetch> fetches = { { 0, 1, 0 }, { 1, 2, 4 }, { 0, 3, 8 }, { 2, 4, 0 } };
		std::vector<uint32> program = Program(fetches);
		const auto hash = LatteFetchShader::CalculateCacheHash(program.data(), (uint32)program.size() * 4);
		delete LatteShaderRecompiler_createFetchShader(hash, s_regs, program.data(), (uint32)program.size() * 4);
		const unsigned long long before = s_allocs;
		for (int i = 0; i < 1000; i++)
			delete LatteShaderRecompiler_createFetchShader(hash, s_regs, program.data(), (uint32)program.size() * 4);
		std::printf("fetch_shader: %llu allocations for 1000 fetch shaders made\n", s_allocs - before);
		CHECK(s_allocs == before);
	}
	if (s_failures)
	{
		std::printf("fetch_shader_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("fetch_shader_test: all passed\n");
	return 0;
}
