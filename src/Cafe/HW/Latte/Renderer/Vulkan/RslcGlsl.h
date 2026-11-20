#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gfx/shaders/rslc.h>

// What RendererShaderVk hands rslc for a stage's GLSL.
//
// rslc takes #version 450 and nothing else; the decompiler writes 430, the
// output shaders 420 to 450, the graphic packs anything. Their sources test
// VULKAN, which glslang predefined for a Vulkan target. #line keeps a
// message's line numbers the source's.
//
// A round-to-even mode (spirv_execution_mode 4462) stays only for 32 bit, and
// only where the device takes it (@rte32), as the decompiler writes it: the
// graphic packs carry the 16 and 64 bit ones older decompilers wrote, whose
// device features nothing checks. glslang kept the last mode a stage
// declared alone; rslc would keep every one.
inline bool RslcGlslDropMode(std::string_view line, bool rte32)
{
	const size_t at = line.find_first_not_of(" \t");
	if (at == std::string_view::npos || line.compare(at, 21, "spirv_execution_mode(") != 0)
		return false;
	const size_t close = line.rfind(')');
	const size_t comma = line.rfind(',', close);
	if (close == std::string_view::npos || comma == std::string_view::npos)
		return false;
	const size_t mode = line.rfind(',', comma - 1);
	const auto field = [&line](size_t from, size_t to) {
		const size_t b = line.find_first_not_of(" \t(", from + 1);
		return b < to ? line.substr(b, line.find_last_not_of(" \t", to - 1) + 1 - b) : std::string_view();
	};
	if (field(mode == std::string_view::npos ? line.find('(') : mode, comma) != "4462")
		return false;
	return !rte32 || field(comma, close) != "32";
}

// into out, emptied first: no allocation while out has room
inline void RslcGlslInto(std::string_view glsl, bool rte32, std::string& out)
{
	out.assign("#version 450\n#define VULKAN 100\n#line 1\n");
	size_t at = 0;
	bool versionSeen = false;
	while (at < glsl.size())
	{
		size_t end = glsl.find('\n', at);
		if (end == std::string_view::npos)
			end = glsl.size();
		const std::string_view line = glsl.substr(at, end - at);
		const size_t first = line.find_first_not_of(" \t");
		if (!versionSeen && first != std::string_view::npos && line.compare(first, 8, "#version") == 0)
			versionSeen = true; // its line stays, empty
		else if (!RslcGlslDropMode(line, rte32))
			out.append(line);
		out.push_back('\n');
		at = end + 1;
	}
}

inline std::string RslcGlsl(const std::string& glsl, bool rte32)
{
	std::string out;
	out.reserve(64 + glsl.size());
	RslcGlslInto(glsl, rte32, out);
	return out;
}

// A thread's room for rslc's compiles: the source as rslc takes it, the arena
// a compile works in (no chunks of its own), and the module it returns. Each
// grows only past the largest compile the thread has made - the arena when a
// compile runs out of it, doubled and the compile made again - so once warm
// (Warm, as the thread starts) a compile allocates nothing.
struct RslcScratch
{
	static constexpr size_t kArena = 2u << 20;
	static constexpr size_t kArenaMax = 64u << 20;

	std::string glsl;
	std::vector<uint8_t> arena;
	std::vector<uint8_t> blob;
	bool blobLent = false;
	// modules rslc asked more room for while blob was lent: from malloc
	uint32_t heapBlobs = 0;

	void Warm(size_t arenaBytes = kArena)
	{
		glsl.reserve(64 * 1024);
		if (arena.size() < arenaBytes)
			arena.resize(arenaBytes);
		blob.reserve(64 * 1024);
	}
};

// rslc's allocator for a scratch: the module into its room, or - while that
// is lent, which a compile does not do - from the heap
inline void* RslcScratchAlloc(void* ud, size_t len)
{
	RslcScratch& s = *(RslcScratch*)ud;
	if (s.blobLent)
	{
		s.heapBlobs++;
		return std::malloc(len);
	}
	s.blob.resize(len);
	s.blobLent = true;
	return s.blob.data();
}

inline void RslcScratchFree(void* ud, void* p)
{
	RslcScratch& s = *(RslcScratch*)ud;
	if (s.blobLent && p == s.blob.data())
		s.blobLent = false;
	else
		std::free(p);
}

// source compiled to SPIR-V in scratch's room: the words, lent until the next
// compile with it; empty, with err filled, when it does not compile
inline std::span<const uint32_t> RslcCompileSpirv(std::string_view source, rslc_stage stage, bool rte32, RslcScratch& s, char* err, size_t errCap)
{
	RslcGlslInto(source, rte32, s.glsl);
	if (s.arena.empty())
		s.Warm();
	while (true)
	{
		err[0] = '\0';
		rslc_opts opts{};
		opts.dst = RSLC_DST_SPIRV;
		opts.err = err;
		opts.err_cap = errCap;
		opts.arena = s.arena.data();
		opts.arena_len = s.arena.size();
		opts.ud = &s;
		opts.alloc = RslcScratchAlloc;
		opts.free_fn = RslcScratchFree;
		rslc_result result{};
		if (rslc_compile(s.glsl.data(), s.glsl.size(), stage, &opts, &result))
		{
			const size_t words = result.spirv_len;
			const size_t offset = (const uint8_t*)result.spirv - (const uint8_t*)result.blob;
			if (result.blob != s.blob.data())
			{
				// from malloc: into the scratch's own
				s.blob.assign((const uint8_t*)result.blob, (const uint8_t*)result.blob + result.blob_len);
				rslc_result_free(&result, &opts);
			}
			s.blobLent = false;
			return std::span<const uint32_t>((const uint32_t*)(s.blob.data() + offset), words);
		}
		s.blobLent = false;
		const bool outOfArena = std::strstr(err, "caller arena") != nullptr;
		if (!outOfArena || s.arena.size() >= RslcScratch::kArenaMax)
			return {};
		s.arena.resize(s.arena.size() * 2);
	}
}
