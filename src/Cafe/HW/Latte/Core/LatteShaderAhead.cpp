#include "Cafe/HW/Latte/Core/LatteShaderAhead.h"
#include "Cafe/HW/Latte/Core/LattePM4Scan.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/Core/FetchShader.h"
#include "Cafe/HW/Latte/Core/LattePM4.h"
#include "Cafe/HW/Latte/Core/Latte.h"
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "Cafe/HW/MMU/MMU.h"
#include "util/helpers/AheadSlots.h"
#ifdef ENABLE_VULKAN
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/PreparedPipeline.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/RendererShaderVk.h"
#endif

void _calcShaderHashGeneric(uint32* programCode, uint32 programSize, uint64& outputHash1, uint64& outputHash2);

namespace
{
	struct AheadShader
	{
		uint64 digest; // the registers its decompile read (LattePM4Scan.h)
		uint64 auxHash;
		uint64 fetchShader; // a vertex shader's: its fetch shader's Fingerprint
		uint32 scan; // the read it was made in, to let one never taken go
		// its module made (Vulkan), its parts not yet queued by the GPU thread
		bool partsPending;
		bool robust; // what the shader itself asks of its parts
		LatteDecompilerOutput_t output;
	};
	// one key a slot (AheadSlots.h); a shader whose slot is taken is left to
	// its draw
	using Slots = AheadSlots<AheadShader, 64>;
	// the shaders the GPU thread has: none decompiled ahead again
	using Seen = SeenKeys<1u << 15>;
	Slots s_pixel, s_vertex;
	Seen s_seenPixel, s_seenVertex;
	// what happened to them, logged as the title stops: made ahead, taken by
	// their draw, not usable there, the draw first while one was being made,
	// none there for the draw, let go unused (vertex, pixel)
	struct Counts
	{
		std::atomic<uint64> made{0}, taken{0}, unusable{0}, late{0}, none{0}, letGo{0};
	};
	Counts s_countsVertex, s_countsPixel;

	// the slots whose shader waits for the GPU thread to queue its parts
	// (LatteShaderAhead_Poll), a bit each
	std::atomic<uint64> s_partsPixel{0}, s_partsVertex{0};

	constexpr uint32 kJobs = 4;
	constexpr uint32 kMaxProgram = 128 * 1024;
	constexpr uint32 kMaxFetchProgram = 16 * 1024;
	constexpr uint32 kMaxPerScan = 8;
	// a shader made ahead and not taken by this many reads later goes
	constexpr uint32 kStaleScans = 64;

	// the registers copied from the GPU's for a read: what a shader's
	// decompile and its hashes read is in these (LattePM4_PixelShaderInputs,
	// LattePM4_VertexShaderInputs)
	constexpr struct { uint32 start, count; } kCopied[] = {
		{LATTE_REG_BASE_CONFIG, 0x1000},
		{LATTE_REG_BASE_CONTEXT, 0x1000},
		{LATTE_REG_BASE_RESOURCE, 0x1000},
	};

	struct Job
	{
		uint32 regs[LATTE_MAX_REGISTER];
		// copies of the programs, each the one hashed and decompiled: guest
		// memory may change under the read
		uint8 psProgram[kMaxProgram];
		uint8 vsProgram[kMaxProgram];
		uint8 fsProgram[kMaxFetchProgram];
		// the vertex shader's fetch shader, parsed here, never registered
		LatteFetchShader fetchShader;
		LatteShaderPSInputTable inputs;
		uint32 physAddr;
		uint32 words;
		uint32 scan;
		uint32 found;
#ifdef ENABLE_VULKAN
		alignas(PreparedPipeline) unsigned char prepared[sizeof(PreparedPipeline)];
		bool queued;
		PreparedPipeline* Prepared() { return reinterpret_cast<PreparedPipeline*>(prepared); }
#endif
	};
	Job* s_jobs;
	// reads started (GPU thread), and how many command buffers to pass over
	// before the next: doubled, up to 63, for every read that found nothing
	std::atomic<uint32> s_scans{0};
	std::atomic<uint32> s_backoff{0};
	uint32 s_skip;

	const uint32be* MapGuest(void*, uint32 physAddr, uint32 count)
	{
		if (count == 0 || count > (1u << 22) || !memory_isAddressRangeAccessible(physAddr, count * 4))
			return nullptr;
		return reinterpret_cast<const uint32be*>(memory_getPointerFromPhysicalOffset(physAddr));
	}

	// size bytes of guest memory at a physical address, copied; false where
	// they cannot be read or do not fit
	bool CopyGuest(uint8* to, uint32 addr, uint32 size, uint32 room)
	{
		if (size > room || (size && (addr == 0 || !memory_isAddressRangeAccessible(addr, size))))
			return false;
		if (size)
			memcpy(to, memory_getPointerFromPhysicalOffset(addr), size);
		return true;
	}

	// decompile(output) for baseHash, published for its draw with finish(it)
	// filling in what the draw checks it by, and its module made, for the
	// GPU thread to queue its parts (parts: the slots' pending bits)
	template<typename Decompile, typename Finish>
	bool MakeAhead(Slots& slots, Seen& seen, std::atomic<uint64>& parts, Counts& counts, uint64 baseHash, uint32 scan, Decompile decompile, Finish finish)
	{
		if (seen.Contains(baseHash))
			return false;
		bool hadPayload;
		auto held = slots.Claim(baseHash, [&seen, scan](uint64 key, const AheadShader& p) {
			return seen.Contains(key) || scan - p.scan > kStaleScans;
		}, &hadPayload);
		if (!held)
			return false;
		if (hadPayload)
		{
			LatteShader_FreeDecompilerOutput(held.payload->output);
			counts.letGo.fetch_add(1, std::memory_order_relaxed);
		}
		held.payload->output = {};
		decompile(held.payload->output);
		if (!held.payload->output.shader)
		{
			slots.Abandon(held);
			return false;
		}
		finish(*held.payload);
		held.payload->scan = scan;
		held.payload->robust = false;
		held.payload->partsPending = LatteShader_MakeRendererShaderAhead(held.payload->output, baseHash, held.payload->auxHash);
#ifdef ENABLE_VULKAN
		held.payload->robust = held.payload->output.shader->shaderType == LatteConst::ShaderType::Vertex
			? PipelineCompiler::CalcRobustBufferAccessRequirement(held.payload->output.shader, nullptr, nullptr)
			: PipelineCompiler::CalcRobustBufferAccessRequirement(nullptr, held.payload->output.shader, nullptr);
#endif
		const size_t index = held.index;
		const bool pending = held.payload->partsPending;
		slots.Publish(held);
		counts.made.fetch_add(1, std::memory_order_relaxed);
		if (pending)
			parts.fetch_or(1ull << index, std::memory_order_release);
		return true;
	}

	void PixelAhead(Job* job, uint32* regs)
	{
		const uint32 size = regs[mmSQ_PGM_START_PS + 1] << 3;
		if (size == 0 || !CopyGuest(job->psProgram, regs[mmSQ_PGM_START_PS] << 8, size, kMaxProgram))
			return;
		uint64 h1, h2;
		_calcShaderHashGeneric(reinterpret_cast<uint32*>(job->psProgram), size, h1, h2);
		const uint64 baseHash = h1 + h2 + job->inputs.key; // LatteSHRC_UpdatePSBaseHash, no geometry shader
		job->found += MakeAhead(s_pixel, s_seenPixel, s_partsPixel, s_countsPixel, baseHash, job->scan,
			[&](LatteDecompilerOutput_t& out) {
				LatteShader_DecompilePixelShaderAhead(regs, job->psProgram, size, &job->inputs, baseHash, &out);
			},
			[&](AheadShader& p) {
				p.auxHash = LatteSHRC_CalcPSAuxHash(p.output.shader, regs);
				p.digest = LattePM4_PixelShaderInputs(regs);
				p.fetchShader = 0;
			});
	}

	void VertexAhead(Job* job, uint32* regs)
	{
		const uint32 size = regs[mmSQ_PGM_START_VS + 1] << 3;
		const uint32 fsSize = regs[mmSQ_PGM_START_FS + 1] << 3;
		if (size == 0 || !CopyGuest(job->vsProgram, regs[mmSQ_PGM_START_VS] << 8, size, kMaxProgram) ||
			!CopyGuest(job->fsProgram, regs[mmSQ_PGM_START_FS] << 8, fsSize, kMaxFetchProgram))
			return;
		// the fetch shader parsed as LatteFetchShader::FindByGPUState would parse it
		// for these registers; the draw's own, which may have been parsed for
		// others, has to match it (its Fingerprint)
		job->fetchShader.Reset();
		LatteFetchShader_Parse(&job->fetchShader, LatteFetchShader::CalculateCacheHash(job->fsProgram, fsSize), regs,
			reinterpret_cast<uint32*>(job->fsProgram), fsSize);
		uint64 h1, h2;
		_calcShaderHashGeneric(reinterpret_cast<uint32*>(job->vsProgram), size, h1, h2);
		const uint64 baseHash = LatteSHRC_VertexBaseHash(h1, h2, &job->fetchShader, job->inputs.key, false, regs);
		job->found += MakeAhead(s_vertex, s_seenVertex, s_partsVertex, s_countsVertex, baseHash, job->scan,
			[&](LatteDecompilerOutput_t& out) {
				LatteShader_DecompileVertexShaderAhead(regs, job->vsProgram, size, &job->fetchShader, &job->inputs, baseHash, &out);
			},
			[&](AheadShader& p) {
				p.auxHash = LatteSHRC_CalcVSAuxHash(p.output.shader, regs);
				p.digest = LattePM4_VertexShaderInputs(regs);
				p.fetchShader = job->fetchShader.Fingerprint();
				p.output.shader->compatibleFetchShader = nullptr; // the draw's, once taken
			});
	}

	bool OnDraw(void* ctx, uint32* regs)
	{
		Job* job = static_cast<Job*>(ctx);
		if ((regs[Latte::REGADDR::VGT_GS_MODE] & 3) != 0)
			return true; // the shaders' hashes take the geometry shader's: left to the draw
		LatteShader_CreatePSInputTable(&job->inputs, regs);
		PixelAhead(job, regs);
		VertexAhead(job, regs);
		return job->found < kMaxPerScan;
	}

	void RunJob(Job* job)
	{
		job->found = 0;
		if (const uint32be* buf = MapGuest(nullptr, job->physAddr, job->words))
		{
			LattePM4Scan scan{};
			scan.regs = job->regs;
			scan.map = MapGuest;
			scan.onDraw = OnDraw;
			scan.ctx = job;
			LattePM4_Scan(scan, buf, job->words);
		}
		const uint32 backoff = s_backoff.load(std::memory_order_relaxed);
		s_backoff.store(job->found ? 0 : (backoff < 6 ? backoff + 1 : 6), std::memory_order_relaxed);
	}
}

void LatteShaderAhead_Init()
{
#ifdef ENABLE_VULKAN
	if (s_jobs || g_renderer->GetType() != RendererAPI::Vulkan)
		return;
	s_jobs = new Job[kJobs];
	for (uint32 i = 0; i < kJobs; i++)
	{
		memset(s_jobs[i].regs, 0, sizeof(s_jobs[i].regs));
		s_jobs[i].queued = false;
	}
	s_skip = 0;
	s_backoff.store(0, std::memory_order_relaxed);
#endif
}

void LatteShaderAhead_Shutdown()
{
#ifdef ENABLE_VULKAN
	if (s_jobs)
	{
		// a read under way finished, one not started taken back
		for (uint32 i = 0; i < kJobs; i++)
		{
			if (s_jobs[i].queued)
				s_jobs[i].Prepared()->~PreparedPipeline();
		}
		delete[] s_jobs;
		s_jobs = nullptr;
	}
#endif
	for (Counts* c : {&s_countsVertex, &s_countsPixel})
	{
		if (c->made.load(std::memory_order_relaxed) || c->none.load(std::memory_order_relaxed))
			cemuLog_log(LogType::Force, "[ShaderAhead] {} shaders: {} decompiled ahead, {} taken by their draw, {} not usable there, {} drawn while being made, {} drawn with none made, {} let go unused",
				c == &s_countsVertex ? "vertex" : "pixel", c->made.load(), c->taken.load(), c->unusable.load(), c->late.load(), c->none.load(), c->letGo.load());
		c->made = 0;
		c->taken = 0;
		c->unusable = 0;
		c->late = 0;
		c->none = 0;
		c->letGo = 0;
	}
	for (Slots* slots : {&s_pixel, &s_vertex})
	{
		slots->DrainPublished([](AheadShader& p) {
			LatteShader_FreeDecompilerOutput(p.output);
			p.output = {};
		});
	}
	s_seenPixel.Clear();
	s_seenVertex.Clear();
}

void LatteShaderAhead_OnCommandBuffer(uint32 physAddr, uint32 words)
{
#ifdef ENABLE_VULKAN
	if (!s_jobs || words == 0)
		return;
	if (s_skip)
	{
		s_skip--;
		return;
	}
	Job* job = nullptr;
	for (uint32 i = 0; i < kJobs && !job; i++)
	{
		Job& j = s_jobs[i];
		if (j.queued)
		{
			if (!j.Prepared()->IsBuilt())
				continue; // still reading, or not yet started: never waited for
			j.Prepared()->~PreparedPipeline();
			j.queued = false;
		}
		job = &j;
	}
	if (!job)
		return;
	for (const auto& r : kCopied)
		memcpy(job->regs + r.start, LatteGPUState.contextRegister + r.start, r.count * sizeof(uint32));
	job->physAddr = physAddr;
	job->words = words;
	job->scan = s_scans.fetch_add(1, std::memory_order_relaxed) + 1;
	PreparedPipeline* prepared = new (job->prepared) PreparedPipeline([](void* ctx, bool) {
		RunJob(static_cast<Job*>(ctx));
		return (VkPipeline)VK_NULL_HANDLE;
	}, job);
	job->queued = true;
	PipelineCompiler::CompileThreadPool_QueuePreparedUrgent(*prepared);
	s_skip = (1u << s_backoff.load(std::memory_order_relaxed)) - 1;
#endif
}

bool LatteShaderAhead_TakePixelShader(uint64 baseHash, LatteDecompilerOutput_t& output)
{
	auto held = s_pixel.Take(baseHash);
	if (!held)
	{
		(s_pixel.InHand(baseHash) ? s_countsPixel.late : s_countsPixel.none).fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	AheadShader& p = *held.payload;
	const bool same = p.digest == LattePM4_PixelShaderInputs(LatteGPUState.contextRegister) &&
		p.auxHash == LatteSHRC_CalcPSAuxHash(p.output.shader, LatteGPUState.contextRegister);
	if (!same)
	{
		// put back, not freed here: its module and parts may be being made
		// (destroying them would wait); a read frees it once its base hash is
		// known, which the draw's own decompile makes it
		s_pixel.Publish(held);
		s_countsPixel.unusable.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	s_countsPixel.taken.fetch_add(1, std::memory_order_relaxed);
	output = p.output;
	p.output = {};
	s_pixel.Release(held);
	return true;
}

bool LatteShaderAhead_TakeVertexShader(uint64 baseHash, LatteFetchShader* fetchShader, LatteDecompilerOutput_t& output)
{
	auto held = s_vertex.Take(baseHash);
	if (!held)
	{
		(s_vertex.InHand(baseHash) ? s_countsVertex.late : s_countsVertex.none).fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	AheadShader& p = *held.payload;
	const bool same = p.digest == LattePM4_VertexShaderInputs(LatteGPUState.contextRegister) &&
		p.fetchShader == fetchShader->Fingerprint() &&
		p.auxHash == LatteSHRC_CalcVSAuxHash(p.output.shader, LatteGPUState.contextRegister);
	if (!same)
	{
		s_vertex.Publish(held); // as a pixel shader's: put back
		s_countsVertex.unusable.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	s_countsVertex.taken.fetch_add(1, std::memory_order_relaxed);
	output = p.output;
	output.shader->compatibleFetchShader = fetchShader;
	p.output = {};
	s_vertex.Release(held);
	return true;
}

void LatteShaderAhead_NotePixelShader(uint64 baseHash)
{
	s_seenPixel.Insert(baseHash);
}

void LatteShaderAhead_NoteVertexShader(uint64 baseHash)
{
	s_seenVertex.Insert(baseHash);
}

void LatteShaderAhead_Poll()
{
#ifdef ENABLE_VULKAN
	struct Kind
	{
		Slots& slots;
		std::atomic<uint64>& parts;
	};
	for (Kind k : {Kind{s_pixel, s_partsPixel}, Kind{s_vertex, s_partsVertex}})
	{
		if (k.parts.load(std::memory_order_relaxed) == 0)
			continue;
		for (uint64 bits = k.parts.exchange(0, std::memory_order_acquire); bits; bits &= bits - 1)
		{
			// held while its parts are queued: no draw or read takes it meanwhile
			auto held = k.slots.TakeAt((size_t)std::countr_zero(bits));
			if (!held)
				continue;
			AheadShader& p = *held.payload;
			if (p.partsPending && p.output.shader && p.output.shader->shader)
				PipelineCompiler::PrepareAheadPartsFor(static_cast<RendererShaderVk*>(p.output.shader->shader), p.output.shader, p.robust);
			p.partsPending = false;
			k.slots.Publish(held);
		}
	}
#endif
}
