#include "Cafe/HW/Latte/Core/LatteShaderAhead.h"
#include "Cafe/HW/Latte/Core/LattePM4Scan.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
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
#endif

void _calcShaderHashGeneric(uint32* programCode, uint32 programSize, uint64& outputHash1, uint64& outputHash2);

namespace
{
	struct AheadPixelShader
	{
		uint64 digest;
		uint64 auxHash;
		uint32 scan; // the read it was made in, to let one never taken go
		LatteDecompilerOutput_t output;
	};
	// one key a slot (AheadSlots.h); a shader whose slot is taken is left to
	// its draw
	AheadSlots<AheadPixelShader, 64> s_pixel;
	// the pixel shaders the GPU thread has: none decompiled ahead again
	SeenKeys<1u << 15> s_seenPixel;

	constexpr uint32 kJobs = 4;
	constexpr uint32 kMaxProgram = 128 * 1024;
	constexpr uint32 kMaxPerScan = 8;
	// a shader made ahead and not taken by this many reads later goes
	constexpr uint32 kStaleScans = 64;

	// the registers copied from the GPU's for a read: what a pixel shader's
	// decompile and its hashes read is in these (LattePM4_PixelShaderInputs)
	constexpr struct { uint32 start, count; } kCopied[] = {
		{LATTE_REG_BASE_CONFIG, 0x1000},
		{LATTE_REG_BASE_CONTEXT, 0x1000},
		{LATTE_REG_BASE_RESOURCE, 0x1000},
	};

	struct Job
	{
		uint32 regs[LATTE_MAX_REGISTER];
		uint8 program[kMaxProgram];
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

	bool OnDraw(void* ctx, uint32* regs)
	{
		Job* job = static_cast<Job*>(ctx);
		if ((regs[Latte::REGADDR::VGT_GS_MODE] & 3) != 0)
			return true; // the pixel shader's hash takes the geometry shader's: left to the draw
		const uint32 addr = regs[mmSQ_PGM_START_PS] << 8;
		const uint32 size = regs[mmSQ_PGM_START_PS + 1] << 3;
		if (addr == 0 || size == 0 || size > kMaxProgram || !memory_isAddressRangeAccessible(addr, size))
			return true;
		// decompiled from a copy, the one hashed: guest memory may change under the read
		memcpy(job->program, memory_getPointerFromPhysicalOffset(addr), size);
		uint64 h1, h2;
		_calcShaderHashGeneric(reinterpret_cast<uint32*>(job->program), size, h1, h2);
		LatteShader_CreatePSInputTable(&job->inputs, regs);
		const uint64 baseHash = h1 + h2 + job->inputs.key; // LatteSHRC_UpdatePSBaseHash, no geometry shader
		if (s_seenPixel.Contains(baseHash))
			return true;
		const uint32 scan = job->scan;
		bool hadPayload;
		auto held = s_pixel.Claim(baseHash, [scan](uint64 key, const AheadPixelShader& p) {
			return s_seenPixel.Contains(key) || scan - p.scan > kStaleScans;
		}, &hadPayload);
		if (!held)
			return true;
		if (hadPayload)
			LatteShader_FreeDecompilerOutput(held.payload->output);
		held.payload->output = {};
		LatteShader_DecompilePixelShaderAhead(regs, job->program, size, &job->inputs, baseHash, &held.payload->output);
		if (!held.payload->output.shader)
		{
			s_pixel.Abandon(held);
			return true;
		}
		held.payload->auxHash = LatteSHRC_CalcPSAuxHash(held.payload->output.shader, regs);
		held.payload->digest = LattePM4_PixelShaderInputs(regs);
		held.payload->scan = scan;
		s_pixel.Publish(held);
		return ++job->found < kMaxPerScan;
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
	s_pixel.DrainPublished([](AheadPixelShader& p) {
		LatteShader_FreeDecompilerOutput(p.output);
		p.output = {};
	});
	s_seenPixel.Clear();
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
	PipelineCompiler::CompileThreadPool_QueuePrepared(*prepared);
	s_skip = (1u << s_backoff.load(std::memory_order_relaxed)) - 1;
#endif
}

bool LatteShaderAhead_TakePixelShader(uint64 baseHash, LatteDecompilerOutput_t& output)
{
	auto held = s_pixel.Take(baseHash);
	if (!held)
		return false;
	AheadPixelShader& p = *held.payload;
	const bool same = p.digest == LattePM4_PixelShaderInputs(LatteGPUState.contextRegister) &&
		p.auxHash == LatteSHRC_CalcPSAuxHash(p.output.shader, LatteGPUState.contextRegister);
	if (same)
		output = p.output;
	else
		LatteShader_FreeDecompilerOutput(p.output);
	p.output = {};
	s_pixel.Release(held);
	return same;
}

void LatteShaderAhead_NotePixelShader(uint64 baseHash)
{
	s_seenPixel.Insert(baseHash);
}
