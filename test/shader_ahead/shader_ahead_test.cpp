// Pixel shaders decompiled ahead of their draw (LatteShaderAhead.cpp), the
// parts that run without a GPU:
//  - a command buffer read ahead (LattePM4Scan.cpp) applies its register
//    writes as the command processor does - context, config, resource,
//    sampler, ALU and control constants, the semantic table clear - loads
//    registers from guest memory, follows the buffers it calls (to the
//    depth limit), calls back at each draw with the registers as they are
//    there, and stops at a wait on memory, at a packet it cannot read, at a
//    buffer it cannot map, at the packet limit and where the reader asks;
//    random buffers of noise end the read without reading out of bounds;
//  - the registers a pixel or vertex shader's decompile reads, as one
//    value (LattePM4_PixelShaderInputs, LattePM4_VertexShaderInputs): moved
//    by each of them, not by the vertex buffers, constants or the other
//    stage's textures;
//  - the hand-over (AheadSlots.h): a key's slot claimed once, published,
//    taken once by that key only, a stale one claimed back for another key,
//    one never published left to the draw; four producers and a consumer
//    racing over 20000 keys, every payload published either taken or
//    drained, none twice (TSan in CI); and the filter of keys seen, which
//    never says unseen for a key added.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "Cafe/HW/Latte/Core/LattePM4Scan.h"
#include "Cafe/HW/Latte/Core/LattePM4.h"
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "util/helpers/AheadSlots.h"

static int s_failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); s_failures++; } } while (0)

// --- guest memory and buffers -------------------------------------------------

// a small guest memory: physical address 0x10000000 up
static constexpr uint32 kGuestBase = 0x10000000;
static std::vector<uint32be> s_guest(1 << 16);

static const uint32be* MapGuest(void*, uint32 addr, uint32 count)
{
	if (addr < kGuestBase || (addr & 3) != 0)
		return nullptr;
	const uint64 first = (addr - kGuestBase) / 4;
	if (first + count > s_guest.size())
		return nullptr;
	return s_guest.data() + first;
}

struct Buf
{
	std::vector<uint32be> w;
	void Packet(uint32 it, std::initializer_list<uint32> data)
	{
		w.push_back(pm4HeaderType3(it, (uint32)data.size()));
		for (uint32 d : data)
			w.push_back(d);
	}
	void SetRegs(uint32 it, uint32 base, uint32 reg, std::initializer_list<uint32> values)
	{
		w.push_back(pm4HeaderType3(it, 1 + (uint32)values.size()));
		w.push_back(reg - base);
		for (uint32 v : values)
			w.push_back(v);
	}
	void Draw() { Packet(IT_DRAW_INDEX_AUTO, {3, 0}); }
};

// a buffer placed in guest memory at word index at
static uint32 Place(const Buf& b, uint32 at)
{
	for (size_t i = 0; i < b.w.size(); i++)
		s_guest[at + i] = b.w[i];
	return kGuestBase + at * 4;
}

struct Seen
{
	std::vector<std::vector<uint32>> at; // registers kept at each draw
	std::vector<uint32> watch;
	uint32 stopAfter = ~0u;
};

static bool OnDraw(void* ctx, uint32* regs)
{
	Seen* s = static_cast<Seen*>(ctx);
	std::vector<uint32> v;
	for (uint32 r : s->watch)
		v.push_back(regs[r]);
	s->at.push_back(v);
	return s->at.size() < s->stopAfter;
}

static std::vector<uint32> s_regs(LATTE_MAX_REGISTER);

static LattePM4Scan::End Scan(const Buf& b, Seen& seen, uint32 maxDepth = 4)
{
	LattePM4Scan scan{};
	scan.regs = s_regs.data();
	scan.map = MapGuest;
	scan.onDraw = OnDraw;
	scan.ctx = &seen;
	scan.maxDepth = maxDepth;
	LattePM4_Scan(scan, b.w.data(), (uint32)b.w.size());
	CHECK(scan.draws == seen.at.size());
	return scan.end;
}

static void Registers()
{
	std::fill(s_regs.begin(), s_regs.end(), 0u);
	Seen seen;
	seen.watch = {mmSQ_PGM_START_PS, mmSQ_PGM_START_PS + 1, mmCB_SHADER_MASK, Latte::REGADDR::VGT_PRIMITIVE_TYPE,
		Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS + 7, mmSQ_VTX_SEMANTIC_0 + 5, mmSPI_PS_INPUT_CNTL_0 + 2,
		LATTE_REG_BASE_ALU_CONST + 9, LATTE_REG_BASE_SAMPLER + 4, mmSQ_VTX_BASE_VTX_LOC + 1};
	s_regs[mmSQ_VTX_SEMANTIC_0 + 5] = 0x12;

	// a called buffer setting the program, another loading registers from memory
	Buf sub;
	sub.SetRegs(IT_SET_CONTEXT_REG, LATTE_REG_BASE_CONTEXT, mmSQ_PGM_START_PS, {0x123456, 0x40});
	const uint32 subAddr = Place(sub, 0x100);
	for (uint32 i = 0; i < 4; i++)
		s_guest[0x200 + i] = 0xA0 + i;
	Buf b;
	b.SetRegs(IT_SET_CONTEXT_REG, LATTE_REG_BASE_CONTEXT, mmCB_SHADER_MASK, {0xF});
	b.SetRegs(IT_SET_CONFIG_REG, LATTE_REG_BASE_CONFIG, Latte::REGADDR::VGT_PRIMITIVE_TYPE, {4});
	b.Draw();
	b.Packet(IT_INDIRECT_BUFFER_PRIV, {subAddr, 0, (uint32)sub.w.size()});
	b.SetRegs(IT_SET_RESOURCE, LATTE_REG_BASE_RESOURCE, Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS + 7, {0x77});
	b.SetRegs(IT_SET_ALU_CONST, LATTE_REG_BASE_ALU_CONST, LATTE_REG_BASE_ALU_CONST + 9, {0x99});
	b.SetRegs(IT_SET_SAMPLER, LATTE_REG_BASE_SAMPLER, LATTE_REG_BASE_SAMPLER + 4, {0x44});
	b.SetRegs(IT_SET_CTL_CONST, mmSQ_VTX_BASE_VTX_LOC, mmSQ_VTX_BASE_VTX_LOC + 1, {0x55});
	b.w.push_back(0x80000000); // a filler
	b.Packet(IT_HLE_SAMPLE_TIMER, {0, 0}); // writes nothing kept
	b.Draw();
	// the semantic table cleared, and SPI_PS_INPUT_CNTL loaded from memory
	b.SetRegs(IT_SET_CONTEXT_REG, LATTE_REG_BASE_CONTEXT, mmSQ_VTX_SEMANTIC_CLEAR, {0xFFFFFFFF});
	b.Packet(IT_LOAD_CONTEXT_REG, {kGuestBase + 0x200 * 4, 0, mmSPI_PS_INPUT_CNTL_0 - LATTE_REG_BASE_CONTEXT, 4});
	b.Packet(IT_DRAW_INDEX_2, {0, 0, 0, 0});
	CHECK(Scan(b, seen) == LattePM4Scan::End::Done);
	CHECK(seen.at.size() == 3);
	if (seen.at.size() == 3)
	{
		CHECK(seen.at[0] == (std::vector<uint32>{0, 0, 0xF, 4, 0, 0x12, 0, 0, 0, 0}));
		CHECK(seen.at[1] == (std::vector<uint32>{0x123456, 0x40, 0xF, 4, 0x77, 0x12, 0, 0x99, 0x44, 0x55}));
		CHECK(seen.at[2] == (std::vector<uint32>{0x123456, 0x40, 0xF, 4, 0x77, 0xFF, 0xA2, 0x99, 0x44, 0x55}));
	}
}

static void Ends()
{
	// stopped by the reader
	{
		Buf b;
		b.Draw();
		b.Draw();
		b.Draw();
		Seen seen;
		seen.stopAfter = 2;
		CHECK(Scan(b, seen) == LattePM4Scan::End::Stopped);
		CHECK(seen.at.size() == 2);
	}
	// a wait on memory: nothing after it read
	{
		Buf b;
		b.Draw();
		b.Packet(IT_WAIT_REG_MEM, {0, 0, 0, 0, 0, 0});
		b.Draw();
		Seen seen;
		CHECK(Scan(b, seen) == LattePM4Scan::End::Waited);
		CHECK(seen.at.size() == 1);
	}
	// a buffer that cannot be mapped, a packet past the end, a type 1 header
	{
		Buf b;
		b.Packet(IT_INDIRECT_BUFFER_PRIV, {0x1000, 0, 16});
		Seen seen;
		CHECK(Scan(b, seen) == LattePM4Scan::End::Unreadable);
		Buf c;
		c.w.push_back(pm4HeaderType3(IT_SET_CONTEXT_REG, 9));
		c.w.push_back(0);
		CHECK(Scan(c, seen) == LattePM4Scan::End::Unreadable);
		Buf d;
		d.w.push_back(0x40000000);
		CHECK(Scan(d, seen) == LattePM4Scan::End::Unreadable);
	}
	// a buffer calling itself: the depth limit
	{
		Buf self;
		self.Draw();
		self.Packet(IT_INDIRECT_BUFFER_PRIV, {kGuestBase + 0x400 * 4, 0, 0});
		self.w[self.w.size() - 1] = (uint32)self.w.size();
		const uint32 at = Place(self, 0x400);
		Buf b;
		b.Packet(IT_INDIRECT_BUFFER_PRIV, {at, 0, (uint32)self.w.size()});
		Seen seen;
		CHECK(Scan(b, seen, 4) == LattePM4Scan::End::Limit);
		CHECK(seen.at.size() == 4);
	}
	// noise: ends, never reads past what it was given
	{
		std::mt19937 rng(20261121u);
		int ended = 0;
		for (int i = 0; i < 2000; i++)
		{
			Buf b;
			const size_t n = 1 + rng() % 64;
			for (size_t k = 0; k < n; k++)
				b.w.push_back((uint32)rng());
			std::fill(s_regs.begin(), s_regs.end(), 0u);
			Seen seen;
			Scan(b, seen);
			ended++;
		}
		CHECK(ended == 2000);
	}
}

static void PixelShaderInputs()
{
	std::vector<uint32> regs(LATTE_MAX_REGISTER, 0);
	const uint64 base = LattePM4_PixelShaderInputs(regs.data());
	const uint32 read[] = {mmCB_SHADER_MASK, Latte::REGADDR::SX_ALPHA_TEST_CONTROL, mmSPI_VS_OUT_ID_0 + 9, mmSPI_PS_INPUT_CNTL_0 + 31,
		mmSPI_PS_IN_CONTROL_0, mmSPI_PS_IN_CONTROL_1, mmSPI_INTERP_CONTROL_0, mmCB_SHADER_CONTROL, mmDB_SHADER_CONTROL,
		Latte::REGADDR::PA_CL_CLIP_CNTL, Latte::REGADDR::PA_CL_VTE_CNTL, mmSQ_PGM_START_PS, mmSQ_PGM_START_PS + 1,
		mmSQ_PGM_RESOURCES_PS, mmSQ_PGM_EXPORTS_PS, Latte::REGADDR::VGT_GS_MODE, mmVGT_GS_OUT_PRIM_TYPE, mmVGT_STRMOUT_EN,
		Latte::REGADDR::VGT_PRIMITIVE_TYPE, Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS,
		Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS + Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE * 7 - 1};
	int moved = 0;
	for (uint32 r : read)
	{
		regs[r] = 1;
		moved += LattePM4_PixelShaderInputs(regs.data()) != base;
		regs[r] = 0;
	}
	CHECK(moved == (int)(sizeof(read) / sizeof(read[0])));
	const uint32 notRead[] = {LATTE_REG_BASE_ALU_CONST, mmSQ_VTX_SEMANTIC_0, 0xA10F /* PA_CL_VPORT_XSCALE */,
		Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS + Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE * 7, LATTE_REG_BASE_SAMPLER};
	int still = 0;
	for (uint32 r : notRead)
	{
		regs[r] = 1;
		still += LattePM4_PixelShaderInputs(regs.data()) == base;
		regs[r] = 0;
	}
	CHECK(still == (int)(sizeof(notRead) / sizeof(notRead[0])));
}

static void VertexShaderInputs()
{
	std::vector<uint32> regs(LATTE_MAX_REGISTER, 0);
	const uint64 base = LattePM4_VertexShaderInputs(regs.data());
	const uint32 read[] = {mmSQ_VTX_SEMANTIC_0, mmSQ_VTX_SEMANTIC_0 + 31, mmSPI_VS_OUT_ID_0, mmSPI_PS_INPUT_CNTL_0 + 5,
		mmSPI_VS_OUT_CONFIG, mmSPI_PS_IN_CONTROL_0, mmSPI_INTERP_CONTROL_0, Latte::REGADDR::PA_CL_CLIP_CNTL,
		Latte::REGADDR::PA_CL_VTE_CNTL, mmPA_CL_VS_OUT_CNTL, mmSQ_PGM_START_VS, mmSQ_PGM_START_VS + 1, mmSQ_PGM_RESOURCES_VS,
		mmSQ_PGM_START_FS, mmSQ_PGM_START_FS + 1, mmSQ_PGM_RESOURCES_FS, mmSQ_GSVS_RING_ITEMSIZE, mmSQ_GS_VERT_ITEMSIZE,
		Latte::REGADDR::VGT_GS_MODE, mmVGT_GS_OUT_PRIM_TYPE, mmVGT_PRIMITIVEID_EN, Latte::REGADDR::VGT_INSTANCE_STEP_RATE_0,
		Latte::REGADDR::VGT_INSTANCE_STEP_RATE_0 + 1, mmVGT_STRMOUT_EN, mmVGT_STRMOUT_VTX_STRIDE_0, mmVGT_STRMOUT_VTX_STRIDE_0 + 12,
		Latte::REGADDR::VGT_PRIMITIVE_TYPE, Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_VS,
		Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_VS + Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE * 7 - 1};
	int moved = 0;
	for (uint32 r : read)
	{
		regs[r] = 1;
		moved += LattePM4_VertexShaderInputs(regs.data()) != base;
		regs[r] = 0;
	}
	CHECK(moved == (int)(sizeof(read) / sizeof(read[0])));
	// the vertex buffers' own registers (their addresses and strides, set per
	// draw), the pixel shaders' textures, the constants: not read
	const uint32 notRead[] = {LATTE_REG_BASE_ALU_CONST, mmSQ_VTX_ATTRIBUTE_BLOCK_START + 2, Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS,
		0xA10F /* PA_CL_VPORT_XSCALE */, Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_VS + Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE * 7};
	int still = 0;
	for (uint32 r : notRead)
	{
		regs[r] = 1;
		still += LattePM4_VertexShaderInputs(regs.data()) == base;
		regs[r] = 0;
	}
	CHECK(still == (int)(sizeof(notRead) / sizeof(notRead[0])));
}

// --- the hand-over -----------------------------------------------------------

struct Payload
{
	uint64 key;
	uint32 made;
};

static void SlotsOneThread()
{
	static AheadSlots<Payload, 8> slots;
	bool had;
	auto never = [](uint64, const Payload&) { return false; };
	auto a = slots.Claim(3, never, &had);
	CHECK(a && !had);
	CHECK(!slots.Claim(3, never, &had)); // in hand
	CHECK(!slots.Claim(11, never, &had)); // the same slot, another key's
	CHECK(!slots.Take(3)); // not published
	a.payload->key = 3;
	slots.Publish(a);
	CHECK(!slots.Claim(3, never, &had)); // published already
	CHECK(!slots.Take(11)); // another key's
	auto t = slots.Take(3);
	CHECK(t && t.payload->key == 3);
	CHECK(!slots.Take(3)); // taken once
	slots.Release(t);
	// a stale one claimed for another key, its payload still in it
	auto b = slots.Claim(5, never, &had);
	b.payload->key = 5;
	slots.Publish(b);
	CHECK(!slots.Claim(13, never, &had)); // published and still wanted: kept
	auto c = slots.Claim(13, [](uint64 key, const Payload& p) { return key == 5 && p.key == 5; }, &had);
	CHECK(c && had && c.payload->key == 5);
	CHECK(!slots.Take(5));
	slots.Abandon(c);
	CHECK(slots.Claim(13, never, &had)); // abandoned: free again
}

static void SlotsRacing()
{
	static AheadSlots<Payload, 64> slots;
	static SeenKeys<1 << 14> seen;
	constexpr uint64 kKeys = 400; // few, so that the consumer's guesses land
	std::atomic<int> published{0}, taken{0}, discarded{0}, wrong{0};
	std::vector<std::atomic<int>> made(kKeys + 1);
	std::atomic<bool> producersDone{false};
	std::vector<std::thread> producers;
	for (int t = 0; t < 4; t++)
	{
		producers.emplace_back([&, t] {
			std::mt19937 rng(77u + t);
			// until the consumer has taken 300 (not left to scheduling)
			for (int i = 0; i < 5000000 && taken.load(std::memory_order_relaxed) < 300; i++)
			{
				const uint64 key = 1 + rng() % kKeys;
				if (seen.Contains(key))
					continue;
				bool had;
				// stale: seen, or now and then old enough to go
				auto h = slots.Claim(key, [&rng](uint64 k, const Payload&) { return seen.Contains(k) || rng() % 8 == 0; }, &had);
				if (!h)
					continue;
				if (had)
					discarded++;
				h.payload->key = key;
				h.payload->made = (uint32)made[key].fetch_add(1) + 1;
				published++;
				slots.Publish(h);
			}
		});
	}
	std::thread consumer([&] {
		std::mt19937 rng(5u);
		while (!producersDone.load())
		{
			const uint64 key = 1 + rng() % kKeys;
			if (auto h = slots.Take(key))
			{
				if (h.payload->key != key)
					wrong++;
				taken++;
				if (key % 2)
					seen.Insert(key); // the draw's own copy now, never made ahead again
				slots.Release(h);
			}
		}
	});
	for (auto& p : producers)
		p.join();
	producersDone = true;
	consumer.join();
	slots.DrainPublished([&](Payload&) { discarded++; });
	CHECK(wrong == 0);
	CHECK(published == taken + discarded);
	std::printf("shader_ahead: %d made ahead by four threads, %d taken, %d let go\n", published.load(), taken.load(), discarded.load());
	CHECK(discarded > 0);
	CHECK(taken >= 300);
}

static void SeenKeysFill()
{
	static SeenKeys<1 << 12> keys;
	for (uint64 k = 0; k < 800; k++)
		keys.Insert(k);
	std::vector<std::thread> threads;
	for (int t = 0; t < 4; t++)
		threads.emplace_back([t] {
			for (uint64 k = 200; k < 3000; k++)
				keys.Insert(k * 4 + t);
		});
	for (auto& th : threads)
		th.join();
	int unseen = 0;
	// 12000 keys in 4096 places: they fill it; one that was added and fits is
	// never unseen, one never added never seen
	int fit = 0;
	for (uint64 k = 0; k < 12000; k++)
		fit += keys.Contains(k);
	for (uint64 k = 0; k < 800; k++)
		unseen += !keys.Contains(k);
	CHECK(fit <= 4097 && fit > 3000); // 0 and 1 the same key
	CHECK(unseen == 0); // the first 800, added before the rest, all fit
	int wrong = 0;
	for (uint64 k = 12000; k < 20000; k++)
		wrong += keys.Contains(k);
	CHECK(wrong == 0);
	SeenKeys<64> zero;
	zero.Insert(0);
	CHECK(zero.Contains(0));
}

int main()
{
	Registers();
	Ends();
	PixelShaderInputs();
	VertexShaderInputs();
	SlotsOneThread();
	SlotsRacing();
	SeenKeysFill();
	if (s_failures)
	{
		std::printf("shader_ahead_test: %d failure(s)\n", s_failures);
		return 1;
	}
	std::printf("shader_ahead_test: all passed\n");
	return 0;
}
