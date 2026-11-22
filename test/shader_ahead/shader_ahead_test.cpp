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
//  - the ring read on past the buffer the GPU thread starts
//    (LattePM4_WholePackets, LattePM4Scan::called): whole packets only, the
//    buffers it calls counted, the registers carried from one buffer into
//    the next, and the buffers read to their end counted where the read
//    stops part-way; and where a read starts (LattePM4ReadTo): not at a
//    buffer a read got through, nor one a read under way may get to, at one
//    a read stopped short of; the furthest of four threads' reads kept;
//  - the idle GPU thread's registers handed to the guest's thread
//    (LattePM4IdleSnapshot): taken only as offered and with nothing
//    submitted since, never while filled, withdrawn once there is work, not
//    filled again while being read; raced by two threads, every snapshot
//    taken read whole (TSan in CI);
//  - the registers a pixel, vertex or geometry shader's decompile reads,
//    as one value (LattePM4_PixelShaderInputs, LattePM4_VertexShaderInputs,
//    LattePM4_GeometryShaderInputs): moved by each of them, not by the
//    vertex buffers, constants or the other stages' textures (that none
//    else moves a decompile: test/decompiler); and those registers copied
//    into a file of nothing else (LattePM4_CopyShaderInputs, a decompile
//    task's) keep the value;
//  - the keys of a geometry shader's draw (LatteShaderKeys.h): its pixel
//    shader's taking the draw's geometry shader, its geometry shader's the
//    ring of the draw's vertex shader, whatever was active before; one a
//    shader cache holds under the key of before found by it, a geometry
//    shader only where made for the draw's ring;
//  - the hand-over (AheadSlots.h): a key's slot claimed once, published,
//    taken once by that key only, a stale one claimed back for another key,
//    one never published left to the draw, one taken by its slot (to queue
//    its parts) or by a draw that cannot use it published again; four producers and a consumer
//    racing over 20000 keys, every payload published either taken or
//    drained, none twice (TSan in CI); the filter of keys seen, which
//    never says unseen for a key added; and the registered shaders'
//    table (PublishedMap): a key's first value found, none for a key never
//    added or past the table's room, one thread adding while three find -
//    never a value another key's, every key found once added (TSan in CI);
//    and a payload never claimed back while a pipeline is made from it
//    (AheadSlotStale);
//  - what new pipelines' first draws met, as logged at title stop
//    (LatteDrawPathCounts.h): each outcome and those a read queued ahead
//    counted apart, only waits timed, the longest kept, the line cut to
//    its room and nothing logged with nothing counted.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "Cafe/HW/Latte/Core/LattePM4Scan.h"
#include "Cafe/HW/Latte/Core/LattePM4.h"
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/Core/LatteShaderKeys.h"
#include "Cafe/HW/Latte/Core/LatteDrawPathCounts.h"
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

static void Ring()
{
	std::fill(s_regs.begin(), s_regs.end(), 0u);
	// the buffer the GPU thread starts, and two the guest submitted since
	Buf a;
	a.SetRegs(IT_SET_CONTEXT_REG, LATTE_REG_BASE_CONTEXT, mmCB_SHADER_MASK, {0xF});
	a.Draw();
	Buf b;
	b.SetRegs(IT_SET_CONTEXT_REG, LATTE_REG_BASE_CONTEXT, mmSQ_PGM_START_PS, {0x111, 0x20});
	b.Draw();
	Buf c;
	c.Draw();
	const uint32 bAt = Place(b, 0x600), cAt = Place(c, 0x700);
	// the ring as TCLSubmitToRing writes it: each call, the read pointer's
	// write and the retire marker; the last packet only part written
	Buf ring;
	for (uint32 at : {bAt, cAt})
	{
		ring.Packet(IT_INDIRECT_BUFFER_PRIV, {at, 0, at == bAt ? (uint32)b.w.size() : (uint32)c.w.size()});
		ring.Packet(IT_MEM_WRITE, {0x1000 | 2, 0x40000, 0x2000, 0});
		ring.Packet(IT_EVENT_WRITE_EOP, {4 | (5 << 8), 0x3000, 0x40000000, 1, 0});
	}
	const uint32 whole = (uint32)ring.w.size();
	ring.w.push_back(pm4HeaderType3(IT_INDIRECT_BUFFER_PRIV, 3));
	ring.w.push_back(cAt);
	uint32 calls = 99;
	CHECK(LattePM4_WholePackets(ring.w.data(), (uint32)ring.w.size(), &calls) == whole);
	CHECK(calls == 2);
	CHECK(LattePM4_WholePackets(ring.w.data(), 3, &calls) == 0 && calls == 0);

	Seen seen;
	seen.watch = {mmCB_SHADER_MASK, mmSQ_PGM_START_PS};
	LattePM4Scan scan{};
	scan.regs = s_regs.data();
	scan.map = MapGuest;
	scan.onDraw = OnDraw;
	scan.ctx = &seen;
	LattePM4_Scan(scan, a.w.data(), (uint32)a.w.size());
	CHECK(scan.end == LattePM4Scan::End::Done && scan.called == 0);
	LattePM4_Scan(scan, ring.w.data(), whole);
	CHECK(scan.end == LattePM4Scan::End::Done && scan.called == 2);
	// a's mask carried into b and c, b's program into c
	CHECK(seen.at.size() == 3);
	CHECK(seen.at.size() == 3 && seen.at[1] == (std::vector<uint32>{0xF, 0x111}) && seen.at[2] == (std::vector<uint32>{0xF, 0x111}));

	// c waits on memory: only b read to its end
	Buf cw;
	cw.Packet(IT_WAIT_REG_MEM, {0, 0, 0, 0, 0, 0});
	cw.Draw();
	Place(cw, 0x700);
	Buf ring2;
	ring2.Packet(IT_INDIRECT_BUFFER_PRIV, {bAt, 0, (uint32)b.w.size()});
	ring2.Packet(IT_INDIRECT_BUFFER_PRIV, {cAt, 0, (uint32)cw.w.size()});
	Seen seen2;
	scan.ctx = &seen2;
	LattePM4_Scan(scan, ring2.w.data(), (uint32)ring2.w.size());
	CHECK(scan.end == LattePM4Scan::End::Waited && scan.called == 1 && seen2.at.size() == 1);
	// a buffer b calls in turn is not one the ring calls
	Buf inner;
	inner.Draw();
	const uint32 innerAt = Place(inner, 0x800);
	Buf outer;
	outer.Packet(IT_INDIRECT_BUFFER_PRIV, {innerAt, 0, (uint32)inner.w.size()});
	const uint32 outerAt = Place(outer, 0x900);
	Buf ring3;
	ring3.Packet(IT_INDIRECT_BUFFER_PRIV, {outerAt, 0, (uint32)outer.w.size()});
	Seen seen3;
	scan.ctx = &seen3;
	LattePM4_Scan(scan, ring3.w.data(), (uint32)ring3.w.size());
	CHECK(scan.end == LattePM4Scan::End::Done && scan.called == 1 && seen3.at.size() == 1);
}

static void ReadTo()
{
	LattePM4ReadTo r;
	bool start = false;
	CHECK(r.Next(false, start) == 1 && start);
	r.Planned(1, 3); // the ring held three buffers more
	CHECK(r.Next(true, start) == 2 && !start); // the read may get here
	r.ReadTo(3);
	CHECK(r.Next(false, start) == 3 && !start); // read, registers carried
	CHECK(r.Next(false, start) == 4 && start);
	r.Planned(4, 5);
	r.ReadTo(4); // stopped at a wait in buffer 5
	CHECK(r.Next(false, start) == 5 && start); // done: started again
	r.Planned(5, 0);
	CHECK(r.Next(true, start) == 6 && start); // past the plan under way
	r.ReadTo(2); // a slower read's: the furthest kept
	r.Reset();
	CHECK(r.Next(false, start) == 1 && start);
	// four threads' reads ending at once: the furthest kept
	LattePM4ReadTo shared;
	std::vector<std::thread> threads;
	for (uint32 t = 0; t < 4; t++)
		threads.emplace_back([&shared, t] {
			for (uint32 i = 1; i <= 5000; i++)
				shared.ReadTo(i * 4 + t);
		});
	for (auto& th : threads)
		th.join();
	for (uint32 i = 0; i < 20002; i++)
		shared.Next(false, start);
	CHECK(shared.Next(false, start) == 20003 && !start); // 5000 * 4 + 3
	CHECK(shared.Next(false, start) == 20004 && start);
}

static void IdleSnapshot()
{
	LattePM4IdleSnapshot s;
	CHECK(!s.Take(5)); // nothing offered
	CHECK(s.BeginFill());
	CHECK(!s.Take(5)); // being filled
	CHECK(!s.BeginFill());
	s.Offer(5);
	CHECK(!s.Take(6)); // submitted since: stale, and let go
	CHECK(!s.Take(5));
	CHECK(s.BeginFill());
	s.Offer(7);
	s.Withdraw(); // work again
	CHECK(!s.Take(7));
	CHECK(s.BeginFill());
	s.Offer(8);
	CHECK(s.Take(8));
	CHECK(!s.BeginFill()); // not filled while read
	s.Withdraw(); // nothing to withdraw while read
	CHECK(!s.BeginFill());
	s.Release();
	CHECK(s.BeginFill());
	s.Offer(9);

	// the GPU thread filling and withdrawing, the guest's thread taking:
	// every snapshot read whole
	LattePM4IdleSnapshot shared;
	std::vector<uint32> words(4096, 0);
	std::atomic<bool> done{false};
	std::atomic<uint32> torn{0}, taken{0};
	std::thread gpu([&] {
		// until the guest's thread took some, however the threads are run
		for (uint32 n = 1; (n <= 40000 || taken.load() < 64) && n < (1u << 26); n++)
		{
			if (!shared.BeginFill())
				continue;
			for (uint32& w : words)
				w = n;
			shared.Offer(n & 1);
			if (n % 3 == 0)
				shared.Withdraw();
		}
		done.store(true);
	});
	std::thread guest([&] {
		uint32 ring = 0;
		while (!done.load())
		{
			if (!shared.Take(ring))
			{
				ring ^= 1;
				continue;
			}
			const uint32 first = words[0];
			for (uint32 w : words)
				if (w != first)
					torn++;
			taken++;
			shared.Release();
		}
	});
	gpu.join();
	guest.join();
	std::printf("idle snapshot: %u taken by the guest's thread, %u torn\n", taken.load(), torn.load());
	CHECK(torn.load() == 0 && taken.load() > 0);
}

static void Stale()
{
	CHECK(!AheadSlotStale(false, false, 64, 64)); // waited, not past
	CHECK(AheadSlotStale(false, false, 65, 64));
	CHECK(AheadSlotStale(false, true, 0, 64)); // known another way
	// a pipeline being made from it: kept, known or waited
	CHECK(!AheadSlotStale(true, true, 0, 64));
	CHECK(!AheadSlotStale(true, false, 1000, 64));
	CHECK(!AheadSlotStale(true, true, 1000, 64));
}

static void Published()
{
	{
		PublishedMap<int, 64> m;
		int a = 1, b = 2;
		CHECK(m.Find(5) == nullptr);
		m.Insert(5, &a);
		m.Insert(5, &b); // the first kept
		CHECK(m.Find(5) == &a);
		m.Insert(0, &b); // 0 is kept as 1
		CHECK(m.Find(0) == &b && m.Find(1) == &b);
		// past its room: not found, never another's
		int values[200];
		for (int i = 0; i < 200; i++)
			m.Insert(1000 + i, &values[i]);
		int found = 0;
		for (int i = 0; i < 200; i++)
		{
			int* v = m.Find(1000 + i);
			CHECK(v == nullptr || v == &values[i]);
			found += v != nullptr;
		}
		CHECK(found > 0 && found <= 62); // two keys in already
		m.Clear();
		CHECK(m.Find(5) == nullptr);
	}
	// a key added again takes no more room: as many others fit after it
	{
		static PublishedMap<int, 64> once, again;
		int value = 0;
		once.Insert(77, &value);
		for (int i = 0; i < 100; i++)
			again.Insert(77, &value);
		int fitOnce = 0, fitAgain = 0;
		for (int i = 0; i < 200; i++)
		{
			once.Insert(5000 + i, &value);
			again.Insert(5000 + i, &value);
			fitOnce += once.Find(5000 + i) != nullptr;
			fitAgain += again.Find(5000 + i) != nullptr;
		}
		CHECK(fitAgain == fitOnce);
	}
	// one thread adding, three finding
	static PublishedMap<uint64, 1u << 14> shared;
	static uint64 values[8000];
	for (uint64 i = 0; i < 8000; i++)
		values[i] = i * 7 + 3;
	std::atomic<uint32> added{0};
	std::atomic<uint32> wrong{0};
	std::vector<std::thread> readers;
	for (int r = 0; r < 3; r++)
		readers.emplace_back([&] {
			while (added.load(std::memory_order_acquire) < 8000)
			{
				const uint32 upTo = added.load(std::memory_order_acquire);
				for (uint32 i = 0; i < 8000; i += 37)
				{
					const uint64* v = shared.Find(0x100000 + i);
					if (v && *v != i * 7 + 3)
						wrong++;
					if (i < upTo && !v)
						wrong++; // added before: found
				}
			}
		});
	for (uint32 i = 0; i < 8000; i++)
	{
		shared.Insert(0x100000 + i, &values[i]);
		added.store(i + 1, std::memory_order_release);
	}
	for (auto& t : readers)
		t.join();
	CHECK(wrong.load() == 0);
	uint32 all = 0;
	for (uint32 i = 0; i < 8000; i++)
		all += shared.Find(0x100000 + i) == &values[i];
	CHECK(all == 8000);
	shared.Clear();
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

static void GeometryShaderInputs()
{
	std::vector<uint32> regs(LATTE_MAX_REGISTER, 0);
	const uint64 base = LattePM4_GeometryShaderInputs(regs.data());
	const uint32 read[] = {mmCB_SHADER_MASK, mmSPI_VS_OUT_ID_0 + 9, mmSPI_PS_INPUT_CNTL_0 + 31, mmSPI_VS_OUT_CONFIG,
		mmSPI_PS_IN_CONTROL_0, mmSPI_INTERP_CONTROL_0, Latte::REGADDR::PA_CL_CLIP_CNTL, Latte::REGADDR::PA_CL_VTE_CNTL,
		mmPA_CL_VS_OUT_CNTL, mmSQ_PGM_START_VS, mmSQ_PGM_START_VS + 1, mmSQ_PGM_START_GS, mmSQ_PGM_START_GS + 1,
		mmSQ_GSVS_RING_ITEMSIZE, mmSQ_GS_VERT_ITEMSIZE, Latte::REGADDR::VGT_GS_MODE, mmVGT_GS_OUT_PRIM_TYPE, mmVGT_PRIMITIVEID_EN,
		mmVGT_STRMOUT_EN, mmVGT_STRMOUT_VTX_STRIDE_0 + 12, Latte::REGADDR::VGT_PRIMITIVE_TYPE, Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_GS,
		Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_GS + Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE * 7 - 1};
	int moved = 0;
	for (uint32 r : read)
	{
		regs[r] = 1;
		moved += LattePM4_GeometryShaderInputs(regs.data()) != base;
		regs[r] = 0;
	}
	CHECK(moved == (int)(sizeof(read) / sizeof(read[0])));
	// what feeds it (the export shader, the fetch shader, the vertex
	// semantics), the other stages' textures, the constants: not read
	const uint32 notRead[] = {LATTE_REG_BASE_ALU_CONST, mmSQ_PGM_START_ES, mmSQ_PGM_START_FS, mmSQ_VTX_SEMANTIC_0,
		Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_VS, Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS, 0xA10F /* PA_CL_VPORT_XSCALE */};
	int still = 0;
	for (uint32 r : notRead)
	{
		regs[r] = 1;
		still += LattePM4_GeometryShaderInputs(regs.data()) == base;
		regs[r] = 0;
	}
	CHECK(still == (int)(sizeof(notRead) / sizeof(notRead[0])));
}

static void CopiedInputs()
{
	std::mt19937 rng(139);
	std::vector<uint32> from(LATTE_MAX_REGISTER), to(LATTE_MAX_REGISTER);
	for (uint32& r : from)
		r = (uint32)rng();
	const struct { LattePM4Inputs kind; uint64 (*digest)(const uint32*); } kinds[] = {
		{LattePM4Inputs::Pixel, LattePM4_PixelShaderInputs},
		{LattePM4Inputs::Vertex, LattePM4_VertexShaderInputs},
		{LattePM4Inputs::Geometry, LattePM4_GeometryShaderInputs},
	};
	for (const auto& k : kinds)
	{
		std::fill(to.begin(), to.end(), 0u);
		LattePM4_CopyShaderInputs(k.kind, to.data(), from.data());
		CHECK(k.digest(to.data()) == k.digest(from.data()));
		// nothing else copied: the constants and the other stages' textures
		CHECK(to[LATTE_REG_BASE_ALU_CONST] == 0 && to[mmSQ_VTX_BASE_VTX_LOC] == 0);
		CHECK(k.kind == LattePM4Inputs::Pixel || to[Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS] == 0);
		CHECK(k.kind != LattePM4Inputs::Pixel || to[Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_VS] == 0);
	}
}

// --- the keys of a geometry shader's draw ------------------------------------

struct KeyedShader
{
	uint32 ringParameterCountFromPrevStage;
};

static void Keys()
{
	// a pixel shader after a geometry shader: keyed by the draw's geometry
	// shader, as decompiled ahead; with none, the geometry shader left out
	const uint64 ps1 = 0x1111, ps2 = 0x2222, imports = 0x30, gsA = 0xA000, gsB = 0xB000;
	CHECK(LatteSHRC_PixelBaseHash(ps1, ps2, imports, true, gsA) != LatteSHRC_PixelBaseHash(ps1, ps2, imports, true, gsB));
	CHECK(LatteSHRC_PixelBaseHash(ps1, ps2, imports, false, gsA) == LatteSHRC_PixelBaseHash(ps1, ps2, imports, false, gsB));
	std::vector<std::pair<uint64, KeyedShader*>> cache;
	std::vector<uint64> asked;
	auto find = [&](uint64 key) -> KeyedShader* {
		asked.push_back(key);
		for (auto& e : cache)
			if (e.first == key)
				return e.second;
		return nullptr;
	};
	KeyedShader pixel{0};
	// held under the key of the geometry shader hashed before (gsA) the
	// draw's (gsB): found by it, after its own key
	cache.push_back({LatteSHRC_PixelBaseHash(ps1, ps2, imports, true, gsA), &pixel});
	CHECK((LatteSHRC_FindPixelShaderFor<KeyedShader>(find, ps1, ps2, imports, true, gsB, gsA)) == &pixel);
	CHECK(asked.size() == 2 && asked[0] == LatteSHRC_PixelBaseHash(ps1, ps2, imports, true, gsB));
	// held under the draw's own: found at once
	asked.clear();
	CHECK((LatteSHRC_FindPixelShaderFor<KeyedShader>(find, ps1, ps2, imports, true, gsA, gsB)) == &pixel);
	CHECK(asked.size() == 1);
	// no geometry shader: its own key only, the geometry shaders' hashes
	// out of it
	asked.clear();
	CHECK((LatteSHRC_FindPixelShaderFor<KeyedShader>(find, ps1, ps2, imports, false, gsB, gsA)) == nullptr);
	CHECK(asked.size() == 1 && asked[0] == LatteSHRC_PixelBaseHash(ps1, ps2, imports, false, 0));
	// the same geometry shader hashed before: one look
	asked.clear();
	CHECK((LatteSHRC_FindPixelShaderFor<KeyedShader>(find, ps1, ps2, imports, true, gsB, gsB)) == nullptr);
	CHECK(asked.size() == 1);

	// a geometry shader: keyed by the ring of the draw's vertex shader
	const uint64 g1 = 0x5000, g2 = 0x6000;
	CHECK(LatteSHRC_GeometryBaseHash(g1, g2, 4, false) != LatteSHRC_GeometryBaseHash(g1, g2, 6, false));
	CHECK(LatteSHRC_GeometryBaseHash(g1, g2, 4, false) != LatteSHRC_GeometryBaseHash(g1, g2, 4, true));
	cache.clear();
	KeyedShader madeFor4{4}, madeFor6{6};
	// the draw's ring 4, the vertex shader active before writing 6: one
	// held under 6's key made for 4 taken, as a shader cache keyed it before
	cache.push_back({LatteSHRC_GeometryBaseHash(g1, g2, 6, false), &madeFor4});
	CHECK((LatteSHRC_FindGeometryShaderFor<KeyedShader>(find, g1, g2, 4, 6, false)) == &madeFor4);
	// one made for 6 is not, for a draw whose ring is 4
	cache[0].second = &madeFor6;
	CHECK((LatteSHRC_FindGeometryShaderFor<KeyedShader>(find, g1, g2, 4, 6, false)) == nullptr);
	// none active before (the first draw): its own key, looked up
	asked.clear();
	CHECK((LatteSHRC_FindGeometryShaderFor<KeyedShader>(find, g1, g2, 4, -1, false)) == nullptr);
	CHECK(asked.size() == 1 && asked[0] == LatteSHRC_GeometryBaseHash(g1, g2, 4, false));
	// its own key first: made for the draw's ring, whatever came before
	cache.push_back({LatteSHRC_GeometryBaseHash(g1, g2, 4, false), &madeFor4});
	asked.clear();
	CHECK((LatteSHRC_FindGeometryShaderFor<KeyedShader>(find, g1, g2, 4, 6, false)) == &madeFor4);
	CHECK(asked.size() == 1);
	// the vertex shader before writing the same ring: one look, found or not
	cache.erase(cache.begin() + 1);
	asked.clear();
	CHECK((LatteSHRC_FindGeometryShaderFor<KeyedShader>(find, g1, g2, 6, 6, false)) == &madeFor6);
	CHECK(asked.size() == 1);
	asked.clear();
	CHECK((LatteSHRC_FindGeometryShaderFor<KeyedShader>(find, g1, g2, 5, 5, false)) == nullptr);
	CHECK(asked.size() == 1);
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
	auto d = slots.Claim(13, never, &had); // abandoned: free again
	CHECK(d);
	// taken by its slot, as the GPU thread does to queue a shader's parts:
	// only once published, its key with it, and published again after
	CHECK(!slots.TakeAt(AheadSlots<Payload, 8>::SlotOf(13))); // still being made
	CHECK(slots.InHand(13) && !slots.InHand(5) && !slots.InHand(21)); // its draw, come first, counts it late
	d.payload->key = 13;
	slots.Publish(d);
	auto e = slots.TakeAt(AheadSlots<Payload, 8>::SlotOf(13));
	CHECK(e && slots.KeyOf(e) == 13);
	CHECK(!slots.TakeAt(AheadSlots<Payload, 8>::SlotOf(13))); // held
	CHECK(!slots.Take(13));
	slots.Publish(e);
	// taken by key, not what the draw needs: put back, there to take again
	auto f = slots.Take(13);
	CHECK(f);
	slots.Publish(f);
	auto g = slots.Take(13);
	CHECK(g && g.payload->key == 13);
	slots.Release(g);
	CHECK(!slots.TakeAt(AheadSlots<Payload, 8>::SlotOf(13))); // released: empty
	CHECK(!slots.InHand(13));
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

static void DrawPathCounts()
{
	LatteDrawPathCounts c;
	char line[512];
	CHECK(!c.Any());
	CHECK(!c.Describe(line, sizeof(line)));
	c.NotePipeline(LatteDrawPathCounts::Ready, true, 999); // not a wait
	c.NotePipeline(LatteDrawPathCounts::Ready, false, 0);
	c.NotePipeline(LatteDrawPathCounts::Skipped, false, 0);
	c.NotePipeline(LatteDrawPathCounts::Waited, true, 2500);
	c.NoteAwait(4000);
	c.NotePipeline(LatteDrawPathCounts::Waited, false, 700); // shorter, after
	for (int i = 0; i < 3; i++)
		c.NoteSkippedDraw();
	CHECK(c.pipelines[LatteDrawPathCounts::Ready] == 2 && c.ahead[LatteDrawPathCounts::Ready] == 1);
	CHECK(c.pipelines[LatteDrawPathCounts::Skipped] == 1 && c.ahead[LatteDrawPathCounts::Skipped] == 0);
	CHECK(c.pipelines[LatteDrawPathCounts::Waited] == 2 && c.ahead[LatteDrawPathCounts::Waited] == 1);
	CHECK(c.skippedDraws == 3 && c.awaited == 1);
	CHECK(c.waitedUs == 7200 && c.longestUs == 4000);
	CHECK(c.Describe(line, sizeof(line)));
	CHECK(std::strcmp(line, "new pipelines at their first draw: 2 ready (1 queued ahead), 1 skipped until made (0 queued ahead), "
		"2 made while their draw waited (1 queued ahead); 3 draws skipped, 1 waited for a compile thread; "
		"7.200 ms waited, the longest 4.000 ms") == 0);
	// cut to its room, nothing written past it
	char small[24];
	std::memset(small, 'x', sizeof(small));
	CHECK(c.Describe(small, 16));
	CHECK(std::strlen(small) == 15 && small[16] == 'x');
	// anything counted alone is told
	LatteDrawPathCounts only;
	only.NoteSkippedDraw();
	CHECK(only.Any());
	LatteDrawPathCounts awaitOnly;
	awaitOnly.NoteAwait(1);
	CHECK(awaitOnly.Any() && awaitOnly.longestUs == 1);
	c.Reset();
	CHECK(!c.Any() && c.waitedUs == 0 && c.longestUs == 0);
}

int main()
{
	Registers();
	Ends();
	Ring();
	ReadTo();
	IdleSnapshot();
	Published();
	Stale();
	PixelShaderInputs();
	VertexShaderInputs();
	GeometryShaderInputs();
	CopiedInputs();
	Keys();
	SlotsOneThread();
	SlotsRacing();
	SeenKeysFill();
	DrawPathCounts();
	if (s_failures)
	{
		std::printf("shader_ahead_test: %d failure(s)\n", s_failures);
		return 1;
	}
	std::printf("shader_ahead_test: all passed\n");
	return 0;
}
