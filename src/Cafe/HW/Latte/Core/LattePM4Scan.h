#pragma once

#include <atomic>

// A command buffer read ahead of the command processor, on any thread: its
// register writes applied to a register file of the reader's own, as the
// command processor applies them to the GPU's (LatteCommandProcessor.cpp),
// the buffers it calls followed, and the reader told of every draw with the
// registers as they are at it. Nothing in it waits, writes guest memory or
// touches the GPU's state: a buffer is read until its end, a draw the reader
// stops at, or the first point where what follows may not have been written
// yet (a wait on memory, a semaphore) or cannot be read (a packet header it
// does not know).
struct LattePM4Scan
{
	// a whole register file (LATTE_MAX_REGISTER), the reader's
	uint32* regs;
	// count words of guest memory at a physical address, big-endian as they
	// are there; null where they cannot be read
	const uint32be* (*map)(void* ctx, uint32 physAddr, uint32 count);
	// a draw, with regs as they are at it; false stops the scan
	bool (*onDraw)(void* ctx, uint32* regs);
	void* ctx;
	// buffers called from buffers, at most
	uint32 maxDepth = 4;
	// packets read, all buffers together, at most
	uint32 maxPackets = 1u << 20;

	// out: draws seen, the buffers the words called themselves read to
	// their end, and why the scan ended
	uint32 draws = 0;
	uint32 called = 0;
	enum class End { Done, Stopped, Waited, Unreadable, Limit } end = End::Done;
};

void LattePM4_Scan(LattePM4Scan& scan, const uint32be* words, uint32 count);

// Of count words of a command stream (the ring's, LatteShaderAhead.cpp),
// how many make whole packets from the start, the buffers those call
// counted in calls
uint32 LattePM4_WholePackets(const uint32be* words, uint32 count, uint32* calls);

// Which of the buffers the ring calls a read ahead starts at: numbered as
// the GPU thread starts them, none that a read got through to the end of
// (the registers carried there), nor one a read under way may still get
// to; any other, once a read stopped short of it.
struct LattePM4ReadTo
{
	// GPU thread, as it starts a buffer: its number, start set where a read
	// is to start here (planRunning: the last read planned is under way)
	uint32 Next(bool planRunning, bool& start)
	{
		const uint32 seq = ++m_seq;
		start = seq > m_readTo.load(std::memory_order_acquire) && !(planRunning && seq <= m_plannedTo);
		return seq;
	}
	// GPU thread: a read started at buffer seq, the ring holding calls more
	void Planned(uint32 seq, uint32 calls) { m_plannedTo = seq + calls; }
	// a read's: through to the end of buffer to
	void ReadTo(uint32 to)
	{
		for (uint32 cur = m_readTo.load(std::memory_order_relaxed); to > cur &&
			!m_readTo.compare_exchange_weak(cur, to, std::memory_order_release, std::memory_order_relaxed);)
			;
	}
	void Reset()
	{
		m_seq = 0;
		m_plannedTo = 0;
		m_readTo.store(0, std::memory_order_relaxed);
	}

private:
	uint32 m_seq = 0;
	uint32 m_plannedTo = 0;
	std::atomic<uint32> m_readTo{0};
};

// Every register a pixel shader's decompile and its hashes read, as one
// value: equal where the decompile would be (LatteShaderAhead.cpp takes a
// shader decompiled ahead only then)
uint64 LattePM4_PixelShaderInputs(const uint32* regs);
// the same for a vertex shader's (its fetch shader's parse included), the
// export shader's where a geometry shader runs
uint64 LattePM4_VertexShaderInputs(const uint32* regs);
// and for a geometry shader's, its copy shader's registers included
uint64 LattePM4_GeometryShaderInputs(const uint32* regs);

// Those registers copied, from one register file into another: a decompile
// from the copy is the one from the whole file (test/decompiler)
enum class LattePM4Inputs { Pixel, Vertex, Geometry };
void LattePM4_CopyShaderInputs(LattePM4Inputs kind, uint32* to, const uint32* from);
