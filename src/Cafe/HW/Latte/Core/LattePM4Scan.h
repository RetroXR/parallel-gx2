#pragma once

#include <atomic>

// A command buffer read ahead of the command processor, on any thread: its
// register writes applied to a register file of the reader's own, as the
// command processor applies them to the GPU's (LatteCommandProcessor.cpp),
// the buffers it calls followed, and the reader told of every draw with the
// registers as they are at it. Nothing in it waits, writes guest memory or
// touches the GPU's state: a buffer is read until its end, a draw the reader
// stops at, or the first point where what follows may not have been written
// yet or cannot be read (a packet header it does not know). A wait is such a
// point only where the command processor would wait there now: a wait on
// memory whose test fails on what memory holds (LattePM4_WaitPasses), a
// semaphore wait with no count left past the waits read before it in the
// scan, a flip or a sync with the asynchronous commands. A wait on a
// register, which the command processor passes, and a semaphore's signal
// are read past.
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
	// semaphore waits read past, all semaphores together: carried from one
	// scan to the next with the same registers, as the reads of the ring do
	uint32 semaphoreWaits = 0;
	// packets read, all buffers together, at most
	uint32 maxPackets = 1u << 20;

	// out: draws seen, the buffers the words called themselves read to
	// their end, and why the scan ended
	uint32 draws = 0;
	uint32 called = 0;
	enum class End { Done, Stopped, Waited, Unreadable, Limit } end = End::Done;
};

void LattePM4_Scan(LattePM4Scan& scan, const uint32be* words, uint32 count);

// Whether a wait on memory (IT_WAIT_REG_MEM) is passed with value, already
// masked, against reference, by its compare function (the packet's low three
// bits): the command processor's test, and the read ahead's
bool LattePM4_WaitPasses(uint32 compareOp, uint32 value, uint32 reference);

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

// The registers the buffer the guest is still writing starts from, handed
// to the guest's thread for a read of it: filled and offered by the GPU
// thread as it goes idle with nothing in the ring (its registers then), or
// by a read ahead that got through to the end of what the ring held (its
// registers there, where the GPU thread is still behind), the newer
// replacing an offer not taken; withdrawn by the GPU thread as it gets
// work. The guest's thread takes it only while it has submitted nothing
// since (the ring's write index the snapshot is for), reads it and gives it
// back offered, to read its buffer again once it has written more.
// Lock-free; nobody ever waits for anybody.
struct LattePM4IdleSnapshot
{
	// GPU thread or a read: true where the snapshot is its to fill (then
	// Offer): one not offered, or offered and not taken
	bool BeginFill()
	{
		uint32 cur = m_state.load(std::memory_order_relaxed);
		while (cur == kFree || cur == kOffered)
		{
			if (m_state.compare_exchange_weak(cur, kFilling, std::memory_order_acquire, std::memory_order_relaxed))
				return true;
		}
		return false;
	}
	void Offer(uint32 ringAt)
	{
		m_ringAt = ringAt;
		m_state.store(kOffered, std::memory_order_release);
	}
	// GPU thread, with work again
	void Withdraw()
	{
		uint32 expected = kOffered;
		m_state.compare_exchange_strong(expected, kFree, std::memory_order_relaxed, std::memory_order_relaxed);
	}
	// guest's thread, the ring's write index ringAt: true where the snapshot
	// is its to read (then Release)
	bool Take(uint32 ringAt)
	{
		uint32 expected = kOffered;
		if (!m_state.compare_exchange_strong(expected, kTaken, std::memory_order_acquire, std::memory_order_relaxed))
			return false;
		if (m_ringAt != ringAt)
		{
			m_state.store(kFree, std::memory_order_release); // submitted since: stale
			return false;
		}
		return true;
	}
	// read: offered again, for the next read of the same buffer
	void Release() { m_state.store(kOffered, std::memory_order_release); }

private:
	static constexpr uint32 kFree = 0, kFilling = 1, kOffered = 2, kTaken = 3;
	std::atomic<uint32> m_state{kFree};
	uint32 m_ringAt = 0;
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
