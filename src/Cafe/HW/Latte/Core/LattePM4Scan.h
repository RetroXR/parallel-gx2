#pragma once

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

	// out: draws seen, and why the scan ended
	uint32 draws = 0;
	enum class End { Done, Stopped, Waited, Unreadable, Limit } end = End::Done;
};

void LattePM4_Scan(LattePM4Scan& scan, const uint32be* words, uint32 count);

// Every register a pixel shader's decompile and its hashes read (no
// geometry shader), as one value: equal where the decompile would be
// (LatteShaderAhead.cpp takes a shader decompiled ahead only then)
uint64 LattePM4_PixelShaderInputs(const uint32* regs);
