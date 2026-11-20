#pragma once

#include <vector>
#include "util/helpers/ThreadOwned.h"

// Storage the recompiler reuses from one compile to the next on the thread
// that compiles, so a compile allocates nothing once the first few have grown
// it: a vector taken empty with the capacity the last one left, and given
// back when done (the larger of the two is kept). Tag tells apart two
// scratch vectors of one type.
template<typename T, int Tag = 0>
struct IMLScratchVector
{
	static std::vector<T> Take()
	{
		std::vector<T> v = std::move(Spare());
		v.clear();
		return v;
	}

	static void Give(std::vector<T>&& v)
	{
		if (v.capacity() > Spare().capacity())
			Spare() = std::move(v);
	}

private:
	static std::vector<T>& Spare()
	{
		thread_local std::vector<T>* t_spare;
		return ThreadOwned(t_spare);
	}
};

struct IMLSegment;
// a segment, recycled: empty, as new, its vectors keeping their capacity
IMLSegment* IMLSegment_New();
// back to the compiling thread's pool
void IMLSegment_Free(IMLSegment* segment);

// a scratch vector for a scope: taken on the way in, given back on the way out
template<typename T, int Tag = 0>
struct IMLScratch
{
	IMLScratch() : v(IMLScratchVector<T, Tag>::Take()) {}
	~IMLScratch() { IMLScratchVector<T, Tag>::Give(std::move(v)); }
	IMLScratch(const IMLScratch&) = delete;
	IMLScratch& operator=(const IMLScratch&) = delete;
	std::vector<T> v;
};
