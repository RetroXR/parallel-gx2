#pragma once

#include <utility>

#include "util/helpers/thread_end.h"

// The calling thread's own T: made with args at its first use, deleted as
// the thread ends (thread_end.h) - not a thread_local T, whose destructor
// MinGW runs on storage already freed. slot is the caller's thread_local
// pointer to it, trivially destroyed. With no room on the thread's list the
// T is kept past the thread's end rather than freed.
template<typename T, typename... Args>
inline T& ThreadOwned(T*& slot, Args&&... args)
{
	if (!slot) [[unlikely]]
	{
		slot = new T(std::forward<Args>(args)...);
		thread_end_register([](void* o) { delete static_cast<T*>(o); }, slot, reinterpret_cast<void**>(&slot));
	}
	return *slot;
}
