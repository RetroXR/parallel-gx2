#pragma once

#include <cstddef>
#include <cstdint>

// A first-in first-out queue of up to N items in place, for one thread: what
// std::queue (a deque, a block allocated for every few dozen items as it
// cycles) was used for where the items in it at once are bounded. N a power
// of two.
template<typename T, size_t N>
class RingQueue
{
	static_assert(N && (N & (N - 1)) == 0, "a power of two");

public:
	bool empty() const { return m_head == m_tail; }
	size_t size() const { return (size_t)(m_tail - m_head); }
	bool full() const { return size() == N; }

	T& front() { return m_items[m_head & (N - 1)]; }
	const T& front() const { return m_items[m_head & (N - 1)]; }
	T& back() { return m_items[(m_tail - 1) & (N - 1)]; }

	// false when full, the item not queued
	bool push(const T& item)
	{
		if (full())
			return false;
		m_items[m_tail++ & (N - 1)] = item;
		return true;
	}

	// the next place, queued as it was left (an item holding a buffer keeps
	// its room): not full
	T& push_slot() { return m_items[m_tail++ & (N - 1)]; }

	void pop() { m_head++; }

private:
	T m_items[N]{};
	uint64_t m_head{0};
	uint64_t m_tail{0};
};
