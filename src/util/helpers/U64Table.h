#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "util/helpers/u64_map.h"

// Objects by a 64-bit key (u64_map.c), for one thread: sized up front
// (Reserve, at load), and grown - its slots doubled and every entry put
// again - only past three quarters of the most it has held, so finding,
// adding and taking out allocate nothing while a game stays within what it
// has already reached.
template<typename T>
class U64Table
{
public:
	U64Table() { Reserve(16); }
	U64Table(const U64Table&) = delete;
	U64Table& operator=(const U64Table&) = delete;

	// room for three quarters of slots (rounded up to a power of two) at once
	void Reserve(size_t slots)
	{
		size_t cap = 16;
		while (cap < slots)
			cap <<= 1;
		if (cap > m_slots.size())
			Rehash(cap);
	}

	T* Find(uint64_t key) const { return static_cast<T*>(u64_map_get(&m_map, key)); }

	// obj under key, in place of what it was
	void Put(uint64_t key, T* obj)
	{
		if (!u64_map_put(&m_map, key, obj))
		{
			Rehash(m_slots.size() * 2);
			u64_map_put(&m_map, key, obj);
		}
	}

	// key out, if obj is what it has
	void Remove(uint64_t key, const T* obj)
	{
		if (Find(key) == obj)
			u64_map_take(&m_map, key);
	}

	size_t Count() const { return m_map.count; }
	size_t Capacity() const { return m_slots.size(); }

	void Clear() { u64_map_clear(&m_map); }

private:
	void Rehash(size_t cap)
	{
		std::vector<u64_map_slot_t> old;
		old.swap(m_slots);
		m_slots.assign(cap, u64_map_slot_t{});
		u64_map_init(&m_map, m_slots.data(), cap);
		for (const u64_map_slot_t& s : old)
			if (s.value)
				u64_map_put(&m_map, s.key, s.value);
	}

	std::vector<u64_map_slot_t> m_slots;
	u64_map_t m_map{};
};
