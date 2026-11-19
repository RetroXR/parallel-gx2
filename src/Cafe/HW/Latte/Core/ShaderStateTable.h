#pragma once

#include <cstddef>
#include <cstdint>

#include "util/helpers/u64_map.h"

// A shader state's aux variants, up to N held in place. Past N a later one
// takes the place of the oldest after the first (the first is the state's
// own shaders, which its keys are kept by): it is looked up again when it
// comes back, from the shader tables.
template<typename V, size_t N>
struct ShaderStateVariants
{
	static_assert(N >= 2, "the first and one more");

	V* begin() { return m_items; }
	V* end() { return m_items + m_count; }
	const V* begin() const { return m_items; }
	const V* end() const { return m_items + m_count; }
	size_t size() const { return m_count; }
	bool empty() const { return m_count == 0; }
	V& operator[](size_t i) { return m_items[i]; }

	// v kept; when every place is taken, the one it took is copied to
	// *replaced and true returned
	bool Add(const V& v, V* replaced)
	{
		if (m_count < N)
		{
			m_items[m_count++] = v;
			return false;
		}
		*replaced = m_items[m_next];
		m_items[m_next] = v;
		m_next = m_next + 1 < N ? m_next + 1 : 1;
		return true;
	}

	void Clear()
	{
		m_count = 0;
		m_next = 1;
	}

private:
	V m_items[N]{};
	size_t m_count{0};
	size_t m_next{1};
};

// The shader states seen, by the hash of their registers: up to three
// quarters of Slots of them, in place, so making and dropping them while a
// game plays allocates nothing; past that a state is not kept (and its
// shaders are looked up for each draw that sets it). The GPU thread's
// alone. Their keys are also in a dense list, which the cleanup walks a few
// at a time.
template<typename Info, size_t Slots>
class ShaderStateTable
{
	static_assert(Slots && (Slots & (Slots - 1)) == 0, "a power of two");

public:
	static constexpr size_t kMax = Slots / 4 * 3;

	ShaderStateTable() { Clear(); }

	Info* Find(uint64_t key) const
	{
		return static_cast<Info*>(u64_map_get(&m_map, key));
	}

	// a new state under key (not in it yet), its Info reset; nullptr when full
	Info* Add(uint64_t key)
	{
		if (m_count >= kMax)
			return nullptr;
		const uint32_t index = m_free[kMax - 1 - m_count];
		Entry& e = m_entries[index];
		e.info = Info{};
		e.dense = (uint32_t)m_count;
		if (!u64_map_put(&m_map, key, &e.info))
			return nullptr;
		m_keys[m_count] = key;
		m_index[m_count] = index;
		m_count++;
		return &e.info;
	}

	// key's state taken out; false when it was not in it
	bool Remove(uint64_t key)
	{
		Info* info = static_cast<Info*>(u64_map_take(&m_map, key));
		if (!info)
			return false;
		Entry* e = reinterpret_cast<Entry*>(info); // its first member
		const uint32_t index = (uint32_t)(e - m_entries);
		// the dense list's last into its place
		const uint32_t at = e->dense;
		const size_t last = m_count - 1;
		if (at != last)
		{
			m_keys[at] = m_keys[last];
			m_index[at] = m_index[last];
			m_entries[m_index[at]].dense = at;
		}
		m_count--;
		m_free[kMax - 1 - m_count] = index;
		return true;
	}

	size_t Count() const { return m_count; }
	bool Empty() const { return m_count == 0; }
	// the i'th key of the dense list (i < Count()): a removal moves the last
	// into the place it leaves
	uint64_t KeyAt(size_t i) const { return m_keys[i]; }

	void Clear()
	{
		for (auto& s : m_slots)
			s = {};
		u64_map_init(&m_map, m_slots, Slots);
		for (size_t i = 0; i < kMax; i++)
			m_free[i] = (uint32_t)(kMax - 1 - i);
		m_count = 0;
	}

private:
	struct Entry
	{
		Info info; // first: an Info's Entry is at its address
		uint32_t dense; // its place in m_keys
	};
	u64_map_t m_map{};
	u64_map_slot_t m_slots[Slots]{};
	Entry m_entries[kMax]{};
	uint64_t m_keys[kMax]{};
	uint32_t m_index[kMax]{};
	uint32_t m_free[kMax]{}; // the free entries' indices, from the end
	size_t m_count{0};
};
