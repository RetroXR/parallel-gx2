#pragma once

#include <array>
#include <bit>
#include <new>
#include <type_traits>
#include <tuple>
#include <utility>

#include "IMLInstruction.h"

// A function's virtual registers are numbered densely from 0, one per
// register name it uses (PPCRecompilerImlGen_LookupReg), and a function
// using more than this is left to the interpreter. The sets and maps keyed by
// them below are a bit per register and a slot per register, in the object:
// the passes and the register allocator that used std::unordered_set and
// std::unordered_map allocated a node for nearly every insert, hundreds per
// compile. They iterate in register order.
inline constexpr size_t IML_REG_ID_CAPACITY = 256;

class IMLRegIdSet
{
	static constexpr size_t kWords = IML_REG_ID_CAPACITY / 64;

public:
	struct InsertResult
	{
		bool second; // newly inserted
	};

	class const_iterator
	{
	public:
		const_iterator(const IMLRegIdSet* set, size_t index) : m_set(set), m_index(index) { Settle(); }
		const IMLRegID& operator*() const { return m_id; }
		const_iterator& operator++()
		{
			m_index++;
			Settle();
			return *this;
		}
		bool operator==(const const_iterator& other) const { return m_index == other.m_index; }
		bool operator!=(const const_iterator& other) const { return m_index != other.m_index; }

	private:
		// on to the next register in the set, or the end
		void Settle()
		{
			while (m_index < IML_REG_ID_CAPACITY)
			{
				const uint64 word = m_set->m_bits[m_index / 64] >> (m_index % 64);
				if (word)
				{
					m_index += std::countr_zero(word);
					m_id = (IMLRegID)m_index;
					return;
				}
				m_index = (m_index / 64 + 1) * 64;
			}
			m_index = IML_REG_ID_CAPACITY;
		}

		const IMLRegIdSet* m_set;
		size_t m_index;
		IMLRegID m_id{};
	};

	InsertResult insert(IMLRegID id)
	{
		cemu_assert_debug(id < IML_REG_ID_CAPACITY);
		uint64& word = m_bits[id / 64];
		const uint64 bit = 1ull << (id % 64);
		const bool inserted = (word & bit) == 0;
		word |= bit;
		return {inserted};
	}

	template<typename It>
	void insert(It first, It last)
	{
		for (; first != last; ++first)
			insert(*first);
	}

	// every register in other, at once
	void insert_all(const IMLRegIdSet& other)
	{
		for (size_t i = 0; i < kWords; i++)
			m_bits[i] |= other.m_bits[i];
	}

	bool contains(IMLRegID id) const
	{
		return id < IML_REG_ID_CAPACITY && (m_bits[id / 64] >> (id % 64)) & 1;
	}
	size_t count(IMLRegID id) const { return contains(id) ? 1 : 0; }

	size_t erase(IMLRegID id)
	{
		if (!contains(id))
			return 0;
		m_bits[id / 64] &= ~(1ull << (id % 64));
		return 1;
	}

	void clear() { m_bits = {}; }

	bool empty() const
	{
		for (uint64 w : m_bits)
			if (w)
				return false;
		return true;
	}

	size_t size() const
	{
		size_t n = 0;
		for (uint64 w : m_bits)
			n += std::popcount(w);
		return n;
	}

	const_iterator begin() const { return const_iterator(this, 0); }
	const_iterator end() const { return const_iterator(this, IML_REG_ID_CAPACITY); }

private:
	std::array<uint64, kWords> m_bits{};
};

// A value per register: a slot each, holding the (register, value) pair the
// iteration hands out as std::unordered_map's does, constructed when the
// register is put in and destroyed when it is taken out. V need not have a
// default constructor.
template<typename V>
class IMLRegIdMap
{
public:
	using value_type = std::pair<const IMLRegID, V>;

	template<bool IsConst>
	class iterator_t
	{
		using Map = std::conditional_t<IsConst, const IMLRegIdMap, IMLRegIdMap>;
		using Ref = std::conditional_t<IsConst, const value_type&, value_type&>;
		using Ptr = std::conditional_t<IsConst, const value_type*, value_type*>;

	public:
		iterator_t(Map* map, IMLRegIdSet::const_iterator it) : m_map(map), m_it(it) {}
		Ref operator*() const { return m_map->Entry(*m_it); }
		Ptr operator->() const { return &m_map->Entry(*m_it); }
		iterator_t& operator++()
		{
			++m_it;
			return *this;
		}
		bool operator==(const iterator_t& other) const { return m_it == other.m_it; }
		bool operator!=(const iterator_t& other) const { return m_it != other.m_it; }
		IMLRegID key() const { return *m_it; }

	private:
		Map* m_map;
		IMLRegIdSet::const_iterator m_it;
	};
	using iterator = iterator_t<false>;
	using const_iterator = iterator_t<true>;

	IMLRegIdMap() = default;
	IMLRegIdMap(const IMLRegIdMap& other) { CopyFrom(other); }
	IMLRegIdMap& operator=(const IMLRegIdMap& other)
	{
		if (this != &other)
		{
			clear();
			CopyFrom(other);
		}
		return *this;
	}
	~IMLRegIdMap() { clear(); }

	// the value, put in with V's default if it is not there yet
	V& operator[](IMLRegID id)
	{
		return try_emplace(id).first->second;
	}

	template<typename... Args>
	std::pair<iterator, bool> try_emplace(IMLRegID id, Args&&... args)
	{
		if (m_present.contains(id))
			return {find(id), false};
		cemu_assert_debug(id < IML_REG_ID_CAPACITY);
		new (&m_slots[id]) value_type(std::piecewise_construct, std::forward_as_tuple(id), std::forward_as_tuple(std::forward<Args>(args)...));
		m_present.insert(id);
		return {find(id), true};
	}

	std::pair<iterator, bool> emplace(IMLRegID id, const V& v) { return try_emplace(id, v); }

	iterator find(IMLRegID id)
	{
		if (!m_present.contains(id))
			return end();
		return iterator(this, IMLRegIdSet::const_iterator(&m_present, id));
	}
	const_iterator find(IMLRegID id) const
	{
		if (!m_present.contains(id))
			return end();
		return const_iterator(this, IMLRegIdSet::const_iterator(&m_present, id));
	}

	bool contains(IMLRegID id) const { return m_present.contains(id); }
	size_t count(IMLRegID id) const { return m_present.count(id); }

	V& at(IMLRegID id)
	{
		cemu_assert_debug(contains(id));
		return Entry(id).second;
	}
	const V& at(IMLRegID id) const
	{
		cemu_assert_debug(contains(id));
		return Entry(id).second;
	}

	size_t erase(IMLRegID id)
	{
		if (!m_present.contains(id))
			return 0;
		Entry(id).~value_type();
		m_present.erase(id);
		return 1;
	}
	void erase(iterator it) { erase(it.key()); }

	void clear()
	{
		if constexpr (!std::is_trivially_destructible_v<value_type>)
			for (IMLRegID id : m_present)
				Entry(id).~value_type();
		m_present.clear();
	}

	bool empty() const { return m_present.empty(); }
	size_t size() const { return m_present.size(); }

	iterator begin() { return iterator(this, m_present.begin()); }
	iterator end() { return iterator(this, m_present.end()); }
	const_iterator begin() const { return const_iterator(this, m_present.begin()); }
	const_iterator end() const { return const_iterator(this, m_present.end()); }
	const_iterator cbegin() const { return begin(); }
	const_iterator cend() const { return end(); }

	value_type& Entry(IMLRegID id) { return *std::launder(reinterpret_cast<value_type*>(&m_slots[id])); }
	const value_type& Entry(IMLRegID id) const { return *std::launder(reinterpret_cast<const value_type*>(&m_slots[id])); }

private:
	void CopyFrom(const IMLRegIdMap& other)
	{
		for (IMLRegID id : other.m_present)
			new (&m_slots[id]) value_type(other.Entry(id));
		m_present = other.m_present;
	}

	struct alignas(value_type) SlotStorage
	{
		unsigned char bytes[sizeof(value_type)];
	};
	IMLRegIdSet m_present;
	std::array<SlotStorage, IML_REG_ID_CAPACITY> m_slots;
};
