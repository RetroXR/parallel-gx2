#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// Things made ahead by one thread for another, in a fixed number of slots,
// one per key at a time (the key's slot is key % N). A producer claims the
// key's slot, fills it and publishes it; the consumer takes it by key,
// moves out what it wants and releases the slot. Either takes a published
// slot from the other only by claiming it, so what is in a slot belongs to
// one thread at a time. A slot taken by another key is not waited for: the
// producer makes nothing ahead for its key, which is then made where it is
// needed, as before. No lock; nothing allocated.
template<typename Payload, size_t N>
class AheadSlots
{
public:
	// A slot held by one thread: its payload, and what to hand back
	struct Held
	{
		Payload* payload;
		size_t index;
		explicit operator bool() const { return payload != nullptr; }
	};

	// The key's slot, now the caller's to fill (Publish) or give up
	// (Abandon): empty, or published for a key stale(key, payload) says is
	// no longer wanted - whose payload the caller then finds still in it (hadPayload)
	// to discard. Empty where the key is already in hand or published, or
	// the slot is another key's.
	template<typename StaleFn>
	Held Claim(uint64_t key, StaleFn stale, bool* hadPayload)
	{
		const size_t i = key % N;
		Slot& s = m_slots[i];
		*hadPayload = false;
		uint32_t state = kEmpty;
		if (s.state.compare_exchange_strong(state, kBusy, std::memory_order_acquire))
		{
			s.key.store(key, std::memory_order_relaxed);
			return {&s.payload, i};
		}
		if (state != kReady)
			return {nullptr, i};
		// a published one, read only once held
		if (!s.state.compare_exchange_strong(state, kBusy, std::memory_order_acquire))
			return {nullptr, i};
		if (s.key.load(std::memory_order_relaxed) == key || !stale(s.key.load(std::memory_order_relaxed), s.payload))
		{
			s.state.store(kReady, std::memory_order_release);
			return {nullptr, i};
		}
		s.key.store(key, std::memory_order_relaxed);
		*hadPayload = true;
		return {&s.payload, i};
	}

	void Publish(Held h) { m_slots[h.index].state.store(kReady, std::memory_order_release); }
	void Abandon(Held h) { m_slots[h.index].state.store(kEmpty, std::memory_order_release); }

	// The consumer's: the payload published for key, now the caller's until
	// Release; empty where none is
	Held Take(uint64_t key)
	{
		const size_t i = key % N;
		Slot& s = m_slots[i];
		if (s.state.load(std::memory_order_relaxed) != kReady)
			return {nullptr, i};
		uint32_t state = kReady;
		if (!s.state.compare_exchange_strong(state, kBusy, std::memory_order_acquire))
			return {nullptr, i};
		if (s.key.load(std::memory_order_relaxed) != key)
		{
			s.state.store(kReady, std::memory_order_release);
			return {nullptr, i};
		}
		return {&s.payload, i};
	}
	void Release(Held h) { m_slots[h.index].state.store(kEmpty, std::memory_order_release); }

	// The consumer's: what is published in slot index, whatever its key, now
	// the caller's to Publish again or Release; empty where nothing is
	Held TakeAt(size_t index)
	{
		Slot& s = m_slots[index % N];
		uint32_t state = kReady;
		if (!s.state.compare_exchange_strong(state, kBusy, std::memory_order_acquire))
			return {nullptr, index % N};
		return {&s.payload, index % N};
	}
	// the key of a slot held
	uint64_t KeyOf(Held h) const { return m_slots[h.index].key.load(std::memory_order_relaxed); }
	static constexpr size_t SlotOf(uint64_t key) { return key % N; }
	// Whether key's slot is held for it right now - by a producer filling it,
	// where the consumer asking holds nothing itself: for counting, a guess
	// that may be stale by the time it is read
	bool InHand(uint64_t key) const
	{
		const Slot& s = m_slots[key % N];
		return s.state.load(std::memory_order_acquire) == kBusy && s.key.load(std::memory_order_relaxed) == key;
	}

	// Every published payload handed to discard and its slot emptied: with
	// no producer running (at the end)
	template<typename DiscardFn>
	void DrainPublished(DiscardFn discard)
	{
		for (Slot& s : m_slots)
		{
			uint32_t state = kReady;
			if (s.state.compare_exchange_strong(state, kBusy, std::memory_order_acquire))
			{
				discard(s.payload);
				s.state.store(kEmpty, std::memory_order_release);
			}
		}
	}

private:
	enum : uint32_t { kEmpty, kBusy, kReady };
	struct Slot
	{
		Payload payload{};
		std::atomic<uint64_t> key{0}; // atomic for InHand, which reads it unheld
		std::atomic<uint32_t> state{kEmpty};
	};
	Slot m_slots[N];
};

// Whether a payload published and not taken may be claimed back for
// another key (AheadSlots::Claim's stale): never while something made from
// it is in use (inUse), else once its key is known another way (known) or
// it has waited past maxAge
inline bool AheadSlotStale(bool inUse, bool known, uint32_t age, uint32_t maxAge)
{
	return !inUse && (known || age > maxAge);
}

// Keys seen, for any thread to ask about while one adds them: insert-only,
// open addressing over a fixed table. A key that no longer fits is reported
// unseen, never the other way round (0 is kept as 1), so it is a filter:
// "seen" may be trusted, "unseen" is a guess.
template<size_t Capacity>
class SeenKeys
{
	static_assert((Capacity & (Capacity - 1)) == 0, "a power of two");
	static constexpr size_t kProbe = 16;

public:
	void Insert(uint64_t key)
	{
		key = key ? key : 1;
		for (size_t i = 0, at = Mix(key); i < kProbe; i++, at++)
		{
			std::atomic<uint64_t>& e = m_keys[at & (Capacity - 1)];
			uint64_t cur = e.load(std::memory_order_relaxed);
			if (cur == key)
				return;
			if (cur == 0 && e.compare_exchange_strong(cur, key, std::memory_order_release))
				return;
			if (cur == key)
				return;
		}
	}
	bool Contains(uint64_t key) const
	{
		key = key ? key : 1;
		for (size_t i = 0, at = Mix(key); i < kProbe; i++, at++)
		{
			const uint64_t cur = m_keys[at & (Capacity - 1)].load(std::memory_order_acquire);
			if (cur == key)
				return true;
			if (cur == 0)
				return false;
		}
		return false;
	}
	// with nobody adding or asking
	void Clear()
	{
		for (std::atomic<uint64_t>& e : m_keys)
			e.store(0, std::memory_order_relaxed);
	}

private:
	static size_t Mix(uint64_t key)
	{
		key ^= key >> 33;
		key *= 0xff51afd7ed558ccdull;
		key ^= key >> 33;
		return (size_t)key;
	}
	std::atomic<uint64_t> m_keys[Capacity]{};
};

// Values by key, one thread adding while any read: insert-only, open
// addressing over a fixed table, a value written before its key is
// published and never changed after. A key added twice keeps its first
// value; one that no longer fits is not found. The values must outlive
// the readers.
template<typename T, size_t Capacity>
class PublishedMap
{
	static_assert((Capacity & (Capacity - 1)) == 0, "a power of two");
	static constexpr size_t kProbe = 32;

public:
	// the adding thread's
	void Insert(uint64_t key, T* value)
	{
		key = key ? key : 1;
		for (size_t i = 0, at = Mix(key); i < kProbe; i++, at++)
		{
			Entry& e = m_entries[at & (Capacity - 1)];
			const uint64_t cur = e.key.load(std::memory_order_relaxed);
			if (cur == key)
				return;
			if (cur == 0)
			{
				e.value.store(value, std::memory_order_relaxed);
				e.key.store(key, std::memory_order_release);
				return;
			}
		}
	}
	T* Find(uint64_t key) const
	{
		key = key ? key : 1;
		for (size_t i = 0, at = Mix(key); i < kProbe; i++, at++)
		{
			const Entry& e = m_entries[at & (Capacity - 1)];
			const uint64_t cur = e.key.load(std::memory_order_acquire);
			if (cur == key)
				return e.value.load(std::memory_order_relaxed);
			if (cur == 0)
				return nullptr;
		}
		return nullptr;
	}
	// with nobody adding or asking
	void Clear()
	{
		for (Entry& e : m_entries)
		{
			e.key.store(0, std::memory_order_relaxed);
			e.value.store(nullptr, std::memory_order_relaxed);
		}
	}

private:
	static size_t Mix(uint64_t key)
	{
		key ^= key >> 33;
		key *= 0xc4ceb9fe1a85ec53ull;
		key ^= key >> 33;
		return (size_t)key;
	}
	struct Entry
	{
		std::atomic<uint64_t> key{0};
		std::atomic<T*> value{nullptr};
	};
	Entry m_entries[Capacity];
};
