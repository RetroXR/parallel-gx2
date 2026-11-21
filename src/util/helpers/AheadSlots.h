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
			s.key = key;
			return {&s.payload, i};
		}
		if (state != kReady)
			return {nullptr, i};
		// a published one, read only once held
		if (!s.state.compare_exchange_strong(state, kBusy, std::memory_order_acquire))
			return {nullptr, i};
		if (s.key == key || !stale(s.key, s.payload))
		{
			s.state.store(kReady, std::memory_order_release);
			return {nullptr, i};
		}
		s.key = key;
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
		if (s.key != key)
		{
			s.state.store(kReady, std::memory_order_release);
			return {nullptr, i};
		}
		return {&s.payload, i};
	}
	void Release(Held h) { m_slots[h.index].state.store(kEmpty, std::memory_order_release); }

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
		uint64_t key{};
		std::atomic<uint32_t> state{kEmpty};
	};
	Slot m_slots[N];
};

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
