#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// Objects shared by a 64-bit key and a second check of what made them, made
// once by whichever thread first asks: insert-only and lock-free, in Slots
// slots in place. Get claims a slot for the key and makes the object there,
// or finds it made; a thread that finds the key claimed and not yet
// published, under another check, or no slot left, gets nullptr and makes
// its own. Nothing taken out but by Clear, when no thread uses the table.
template<typename T, size_t Slots>
class SharedByKey
{
	static_assert(Slots && (Slots & (Slots - 1)) == 0, "a power of two");

public:
	// key not 0; make() is called by the thread that claims the key, once
	template<typename F>
	T* Get(uint64_t key, uint64_t check, F make)
	{
		for (size_t i = 0; i < Slots; i++)
		{
			Slot& slot = m_slots[(key + i) & (Slots - 1)];
			uint64_t at = slot.key.load(std::memory_order_acquire);
			if (at == 0 && slot.key.compare_exchange_strong(at, key, std::memory_order_acq_rel))
			{
				T* obj = make();
				slot.check.store(check, std::memory_order_relaxed);
				slot.obj.store(obj, std::memory_order_release);
				return obj;
			}
			if (at != key)
				continue;
			T* obj = slot.obj.load(std::memory_order_acquire);
			if (!obj || slot.check.load(std::memory_order_relaxed) != check)
				return nullptr;
			return obj;
		}
		return nullptr;
	}

	// every object handed to f and the table emptied
	template<typename F>
	void Clear(F f)
	{
		for (Slot& slot : m_slots)
		{
			if (T* obj = slot.obj.exchange(nullptr))
				f(obj);
			slot.check.store(0);
			slot.key.store(0);
		}
	}

private:
	struct Slot
	{
		std::atomic<uint64_t> key{0};
		std::atomic<uint64_t> check{0};
		std::atomic<T*> obj{nullptr};
	};
	Slot m_slots[Slots];
};
