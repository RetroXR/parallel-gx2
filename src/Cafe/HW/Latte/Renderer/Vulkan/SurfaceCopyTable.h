#pragma once

#include <cstdint>

#include "util/helpers/PoolNew.h"
#include "util/helpers/U64Table.h"

// What the surface copies make for one slice and mip of a texture - a copy
// pipeline's framebuffer of it, or its descriptor set reading it - found by
// the pipeline, the texture, the slice and the mip. The slots come from a
// pool and the table is sized once (Reserve), so a copy into or from a slice
// not seen before allocates nothing. Each texture keeps a list of its own
// slots (its member surfaceCopySlots), which its release walks; the GPU
// thread's alone. Keys that collide share a table entry, chained.
struct SurfaceCopyKeyMix
{
	static uint64_t Key(const void* owner, const void* texture, int32_t slice, int32_t mip)
	{
		uint64_t k = (uint64_t)(uintptr_t)owner * 0x9E3779B97F4A7C15ull;
		k ^= (uint64_t)(uintptr_t)texture * 0xC2B2AE3D27D4EB4Full;
		k ^= ((uint64_t)(uint32_t)slice << 20 | (uint32_t)mip) * 0x165667B19E3779F9ull;
		return k ^ (k >> 29);
	}
};

template<typename Tex, typename Value, typename Hash = SurfaceCopyKeyMix>
class SurfaceCopyTable
{
public:
	struct Slot : PoolNew<Slot, 256>
	{
		const void* owner;
		Tex* texture;
		int32_t slice;
		int32_t mip;
		Value value{};
		Slot* nextSameKey;
		Slot* prevOfTexture;
		Slot* nextOfTexture;
	};

	SurfaceCopyTable() = default;
	SurfaceCopyTable(const SurfaceCopyTable&) = delete;
	SurfaceCopyTable& operator=(const SurfaceCopyTable&) = delete;

	void Reserve(size_t slots) { m_map.Reserve(slots); }

	Value* Find(const void* owner, Tex* texture, int32_t slice, int32_t mip) const
	{
		for (Slot* s = m_map.Find(Hash::Key(owner, texture, slice, mip)); s; s = s->nextSameKey)
			if (s->owner == owner && s->texture == texture && s->slice == slice && s->mip == mip)
				return &s->value;
		return nullptr;
	}

	// a slot for one not in the table yet, its value reset
	Value* Add(const void* owner, Tex* texture, int32_t slice, int32_t mip)
	{
		const uint64_t key = Hash::Key(owner, texture, slice, mip);
		Slot* s = new Slot();
		s->owner = owner;
		s->texture = texture;
		s->slice = slice;
		s->mip = mip;
		s->nextSameKey = m_map.Find(key);
		m_map.Put(key, s);
		s->prevOfTexture = nullptr;
		s->nextOfTexture = texture->surfaceCopySlots;
		if (s->nextOfTexture)
			s->nextOfTexture->prevOfTexture = s;
		texture->surfaceCopySlots = s;
		m_count++;
		return &s->value;
	}

	// the texture's slots, each value handed to release first
	template<typename F>
	void ReleaseTexture(Tex* texture, F release)
	{
		while (Slot* s = texture->surfaceCopySlots)
		{
			UnlinkFromTexture(s);
			Unchain(s);
			release(s->value);
			delete s;
			m_count--;
		}
	}

	// every slot, each value handed to release first
	template<typename F>
	void ReleaseAll(F release)
	{
		m_map.ForEach([&](uint64_t, Slot* head) {
			while (head)
			{
				Slot* next = head->nextSameKey;
				UnlinkFromTexture(head);
				release(head->value);
				delete head;
				head = next;
			}
		});
		m_map.Clear();
		m_count = 0;
	}

	size_t Count() const { return m_count; }
	size_t Capacity() const { return m_map.Capacity(); }

private:
	void Unchain(Slot* s)
	{
		const uint64_t key = Hash::Key(s->owner, s->texture, s->slice, s->mip);
		Slot* head = m_map.Find(key);
		if (head == s)
		{
			if (s->nextSameKey)
				m_map.Put(key, s->nextSameKey);
			else
				m_map.Remove(key, s);
			return;
		}
		for (Slot* p = head; p; p = p->nextSameKey)
		{
			if (p->nextSameKey == s)
			{
				p->nextSameKey = s->nextSameKey;
				return;
			}
		}
	}

	static void UnlinkFromTexture(Slot* s)
	{
		if (s->prevOfTexture)
			s->prevOfTexture->nextOfTexture = s->nextOfTexture;
		else
			s->texture->surfaceCopySlots = s->nextOfTexture;
		if (s->nextOfTexture)
			s->nextOfTexture->prevOfTexture = s->prevOfTexture;
	}

	U64Table<Slot> m_map;
	size_t m_count{};
};
