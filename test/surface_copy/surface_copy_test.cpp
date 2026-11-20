// The surface copies' framebuffers and descriptor sets (SurfaceCopyTable.h),
// by pipeline, texture, slice and mip: slots added, found and released with
// their textures and all at once, against a reference map, with the table's
// own key and with one under which most keys collide; every value released
// once; and none of it allocating once the table and the pool have room.
#include "Cafe/HW/Latte/Renderer/Vulkan/SurfaceCopyTable.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <new>
#include <random>
#include <tuple>
#include <vector>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define COUNT_ALLOCS 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define COUNT_ALLOCS 0
#endif
#endif
#ifndef COUNT_ALLOCS
#define COUNT_ALLOCS 1
#endif
static unsigned long long s_allocs = 0;
#if COUNT_ALLOCS
void* operator new(size_t size)
{
	s_allocs++;
	if (void* p = std::malloc(size ? size : 1))
		return p;
	throw std::bad_alloc();
}
void* operator new[](size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }
#endif

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

struct Val
{
	int id;
};

// most keys alike: the slots under one key chained
struct CollidingKey
{
	static uint64_t Key(const void*, const void*, int32_t slice, int32_t mip) { return (uint64_t)(slice & 1) + 1; }
};

template<typename Hash>
struct Run
{
	struct Tex;
	using Table = SurfaceCopyTable<Tex, Val, Hash>;
	struct Tex
	{
		typename Table::Slot* surfaceCopySlots{};
	};
	using Key = std::tuple<const void*, Tex*, int32_t, int32_t>;

	static constexpr int kTex = 40;
	static constexpr int kOwners = 3;
	Tex textures[kTex];
	int ownerTags[kOwners];
	Table table;
	std::map<Key, int> ref;
	std::vector<int> released; // how often each id was
	int nextId = 0;

	void Release(Val& v)
	{
		CHECK(v.id >= 0 && v.id < (int)released.size());
		if (v.id >= 0 && v.id < (int)released.size())
			released[v.id]++;
	}

	void Go(uint64_t seed, int ops)
	{
		std::mt19937_64 rng(seed);
		table.Reserve(512);
		released.assign(ops + 1, 0);
		nextId = 0;
		for (int op = 0; op < ops; op++)
		{
			Tex* t = &textures[rng() % kTex];
			const void* owner = &ownerTags[rng() % kOwners];
			const int32_t slice = (int32_t)(rng() % 4), mip = (int32_t)(rng() % 4);
			const Key k{ owner, t, slice, mip };
			if (rng() % 16 == 0)
			{
				// the texture goes: each of its values released once
				for (auto it = ref.begin(); it != ref.end();)
					it = std::get<1>(it->first) == t ? ref.erase(it) : std::next(it);
				table.ReleaseTexture(t, [&](Val& v) { Release(v); });
				CHECK(t->surfaceCopySlots == nullptr);
				continue;
			}
			Val* v = table.Find(owner, t, slice, mip);
			auto r = ref.find(k);
			CHECK((v != nullptr) == (r != ref.end()));
			if (v && r != ref.end())
				CHECK(v->id == r->second);
			if (!v)
			{
				v = table.Add(owner, t, slice, mip);
				CHECK(v->id == 0);
				v->id = nextId;
				ref[k] = nextId++;
			}
			CHECK(table.Count() == ref.size());
		}
		// every one still in it found
		for (auto& [k, id] : ref)
		{
			Val* v = table.Find(std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k));
			CHECK(v && v->id == id);
		}
		// all at once: each released once, every texture's list empty
		std::vector<int> before = released;
		table.ReleaseAll([&](Val& v) { Release(v); });
		CHECK(table.Count() == 0);
		for (Tex& t : textures)
			CHECK(t.surfaceCopySlots == nullptr);
		int extra = 0;
		for (size_t i = 0; i < released.size(); i++)
			extra += released[i] - before[i];
		CHECK(extra == (int)ref.size());
		for (int i = 0; i < nextId; i++)
			CHECK(released[i] == 1);
		ref.clear();
	}

	// once warm, as many coming and going as before allocate nothing
	void NoAllocs()
	{
		if (!COUNT_ALLOCS)
			return;
		table.Reserve(512);
		auto cycle = [&]() {
			for (int i = 0; i < kTex; i++)
				for (int s = 0; s < 4; s++)
					table.Add(&ownerTags[i % kOwners], &textures[i], s, i % 3)->id = 0;
			for (int i = 0; i < kTex; i++)
				table.ReleaseTexture(&textures[i], [](Val&) {});
		};
		cycle();
		const unsigned long long before = s_allocs;
		for (int i = 0; i < 100; i++)
			cycle();
		std::printf("surface_copy: %llu allocations for %d slots added and released\n", s_allocs - before, 100 * kTex * 4);
		CHECK(s_allocs == before);
	}
};

int main()
{
	{
		auto* run = new Run<SurfaceCopyKeyMix>();
		run->Go(1, 20000);
		run->Go(2, 20000);
		run->NoAllocs();
		delete run;
	}
	{
		auto* run = new Run<CollidingKey>();
		run->Go(3, 6000);
		run->NoAllocs();
		delete run;
	}
	// U64Table::ForEach: every entry once
	{
		static int objs[300];
		U64Table<int> t;
		for (int i = 0; i < 300; i++)
			t.Put(1000 + i * 7, &objs[i]);
		std::vector<int> seen(300, 0);
		t.ForEach([&](uint64_t key, int* obj) {
			const long i = obj - objs;
			CHECK(i >= 0 && i < 300 && key == (uint64_t)(1000 + i * 7));
			if (i >= 0 && i < 300)
				seen[i]++;
		});
		for (int n : seen)
			CHECK(n == 1);
	}
	if (s_failures)
	{
		std::printf("surface_copy_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("surface_copy_test: all passed\n");
	return 0;
}
