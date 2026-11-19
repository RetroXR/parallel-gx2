// The shader state cache's table and aux variants (ShaderStateTable.h):
// states added, found, taken out and walked as LatteShader.cpp's cleanup
// walks them, against a reference map; a full table refusing a state; the
// variants past their places replacing the oldest but the first; and none
// of it allocating, however many states come and go.
#include "Cafe/HW/Latte/Core/ShaderStateTable.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <new>
#include <random>
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

struct Variant
{
	uint64_t aux{};
	int shader{};
};

struct Info
{
	uint64_t mark{};
	uint32_t lastAccess{};
	ShaderStateVariants<Variant, 4> variants;
};

static ShaderStateTable<Info, 64> s_table; // 48 states at once

static void Variants()
{
	ShaderStateVariants<Variant, 4> v;
	Variant replaced{};
	for (int i = 0; i < 4; i++)
		CHECK(!v.Add(Variant{ (uint64_t)i, i }, &replaced));
	CHECK(v.size() == 4);
	// past four: the second, third and fourth, then the second's again (the
	// first one added past them); never the first
	const int want[6] = { 1, 2, 3, 10, 11, 12 };
	for (int i = 0; i < 6; i++)
	{
		CHECK(v.Add(Variant{ (uint64_t)(10 + i), 10 + i }, &replaced));
		CHECK(replaced.shader == want[i]);
		CHECK(v[0].shader == 0 && v.size() == 4);
	}
	int seen = 0;
	for (auto& it : v)
		seen += it.shader;
	CHECK(seen == 0 + 13 + 14 + 15);
	v.Clear();
	CHECK(v.empty());
}

static void Table()
{
	std::map<uint64_t, uint64_t> ref;
	std::mt19937_64 rng(20261119u);
	CHECK(s_table.kMax == 48);
	const unsigned long long before = s_allocs;
	size_t cursor = 0;
	for (int round = 0; round < 20000; round++)
	{
		const uint64_t key = rng() % 200 + 1;
		const unsigned op = (unsigned)(rng() % 3);
		Info* found = s_table.Find(key);
		CHECK((found != nullptr) == (ref.count(key) != 0));
		if (found)
			CHECK(found->mark == ref[key]);
		if (op == 0 && !found)
		{
			Info* info = s_table.Add(key);
			if (ref.size() >= s_table.kMax)
				CHECK(info == nullptr);
			else
			{
				CHECK(info != nullptr && info->mark == 0 && info->variants.empty());
				if (info)
				{
					info->mark = rng() | 1;
					Variant r;
					info->variants.Add(Variant{ key, 1 }, &r);
					ref[key] = info->mark;
				}
			}
		}
		else if (op == 1)
		{
			CHECK(s_table.Remove(key) == (found != nullptr));
			ref.erase(key);
		}
		else
		{
			// the cleanup's walk: a few keys, every one of them in it
			for (int i = 0; i < 5 && !s_table.Empty(); i++)
			{
				if (cursor >= s_table.Count())
					cursor = 0;
				const uint64_t k = s_table.KeyAt(cursor);
				CHECK(ref.count(k) == 1 && s_table.Find(k) != nullptr);
				if (rng() % 4 == 0)
				{
					s_table.Remove(k);
					ref.erase(k);
					continue;
				}
				cursor++;
			}
		}
		CHECK(s_table.Count() == ref.size());
	}
	// every key the dense list holds is a state, once each
	std::map<uint64_t, int> listed;
	for (size_t i = 0; i < s_table.Count(); i++)
		listed[s_table.KeyAt(i)]++;
	CHECK(listed.size() == ref.size());
	for (auto& it : listed)
		CHECK(it.second == 1 && ref.count(it.first) == 1);
	if (COUNT_ALLOCS)
	{
		// the reference map allocates; the table must not have: count it alone
		const unsigned long long mapAllocs = s_allocs - before;
		const unsigned long long t0 = s_allocs;
		for (int i = 0; i < 100000; i++)
		{
			const uint64_t key = rng() % 100 + 1000;
			if (Info* info = s_table.Find(key))
			{
				Variant r;
				info->variants.Add(Variant{ key, i }, &r);
				if (i % 3 == 0)
					s_table.Remove(key);
			}
			else
				s_table.Add(key);
		}
		std::printf("shader_state: %llu allocations by the reference map, %llu by the table\n", mapAllocs, s_allocs - t0);
		CHECK(s_allocs == t0);
	}
	s_table.Clear();
	CHECK(s_table.Empty() && s_table.Find(1000) == nullptr);
}

// filled to the brim: a state past it refused, the ones in it untouched; one
// taken out makes room for one
static void Full()
{
	s_table.Clear();
	for (uint64_t k = 1; k <= s_table.kMax; k++)
	{
		Info* info = s_table.Add(k * 7919);
		CHECK(info != nullptr);
		if (info)
			info->mark = k;
	}
	CHECK(s_table.Count() == s_table.kMax);
	for (int i = 0; i < 3; i++)
		CHECK(s_table.Add(100000 + i) == nullptr);
	CHECK(s_table.Count() == s_table.kMax);
	for (uint64_t k = 1; k <= s_table.kMax; k++)
	{
		Info* info = s_table.Find(k * 7919);
		CHECK(info != nullptr && info->mark == k);
		CHECK(s_table.KeyAt((size_t)k - 1) == k * 7919);
	}
	CHECK(s_table.Remove(7919 * 5));
	Info* again = s_table.Add(424242);
	CHECK(again != nullptr && s_table.Find(424242) == again && s_table.Find(7919 * 5) == nullptr);
	s_table.Clear();
}

int main()
{
	Variants();
	Table();
	Full();
	if (s_failures)
	{
		std::printf("shader_state_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("shader_state_test: all passed\n");
	return 0;
}
