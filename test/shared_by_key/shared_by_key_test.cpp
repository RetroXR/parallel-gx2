// Objects shared by key (SharedByKey.h), as the Vulkan renderer shares its
// rect emulation geometry shaders: eight threads asking for the same keys at
// once each get the one object made for a key, or nullptr while it is being
// published; each key's object made once; a key under another check, or a
// full table, gives nullptr; and Clear hands every object over once.
#include "util/helpers/SharedByKey.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

static std::atomic<int> s_failures{0};
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

struct Obj
{
	uint64_t key;
	std::atomic<int> uses{0};
};

int main()
{
	constexpr int kKeys = 600;
	constexpr int kThreads = 8;
	static SharedByKey<Obj, 1024> table;
	static std::atomic<int> made[kKeys];
	std::atomic<int> privates{0};
	// keys that probe into each other: some alike in their low bits
	auto keyOf = [](int i) { return (uint64_t)(i % 7) * 1024 + (uint64_t)i * 0x10000 + 1; };
	std::atomic<bool> go{false};
	std::vector<std::thread> threads;
	for (int t = 0; t < kThreads; t++)
	{
		threads.emplace_back([&, t] {
			while (!go.load())
				;
			for (int round = 0; round < 3; round++)
			{
				for (int j = 0; j < kKeys; j++)
				{
					const int i = (j * (t + 1) * 7 + round) % kKeys;
					Obj* o = table.Get(keyOf(i), (uint64_t)i * 3, [&] {
						made[i]++;
						return new Obj{ keyOf(i) };
					});
					if (!o)
					{
						privates++; // its own, as the renderer would make
						continue;
					}
					CHECK(o->key == keyOf(i));
					o->uses++;
				}
			}
		});
	}
	go = true;
	for (auto& th : threads)
		th.join();
	int total = 0;
	for (int i = 0; i < kKeys; i++)
	{
		CHECK(made[i].load() == 1);
		// published now: every thread finds it
		Obj* o = table.Get(keyOf(i), (uint64_t)i * 3, [] { return (Obj*)nullptr; });
		CHECK(o && o->key == keyOf(i));
		if (o)
			total += o->uses.load();
		// another source under the same key
		CHECK(table.Get(keyOf(i), (uint64_t)i * 3 + 1, [] { return (Obj*)nullptr; }) == nullptr);
	}
	CHECK(total + privates.load() == kThreads * 3 * kKeys);
	std::printf("shared_by_key: %d shared uses, %d made apart while being published\n", total, privates.load());
	// the rest of the slots, then none left
	for (int i = kKeys; i < 1024; i++)
		CHECK(table.Get(keyOf(i), 0, [&] { return new Obj{ keyOf(i) }; }) != nullptr);
	CHECK(table.Get(keyOf(5000), 0, [] { return new Obj{ 0 }; }) == nullptr);
	int cleared = 0;
	table.Clear([&](Obj* o) { cleared++; delete o; });
	CHECK(cleared == 1024);
	// empty again: a key is made anew
	bool remade = false;
	Obj* again = table.Get(keyOf(1), 3, [&] { remade = true; return new Obj{ keyOf(1) }; });
	CHECK(again && remade);
	table.Clear([](Obj* o) { delete o; });
	if (s_failures)
	{
		std::printf("shared_by_key_test: %d failed\n", s_failures.load());
		return 1;
	}
	std::printf("shared_by_key_test: all passed\n");
	return 0;
}
