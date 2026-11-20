// An emulated controller's mappings read together, as the GamePad, the Wii
// Remote and the Pro and Classic Controllers read them every frame
// (EmulatedController::get_mappings_down and get_axis_values): for two
// physical controllers whose buttons and sticks change at random, with
// mappings to either, to buttons and to axis directions, and unmapped ones,
// the same as each mapping read on its own, and nothing allocated.
#include "input/api/Controller.h"
#include "input/emulated/EmulatedController.h"
#include "WindowSystem.h"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>

void WindowSystem::CaptureInput(const ControllerState&, const ControllerState&) {}

#if defined(__SANITIZE_ADDRESS__)
#define COUNT_ALLOCS 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
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
#define CHECK(cond) do { if (!(cond)) { if (s_failures < 20) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

struct TestController : ControllerBase
{
	explicit TestController(const char* uuid) : ControllerBase(uuid, uuid) {}
	std::string_view api_name() const override { return "test"; }
	InputAPI::Type api() const override { return InputAPI::Type(0); }
	bool is_connected() override { return true; }
	ControllerState raw = {};
	ControllerState raw_state() override { return raw; }
};

struct TestEmulated : EmulatedController
{
	TestEmulated() : EmulatedController(0) {}
	using EmulatedController::get_mappings_down;
	using EmulatedController::get_axis_values;
	using EmulatedController::get_axis_value;
	Type type() const override { return Type(0); }
	Vector2<float> get_axis() const override { return {}; }
	Vector2<float> get_rotation() const override { return {}; }
	Vector2<float> get_trigger() const override { return {}; }
	uint32 get_emulated_button_flag(uint32) const override { return 0; }
	size_t get_highest_mapping_id() const override { return 48; }
	bool is_axis_mapping(uint64) const override { return false; }
	bool is_start_down() const override { return false; }
	bool is_left_down() const override { return false; }
	bool is_right_down() const override { return false; }
	bool is_up_down() const override { return false; }
	bool is_down_down() const override { return false; }
	bool is_a_down() const override { return false; }
	bool is_b_down() const override { return false; }
	bool is_home_down() const override { return false; }
};

int main()
{
	auto a = std::make_shared<TestController>("a");
	auto b = std::make_shared<TestController>("b");
	TestEmulated emu;
	emu.add_controller(a);
	emu.add_controller(b);
	std::mt19937 rng(21);
	ControllerBase::Settings none;
	none.axis = none.rotation = none.trigger = ControllerBase::AxisSetting(0.0f);
	a->set_settings(none);
	b->set_settings(none);
	a->calibrate();
	b->calibrate();
	// mappings 1 to 47, one in five unmapped, to buttons and axis directions
	for (uint64 m = 1; m < 48; m++)
	{
		if (m % 5 == 0)
			continue;
		const uint64 button = (m % 3 == 0) ? (kAxisXP + rng() % 12) : (rng() % 24);
		emu.set_mapping(m, (m & 1) ? std::shared_ptr<ControllerBase>(a) : std::shared_ptr<ControllerBase>(b), button);
	}
	std::uniform_real_distribution<float> stick(-1.0f, 1.0f);
	unsigned long long allocs = 0;
	for (int round = 0; round < 3000; round++)
	{
		for (TestController* c : { a.get(), b.get() })
		{
			c->raw = {};
			c->raw.axis = Vector2<float>(stick(rng), stick(rng));
			c->raw.rotation = Vector2<float>(stick(rng), stick(rng));
			c->raw.trigger = Vector2<float>(stick(rng), stick(rng));
			for (uint64 bt = 0; bt < 24; bt++)
				if (rng() % 3 == 0)
					c->raw.buttons.SetButtonState(bt, true);
			c->update_state();
		}
		const unsigned long long before = s_allocs;
		const uint64 down = emu.get_mappings_down(1, 48);
		uint64 mappings[47];
		float values[47];
		for (uint64 m = 1; m < 48; m++)
			mappings[m - 1] = m;
		emu.get_axis_values(mappings, values, 47);
		allocs += s_allocs - before;
		CHECK((down & 1) == 0); // mapping 0 not asked for
		for (uint64 m = 1; m < 48; m++)
		{
			CHECK(((down >> m) & 1) == (emu.is_mapping_down(m) ? 1u : 0u));
			CHECK(values[m - 1] == emu.get_axis_value(m));
		}
	}
	std::printf("emulated_input: %llu allocations for 3000 reads of 47 mappings\n", allocs);
	CHECK(!COUNT_ALLOCS || allocs == 0);
	if (s_failures)
	{
		std::printf("emulated_input_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("emulated_input_test: all passed\n");
	return 0;
}
