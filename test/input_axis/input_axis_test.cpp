// The stick dead zone and range (ControllerBase::apply_axis_setting), written
// out in plain arithmetic, against the way it was written with glm - length()
// a square root of the dot product, normalize() the vector times one over
// it - for sticks all over their range, defaults off centre, dead zones and
// ranges of every kind: the same result to a few units in the last place,
// zero inside the dead zone, the smallest value where an axis is near zero.
// Both are timed.
#include "input/api/Controller.h"
#include "WindowSystem.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

void WindowSystem::CaptureInput(const ControllerState&, const ControllerState&) {}

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { if (s_failures < 20) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)

struct TestController : ControllerBase
{
	TestController() : ControllerBase("test", "test") {}
	std::string_view api_name() const override { return "test"; }
	InputAPI::Type api() const override { return InputAPI::Type(0); }
	bool is_connected() override { return true; }
	ControllerState raw_state() override { return {}; }
};

// as it was: glm's length() and normalize()
__attribute__((noinline)) static void Reference(float& ax, float& ay, float dx, float dy, const ControllerBase::AxisSetting& setting)
{
	constexpr float kMaxValue = 1.0f + ControllerState::kMinAxisValue;
	auto length = [](float x, float y) { return std::sqrt(x * x + y * y); };
	if (setting.deadzone < 1.0f)
	{
		ax = ax < dx ? (ax - dx) / (kMaxValue + dx) : (ax - dx) / (kMaxValue - dx);
		ay = ay < dy ? (ay - dy) / (kMaxValue + dy) : (ay - dy) / (kMaxValue - dy);
		float len = length(ax, ay);
		if (len >= setting.deadzone)
		{
			ax *= setting.range;
			ay *= setting.range;
			len = length(ax, ay);
			if (len > 0)
			{
				const float inv = 1.0f / std::sqrt(ax * ax + ay * ay);
				ax *= inv;
				ay *= inv;
				const float s = (len - setting.deadzone) / (kMaxValue - setting.deadzone);
				ax *= s;
				ay *= s;
				if (length(ax, ay) > 1.0f)
				{
					const float inv2 = 1.0f / std::sqrt(ax * ax + ay * ay);
					ax *= inv2;
					ay *= inv2;
				}
			}
			if (ax != 0 || ay != 0)
			{
				if (std::abs(ax) < ControllerState::kMinAxisValue)
					ax = ControllerState::kMinAxisValue;
				if (std::abs(ay) < ControllerState::kMinAxisValue)
					ay = ControllerState::kMinAxisValue;
			}
			return;
		}
	}
	ax = 0;
	ay = 0;
}

int main()
{
	TestController c;
	std::mt19937 rng(11);
	std::uniform_real_distribution<float> stick(-1.0f, 1.0f), dz(0.0f, 0.6f), range(0.3f, 1.5f), def(-0.05f, 0.05f);
	int nearEdge = 0;
	for (int i = 0; i < 2000000; i++)
	{
		ControllerBase::AxisSetting setting(i % 50 == 0 ? 0.0f : dz(rng));
		setting.range = i % 7 == 0 ? 1.0f : range(rng);
		const Vector2<float> d = i % 3 == 0 ? Vector2<float>() : Vector2<float>(def(rng), def(rng));
		float x = stick(rng), y = stick(rng);
		if (i % 11 == 0)
			x = d.x; // one axis at rest
		Vector2<float> a(x, y);
		c.apply_axis_setting(a, d, setting);
		float rx = x, ry = y;
		Reference(rx, ry, d.x, d.y, setting);
		// the dead zone's edge, compared squared or not, may fall either way
		// for a stick exactly on it
		const bool zero = a.x == 0 && a.y == 0, rzero = rx == 0 && ry == 0;
		if (zero != rzero)
		{
			// only a stick on the edge: its length, as the dead zone sees it
			constexpr float kMaxValue = 1.0f + ControllerState::kMinAxisValue;
			const float nx = x < d.x ? (x - d.x) / (kMaxValue + d.x) : (x - d.x) / (kMaxValue - d.x);
			const float ny = y < d.y ? (y - d.y) / (kMaxValue + d.y) : (y - d.y) / (kMaxValue - d.y);
			nearEdge++;
			CHECK(std::abs(std::sqrt(nx * nx + ny * ny) - setting.deadzone) < 1e-6f);
			continue;
		}
		CHECK(std::abs(a.x - rx) <= 4e-7f + 2e-6f * std::abs(rx));
		CHECK(std::abs(a.y - ry) <= 4e-7f + 2e-6f * std::abs(ry));
		CHECK(a.x * a.x + a.y * a.y <= 1.0f + 1e-5f);
	}
	// the smallest value for an axis at rest, the other moved
	{
		ControllerBase::AxisSetting setting(0.1f);
		Vector2<float> a(0.0f, 0.8f);
		c.apply_axis_setting(a, Vector2<float>(), setting);
		CHECK(a.x == ControllerState::kMinAxisValue && a.y > 0.7f);
		Vector2<float> z(0.05f, 0.05f);
		c.apply_axis_setting(z, Vector2<float>(), setting);
		CHECK(z.x == 0 && z.y == 0);
		Vector2<float> off(0.9f, 0.9f); // past full tilt: of length 1
		c.apply_axis_setting(off, Vector2<float>(), setting);
		CHECK(std::abs(off.x * off.x + off.y * off.y - 1.0f) < 1e-5f);
	}
	// timed: the same sticks both ways
	std::vector<Vector2<float>> sticks(1 << 16);
	for (auto& s : sticks)
		s = Vector2<float>(stick(rng), stick(rng));
	ControllerBase::AxisSetting setting(0.2f);
	setting.range = 0.9f;
	float sink = 0;
	auto t0 = std::chrono::steady_clock::now();
	for (int r = 0; r < 100; r++)
		for (const auto& s : sticks)
		{
			Vector2<float> a = s;
			c.apply_axis_setting(a, Vector2<float>(), setting);
			sink += a.x + a.y;
		}
	auto t1 = std::chrono::steady_clock::now();
	for (int r = 0; r < 100; r++)
		for (const auto& s : sticks)
		{
			float x = s.x, y = s.y;
			Reference(x, y, 0, 0, setting);
			sink += x + y;
		}
	auto t2 = std::chrono::steady_clock::now();
	const double nNew = std::chrono::duration<double, std::nano>(t1 - t0).count() / (100.0 * sticks.size());
	const double nRef = std::chrono::duration<double, std::nano>(t2 - t1).count() / (100.0 * sticks.size());
	std::printf("input_axis: %.2f ns a stick, %.2f ns as with glm (%d on the dead zone's edge, sum %g)\n", nNew, nRef, nearEdge, (double)sink);
	if (s_failures)
	{
		std::printf("input_axis_test: %d failed\n", s_failures);
		return 1;
	}
	std::printf("input_axis_test: all passed\n");
	return 0;
}
