#include "input/api/Libretro/LibretroController.h"

#include <boost/container/small_vector.hpp>

// Provided by the libretro core (src/libretro/CemuLibretro.cpp), same shape as
// the VPAD hooks in vpad.cpp.
extern bool libretro_get_joypad_button(uint32_t port, uint32_t retro_id);
extern void libretro_get_joypad_analog(uint32_t port, float* lx, float* ly, float* rx, float* ry);
extern float libretro_stick_deadzone(uint32_t port);
extern void libretro_set_port_rumble(uint32_t port, bool on);
// sub 0 is the remote, 1 its Nunchuk (SensorIndex.h)
extern bool libretro_get_remote_motion(uint32_t port, uint32_t sub, MotionSample* out);
extern bool libretro_get_remote_pointer(uint32_t port, Vector2f* position, Vector2f* prev,
	PositionVisibility* visibility);

namespace
{
	// RETRO_DEVICE_ID_JOYPAD_*, in id order. Named after what the button reads
	// as on a RetroPad, not after the Wii U button it ends up driving.
	constexpr std::string_view kButtonNames[] = {
		"B", "Y", "Select", "Start", "Up", "Down", "Left", "Right",
		"A", "X", "L", "R", "L2", "R2", "L3", "R3",
	};
	constexpr uint32 kButtonCount = (uint32)std::size(kButtonNames);
}

LibretroController::LibretroController(uint32 port)
	: base_type(fmt::format("libretro_{}", port), fmt::format("RetroPad {}", port + 1)), m_port(port)
{
}

std::string LibretroController::get_button_name(uint64 button) const
{
	if (button >= kButton0 && button < kButton0 + kButtonCount)
		return std::string(kButtonNames[button - kButton0]);

	return base_type::get_button_name(button);
}

void LibretroController::start_rumble()
{
	libretro_set_port_rumble(m_port, true);
}

void LibretroController::stop_rumble()
{
	libretro_set_port_rumble(m_port, false);
}

bool LibretroController::has_motion()
{
	return libretro_get_remote_motion(m_port, 0, nullptr);
}

MotionSample LibretroController::get_motion_sample()
{
	MotionSample sample{};
	libretro_get_remote_motion(m_port, 0, &sample);
	return sample;
}

bool LibretroController::has_position()
{
	return libretro_get_remote_pointer(m_port, nullptr, nullptr, nullptr);
}

Vector2f LibretroController::get_position()
{
	Vector2f position{};
	libretro_get_remote_pointer(m_port, &position, nullptr, nullptr);
	return position;
}

Vector2f LibretroController::get_prev_position()
{
	Vector2f prev{};
	libretro_get_remote_pointer(m_port, nullptr, &prev, nullptr);
	return prev;
}

PositionVisibility LibretroController::GetPositionVisibility()
{
	PositionVisibility visibility = PositionVisibility::NONE;
	libretro_get_remote_pointer(m_port, nullptr, nullptr, &visibility);
	return visibility;
}

LibretroNunchukMotion::LibretroNunchukMotion(uint32 port)
	: base_type(fmt::format("libretro_nunchuk_{}", port), fmt::format("Nunchuk {}", port + 1)), m_port(port)
{
}

bool LibretroNunchukMotion::has_motion()
{
	return libretro_get_remote_motion(m_port, 0, nullptr) && libretro_get_remote_motion(m_port, 1, nullptr);
}

MotionSample LibretroNunchukMotion::get_motion_sample()
{
	MotionSample sample{};
	libretro_get_remote_motion(m_port, 1, &sample);
	return sample;
}

ControllerState LibretroController::raw_state()
{
	// The port's Stick Dead Zone option, as the sticks' axis setting, which
	// ControllerBase applies to what this returns. Checked at every read, so a
	// change in the menu applies at once.
	const float deadzone = libretro_stick_deadzone(m_port);
	if (get_settings().axis.deadzone != deadzone)
	{
		set_axis_settings(AxisSetting(deadzone));
		set_rotation_settings(AxisSetting(deadzone));
	}

	ControllerState result{};

	boost::container::small_vector<uint32, 16> pressedButtons;
	for (uint32 id = 0; id < kButtonCount; ++id)
	{
		if (libretro_get_joypad_button(m_port, id))
			pressedButtons.emplace_back(kButton0 + id);
	}
	result.buttons.SetPressedButtons(pressedButtons);

	float lx, ly, rx, ry;
	libretro_get_joypad_analog(m_port, &lx, &ly, &rx, &ry);
	result.axis = { lx, ly };
	result.rotation = { rx, ry };

	return result;
}
