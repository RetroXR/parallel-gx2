#pragma once

#include "input/api/Libretro/LibretroControllerProvider.h"
#include "input/api/Controller.h"

// A libretro port seen as a Cemu controller. Button ids are the libretro joypad
// ids (RETRO_DEVICE_ID_JOYPAD_*) offset by kButton0, so a mapping reads as
// kButton0 + RETRO_DEVICE_ID_JOYPAD_B and needs no translation table.
class LibretroController : public Controller<LibretroControllerProvider>
{
public:
	explicit LibretroController(uint32 port);

	std::string_view api_name() const override
	{
		static_assert(to_string(InputAPI::Libretro) == "Libretro");
		return to_string(InputAPI::Libretro);
	}
	InputAPI::Type api() const override { return InputAPI::Libretro; }

	bool is_connected() override { return true; }

	// Through the frontend's rumble interface, on this port. The strength is
	// the core's Rumble Strength option, not the per-controller setting.
	bool has_rumble() override { return true; }
	void start_rumble() override;
	void stop_rumble() override;

	// A Wii Remote's own accelerometer, and its MotionPlus gyroscope when one
	// is fitted, from the frontend's sensors on this port (Wii Remote Motion).
	// False while the frontend gives this port none, so a title sees a remote
	// lying still rather than one reading zero g.
	bool has_motion() override;
	MotionSample get_motion_sample() override;

	// Where the remote points (Wii Remote Pointer): the frontend's pointer
	// over the TV picture, or the IR camera's own dots.
	bool has_position() override;
	Vector2f get_position() override;
	Vector2f get_prev_position() override;
	PositionVisibility GetPositionVisibility() override;

	uint32 port() const { return m_port; }

	std::string get_button_name(uint64 button) const override;

protected:
	ControllerState raw_state() override;

private:
	uint32 m_port;
};

// The Nunchuk's accelerometer on a Wii Remote's port: the frontend's sensor
// sub-device 1 there (SensorIndex.h). Added to the remote's emulated
// controller after the LibretroController, because WPADController takes the
// SECOND motion source it finds as the Nunchuk's (has_second_motion). It has
// no buttons or sticks of its own; those are the port's pad, mapped onto the
// remote's Nunchuk ids.
class LibretroNunchukMotion : public Controller<LibretroControllerProvider>
{
public:
	explicit LibretroNunchukMotion(uint32 port);

	std::string_view api_name() const override
	{
		return to_string(InputAPI::Libretro);
	}
	InputAPI::Type api() const override { return InputAPI::Libretro; }

	bool is_connected() override { return true; }
	bool has_axis() const override { return false; }

	// Only while the remote's own motion is there too: with the remote's
	// missing, this one would be the first source and be taken for the remote.
	bool has_motion() override;
	MotionSample get_motion_sample() override;

protected:
	ControllerState raw_state() override { return {}; }

private:
	uint32 m_port;
};
