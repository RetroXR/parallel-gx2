#include "input/emulated/ClassicController.h"

#include "input/api/Controller.h"

ClassicController::ClassicController(size_t player_index)
	: WPADController(player_index, kDataFormat_CLASSIC)
{
}

uint32 ClassicController::get_emulated_button_flag(uint32 id) const
{
	return s_get_emulated_button_flag(id);
}

uint32 ClassicController::s_get_emulated_button_flag(uint32 id)
{
	switch (id)
	{
	case kButtonId_A:
		return kCLButton_A;
	case kButtonId_B:
		return kCLButton_B;
	case kButtonId_X:
		return kCLButton_X;
	case kButtonId_Y:
		return kCLButton_Y;

	case kButtonId_Plus:
		return kCLButton_Plus;
	case kButtonId_Minus:
		return kCLButton_Minus;

	case kButtonId_Up:
		return kCLButton_Up;
	case kButtonId_Down:
		return kCLButton_Down;
	case kButtonId_Left:
		return kCLButton_Left;
	case kButtonId_Right:
		return kCLButton_Right;

	case kButtonId_L:
		return kCLButton_L;
	case kButtonId_ZL:
		return kCLButton_ZL;
	case kButtonId_R:
		return kCLButton_R;
	case kButtonId_ZR:
		return kCLButton_ZR;
	}

	return 0;
}

std::string_view ClassicController::get_button_name(ButtonId id)
{
	switch (id)
	{
	case kButtonId_A: return "A";
	case kButtonId_B: return "B";
	case kButtonId_X: return "X";
	case kButtonId_Y: return "Y";
	case kButtonId_L: return "L";
	case kButtonId_R: return "R";
	case kButtonId_ZL: return "ZL";
	case kButtonId_ZR: return "ZR";

	case kButtonId_Plus: return "+";
	case kButtonId_Minus: return "-";
	case kButtonId_Home: return TR_NOOP("home");

	case kButtonId_Up: return TR_NOOP("up");
	case kButtonId_Down: return TR_NOOP("down");
	case kButtonId_Left: return TR_NOOP("left");
	case kButtonId_Right: return TR_NOOP("right");

	case kButtonId_StickL_Up: return TR_NOOP("up");
	case kButtonId_StickL_Down: return TR_NOOP("down");
	case kButtonId_StickL_Left: return TR_NOOP("left");
	case kButtonId_StickL_Right: return TR_NOOP("right");

	case kButtonId_StickR_Up: return TR_NOOP("up");
	case kButtonId_StickR_Down: return TR_NOOP("down");
	case kButtonId_StickR_Left: return TR_NOOP("left");
	case kButtonId_StickR_Right: return TR_NOOP("right");

	default:
		return "";
	}
}

Vector2<float> ClassicController::get_axis() const
{
	const uint64 mappings[4] = { kButtonId_StickL_Left, kButtonId_StickL_Right, kButtonId_StickL_Up, kButtonId_StickL_Down };
	float v[4];
	get_axis_values(mappings, v, 4);
	const float left = v[0], right = v[1], up = v[2], down = v[3];

	float x = (left > right) ? -left : right;
	float y = (up > down) ? up : -down;
	// at most of length 1: the square root only past it
	const float lengthSq = x * x + y * y;
	if (lengthSq > 1.0f)
	{
		const float inv = 1.0f / sqrtf(lengthSq);
		x *= inv;
		y *= inv;
	}
	return { x, y };
}

Vector2<float> ClassicController::get_rotation() const
{
	const uint64 mappings[4] = { kButtonId_StickR_Left, kButtonId_StickR_Right, kButtonId_StickR_Up, kButtonId_StickR_Down };
	float v[4];
	get_axis_values(mappings, v, 4);
	const float left = v[0], right = v[1], up = v[2], down = v[3];

	float x = (left > right) ? -left : right;
	float y = (up > down) ? up : -down;
	// at most of length 1: the square root only past it
	const float lengthSq = x * x + y * y;
	if (lengthSq > 1.0f)
	{
		const float inv = 1.0f / sqrtf(lengthSq);
		x *= inv;
		y *= inv;
	}
	return { x, y };
}

Vector2<float> ClassicController::get_trigger() const
{
	const uint64 mappings[2] = { kButtonId_ZL, kButtonId_ZR };
	float v[2];
	get_axis_values(mappings, v, 2);
	const float left = v[0], right = v[1];
	return { left, right };
}

bool ClassicController::set_default_mapping(const std::shared_ptr<ControllerBase>& controller)
{
	std::vector<std::pair<uint64, uint64>> mapping;
	switch (controller->api())
	{
	case InputAPI::XInput:
	{
		mapping =
		{
			{kButtonId_A, kButton13},
			{kButtonId_B, kButton12},
			{kButtonId_X, kButton15},
			{kButtonId_Y, kButton14},

			{kButtonId_L, kButton8},
			{kButtonId_R, kButton9},
			{kButtonId_ZL, kTriggerXP},
			{kButtonId_ZR, kTriggerYP},

			{kButtonId_Plus, kButton4},
			{kButtonId_Minus, kButton5},

			{kButtonId_Up, kButton0},
			{kButtonId_Down, kButton1},
			{kButtonId_Left, kButton2},
			{kButtonId_Right, kButton3},

			{kButtonId_StickL_Up, kAxisYP},
			{kButtonId_StickL_Down, kAxisYN},
			{kButtonId_StickL_Left, kAxisXN},
			{kButtonId_StickL_Right, kAxisXP},

			{kButtonId_StickR_Up, kRotationYP},
			{kButtonId_StickR_Down, kRotationYN},
			{kButtonId_StickR_Left, kRotationXN},
			{kButtonId_StickR_Right, kRotationXP},
		};
		
		break;
	}
	}

	bool mapping_updated = false;
	std::for_each(mapping.cbegin(), mapping.cend(), [this, &controller, &mapping_updated](const auto& m)
		{
			if (m_mappings.find(m.first) == m_mappings.cend())
			{
				set_mapping(m.first, controller, m.second);
				mapping_updated = true;
			}
		});

	return mapping_updated;
}