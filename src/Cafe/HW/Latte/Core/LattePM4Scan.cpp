#include "Cafe/HW/Latte/Core/LattePM4Scan.h"
#include "Cafe/HW/Latte/Core/LattePM4.h"
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"

namespace
{
	// a register range written, as LatteCP_itSetRegistersGeneric writes it
	void WriteRegs(LattePM4Scan& scan, uint32 base, const uint32be* data, uint32 nWords)
	{
		if (nWords < 1)
			return;
		const uint32 start = base + (uint32)data[0];
		const uint32 count = nWords - 1;
		if (start >= LATTE_MAX_REGISTER || count > LATTE_MAX_REGISTER - start)
			return;
		for (uint32 i = 0; i < count; i++)
			scan.regs[start + i] = (uint32)data[1 + i];
		// the command processor's one special range
		if (base == LATTE_REG_BASE_CONTEXT && start <= mmSQ_VTX_SEMANTIC_CLEAR && start + count >= mmSQ_VTX_SEMANTIC_CLEAR)
		{
			for (uint32 i = 0; i < 32; i++)
				scan.regs[mmSQ_VTX_SEMANTIC_0 + i] = 0xFF;
		}
	}

	// registers loaded from guest memory, as LatteCP_itLoadReg loads them
	bool LoadRegs(LattePM4Scan& scan, uint32 base, const uint32be* data, uint32 nWords)
	{
		if (nWords < 2 || (nWords & 1) != 0)
			return true;
		uint32 memAddr = (uint32)data[0];
		const uint32 entries = (nWords - 2) / 2;
		for (uint32 e = 0; e < entries; e++)
		{
			const uint32 start = base + (uint32)data[2 + e * 2];
			const uint32 count = (uint32)data[3 + e * 2];
			if (start >= LATTE_MAX_REGISTER || count > LATTE_MAX_REGISTER - start)
				return false;
			const uint32be* src = count ? scan.map(scan.ctx, memAddr, count) : nullptr;
			if (count && !src)
				return false;
			for (uint32 i = 0; i < count; i++)
				scan.regs[start + i] = (uint32)src[i];
			memAddr += count * 4;
		}
		return true;
	}

	struct Frame
	{
		const uint32be* cmd;
		const uint32be* end;
	};
}

void LattePM4_Scan(LattePM4Scan& scan, const uint32be* words, uint32 count)
{
	scan.draws = 0;
	scan.end = LattePM4Scan::End::Done;
	Frame stack[8];
	const uint32 maxDepth = scan.maxDepth < 7 ? scan.maxDepth : 7;
	uint32 depth = 0;
	stack[0] = {words, words + count};
	uint32 packets = 0;
	while (true)
	{
		Frame& f = stack[depth];
		if (f.cmd >= f.end)
		{
			if (depth == 0)
				return;
			depth--;
			continue;
		}
		if (++packets > scan.maxPackets)
		{
			scan.end = LattePM4Scan::End::Limit;
			return;
		}
		const uint32 header = (uint32)*f.cmd++;
		const uint32 type = (header >> 30) & 3;
		if (type == 2)
			continue; // filler
		if (type == 0)
		{
			// the two register writes the command processor takes (timestamps)
			const uint32 regBase = header & 0xFFFF;
			const uint32 n = ((header >> 16) & 0x3FFF) + 1;
			if ((regBase != 0x304A && regBase != 0x304B) || n > (uint32)(f.end - f.cmd))
			{
				scan.end = LattePM4Scan::End::Unreadable;
				return;
			}
			f.cmd += n;
			continue;
		}
		if (type != 3)
		{
			scan.end = LattePM4Scan::End::Unreadable;
			return;
		}
		const uint32 it = (header >> 8) & 0xFF;
		const uint32 nWords = ((header >> 16) & 0x3FFF) + 1;
		if (nWords > (uint32)(f.end - f.cmd))
		{
			scan.end = LattePM4Scan::End::Unreadable;
			return;
		}
		const uint32be* data = f.cmd;
		f.cmd += nWords;
		switch (it)
		{
		case IT_SET_CONTEXT_REG:
		case IT_SET_ALL_CONTEXTS:
			WriteRegs(scan, LATTE_REG_BASE_CONTEXT, data, nWords);
			break;
		case IT_SET_RESOURCE:
			WriteRegs(scan, LATTE_REG_BASE_RESOURCE, data, nWords);
			break;
		case IT_SET_ALU_CONST:
			WriteRegs(scan, LATTE_REG_BASE_ALU_CONST, data, nWords);
			break;
		case IT_SET_CTL_CONST:
			WriteRegs(scan, mmSQ_VTX_BASE_VTX_LOC, data, nWords);
			break;
		case IT_SET_SAMPLER:
			WriteRegs(scan, LATTE_REG_BASE_SAMPLER, data, nWords);
			break;
		case IT_SET_CONFIG_REG:
			WriteRegs(scan, LATTE_REG_BASE_CONFIG, data, nWords);
			break;
		case IT_LOAD_CONFIG_REG:
		case IT_LOAD_CONTEXT_REG:
		case IT_LOAD_ALU_CONST:
		case IT_LOAD_LOOP_CONST:
		case IT_LOAD_RESOURCE:
		case IT_LOAD_SAMPLER:
		{
			const uint32 base = it == IT_LOAD_CONFIG_REG ? LATTE_REG_BASE_CONFIG
				: it == IT_LOAD_CONTEXT_REG ? LATTE_REG_BASE_CONTEXT
				: it == IT_LOAD_ALU_CONST ? LATTE_REG_BASE_ALU_CONST
				: it == IT_LOAD_LOOP_CONST ? LATTE_REG_BASE_LOOP_CONST
				: it == IT_LOAD_RESOURCE ? LATTE_REG_BASE_RESOURCE
				: LATTE_REG_BASE_SAMPLER;
			if (!LoadRegs(scan, base, data, nWords))
			{
				scan.end = LattePM4Scan::End::Unreadable;
				return;
			}
			break;
		}
		case IT_INDIRECT_BUFFER_PRIV:
		{
			if (nWords < 3)
				break;
			const uint32 addr = (uint32)data[0];
			const uint32 size = (uint32)data[2];
			if (size == 0)
				break;
			if (depth >= maxDepth)
			{
				scan.end = LattePM4Scan::End::Limit;
				return;
			}
			const uint32be* sub = scan.map(scan.ctx, addr, size);
			if (!sub)
			{
				scan.end = LattePM4Scan::End::Unreadable;
				return;
			}
			stack[++depth] = {sub, sub + size};
			break;
		}
		case IT_DRAW_INDEX_2:
		case IT_DRAW_INDEX_AUTO:
		case IT_DRAW_INDEX_IMMD:
			scan.draws++;
			if (!scan.onDraw(scan.ctx, scan.regs))
			{
				scan.end = LattePM4Scan::End::Stopped;
				return;
			}
			break;
		case IT_WAIT_REG_MEM:
		case IT_MEM_SEMAPHORE:
		case IT_HLE_WAIT_FOR_FLIP:
		case IT_HLE_SYNC_ASYNC_OPERATIONS:
			// what follows may be written only once this is passed
			scan.end = LattePM4Scan::End::Waited;
			return;
		default:
			// the rest writes no register the reader keeps
			break;
		}
	}
}

uint64 LattePM4_PixelShaderInputs(const uint32* regs)
{
	static constexpr struct { uint32 start, count; } kRead[] = {
		{mmCB_SHADER_MASK, 1},
		{Latte::REGADDR::SX_ALPHA_TEST_CONTROL, 1},
		{mmSPI_VS_OUT_ID_0, 10},
		{mmSPI_PS_INPUT_CNTL_0, 32},
		{mmSPI_PS_IN_CONTROL_0, 3}, // and _1, SPI_INTERP_CONTROL_0
		{mmCB_SHADER_CONTROL, 1},
		{mmDB_SHADER_CONTROL, 4}, // PA_CL_CLIP_CNTL, PA_CL_VTE_CNTL
		{mmSQ_PGM_START_PS, 6}, // its size, resources, exports
		{Latte::REGADDR::VGT_GS_MODE, 1},
		{mmVGT_GS_OUT_PRIM_TYPE, 1},
		{mmVGT_STRMOUT_EN, 1},
		{Latte::REGADDR::VGT_PRIMITIVE_TYPE, 1},
		{Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS, Latte::GPU_LIMITS::NUM_TEXTURES_PER_STAGE * 7},
	};
	uint64 h = 0xcbf29ce484222325ull;
	for (const auto& r : kRead)
		for (uint32 i = 0; i < r.count; i++)
		{
			h ^= regs[r.start + i];
			h *= 0x100000001b3ull;
			h ^= h >> 29;
		}
	return h;
}
