// The shader decompiler's two Vulkan outputs against each other: random
// R600 pixel shaders - ALU clauses of every operation, operand kind and
// modifier the SPIR-V emitter covers (and some it declines), with PV/PS
// forwarding, register backups within a group, literals, uniforms, omod,
// clamp, reductions, kills and alpha tests - each decompiled to GLSL and
// compiled by rslc as RendererShaderVk compiles it, and written as SPIR-V
// directly (LatteDecompilerEmitSPIRV.cpp). Both are drawn on a real driver
// (lavapipe in CI) with the same inputs and uniforms over a 32 bit float
// target, and every pixel must be the same to the bit (a NaN may be another
// NaN, as the driver's optimizer has it); the uniform offsets
// the renderer writes with must be the same too. A shader the emitter
// declines must still compile through GLSL. The direct modules are written
// to the directory given, for spirv-val (run_tests.sh). Skipped without a
// device - a failure with VN_REQUIRE_DEVICE set, as the CIs set it.
#include "Cafe/HW/Latte/Core/Latte.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/Core/LatteShaderAssembly.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "Cafe/HW/Latte/ISA/LatteReg.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompilerInstructions.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/RslcGlsl.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanAPI.h"
#include "Cafe/HW/Latte/Core/FetchShader.h"

#include <gfx/shaders/rslc.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

extern LatteShaderPSInputTable g_testPSInputTable;
extern bool g_testPreventInfiniteLoops;
extern bool g_testRoundToPrecision; // the texel edge nudge's titles

static int s_failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); s_failures++; } } while (0)
#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::printf("FAIL %s:%d: %s = %d\n", __FILE__, __LINE__, #x, (int)r_); std::exit(1); } } while (0)

// --- an R600 pixel shader, assembled ------------------------------------------

struct AluOp
{
	bool op3;
	uint32 opcode;
	uint32 src[3];   // sel
	uint8 chan[3];
	uint8 neg[3];
	uint8 rel[3];    // relative to AR by indexMode
	uint8 abs[2];
	uint8 indexMode, destRel;
	uint8 destGpr, destElem, writeMask, clamp, omod;
	uint8 exec; // update execute mask (a predicate's, in a break clause)
};

struct AluGroup
{
	std::vector<AluOp> ops;
	uint32 literals[4];
};

// a texture instruction: a sample of a unit, from a register, into one
struct TexOp
{
	uint8 opcode, unit, srcGpr, dstGpr;
	uint8 srcSel[4], dstSel[4];
	sint8 offset[3];
	bool unnormalized; // texel coordinates
	// a VFETCH (the unit the buffer, 0x9F the ring) at this byte offset
	bool vfetch;
	uint16 vtxOffset;
	// a scatter buffer read (TEX MEM, memory op 2) of this format (0: none)
	uint8 memFormat;
};

// a CF instruction: an ALU or TEX clause of a kind, or one without a clause
struct CFEntry
{
	uint32 type;
	int clause; // into Program::clauses (ALU kinds) or Program::texClauses (TEX)
	uint8 popCount;
};

// the texture units' kinds, as the test makes their textures
enum TexKind : uint8 { TEX_NONE, TEX_2D, TEX_2D_ARRAY, TEX_3D, TEX_2D_UINT, TEX_1D, TEX_1D_UINT, TEX_CUBE, TEX_KINDS };

struct Program
{
	std::vector<std::vector<AluGroup>> clauses; // ALU clauses
	std::vector<std::vector<TexOp>> texClauses;
	std::vector<CFEntry> cf; // the CF list; none: each clause a plain ALU, in order
	// a subroutine: plain ALU and TEX entries placed after the program's
	// end, then a RETURN; each CALL in the CF list calls it
	std::vector<CFEntry> sub;
	uint8 texKind[4]; // units 0 to 3
	uint32 exportGpr;
	uint8 exportSel[4];
	// a vertex shader's exports, in order, in place of the pixel export:
	// type 1 (60 the position) or 2 (a parameter), the last one done; or
	// type 3, a MEM_RING_WRITE to the geometry shader (base in dwords, the
	// channel mask in sel[0]); type 4, a geometry shader's EMIT_VERTEX;
	// type 5, a MEM_STREAM<sel[1]>_WRITE (base in dwords of the vertex, the
	// mask in sel[0], the array size - 1 in sel[2])
	struct VsExport
	{
		uint8 type, gpr, burst;
		uint16 base;
		uint8 sel[4];
	};
	// each ALU clause's kcache: the banks' uniform buffers and their bases
	// (in 16 vec4 units); none: banks 0 at 0
	struct Kcache
	{
		uint8 bank[2], base[2];
	};
	std::vector<Kcache> kcache;
	bool vertex = false;
	bool geometry = false; // a geometry shader's (with vertex set)
	std::vector<VsExport> vsExports;
	// ring writes in the CF list (a CFEntry of MEM_RING_WRITE, its clause
	// the index here), to sit between clauses as EMIT_VERTEX entries do
	std::vector<VsExport> cfRing;
	std::vector<float> inputs; // a directed program's, in place of random ones
	// a directed program's pixel 0 x and y: uniform register c3.x and
	// bank 0's (buffer 0's) vec4 3's y, checked as drawn
	bool expectRemap = false;
	// a directed program's pixel 0: buffer 1's vec4 5's z, then buffer 2's
	// vec4 9's w, x and y (uniform buffer VFETCHes), checked as drawn
	bool expectFetch = false;
	// a directed program's pixel 0, checked as drawn
	bool expectPixel = false;
	uint32 pixel0[4];
	// a pixel shader's depth export (-1: none): R<depthGpr>.<depthSel>,
	// after the color; a directed program's pixel 0 depth, checked
	sint8 depthGpr = -1;
	uint8 depthSel = 0;
	bool expectDepth = false;
	float depth0 = 0.0f;
	// a directed program's system inputs: input 1 gl_FragCoord (scaled by
	// (2, 4)), the front face into R0.y; pixel 0 then (1, 2) and the face
	bool sysInputs = false;
	// a directed program's: drawn as a texel edge title (the nudge on)
	bool roundPrecision = false;
	// a directed program's: drawn as a point, param gen into this register
	sint8 pointCoordGpr = -1;
	// a directed program's pixel 0: x equal to z and y to w
	bool expectPairs = false;
};

static void EncodeTex(const TexOp& t, uint32 w[4])
{
	if (t.memFormat)
	{
		// MEM_RD_WORD0-2: memory op 2 (a scatter buffer read), the source
		// register's channel, the destination's selects and the format
		w[0] = (uint32)GPU7_TEX_INST_MEM | (2u << 8) | ((uint32)t.srcGpr << 16) | (((uint32)t.srcSel[0] & 3) << 24);
		w[1] = (uint32)t.dstGpr | ((uint32)t.dstSel[0] << 9) | ((uint32)t.dstSel[1] << 12) | ((uint32)t.dstSel[2] << 15) |
			((uint32)t.dstSel[3] << 18) | ((uint32)t.memFormat << 22);
		w[2] = 0;
		w[3] = 0;
		return;
	}
	if (t.vfetch)
	{
		// VTX_WORD0-2: fetch type 2, the buffer, source R<srcGpr>.<srcSel[0]>
		w[0] = (uint32)GPU7_TEX_INST_VFETCH | (2u << 5) | ((uint32)t.unit << 8) | ((uint32)t.srcGpr << 16) |
			(((uint32)t.srcSel[0] & 3) << 24);
		w[1] = (uint32)t.dstGpr | ((uint32)t.dstSel[0] << 9) | ((uint32)t.dstSel[1] << 12) | ((uint32)t.dstSel[2] << 15) |
			((uint32)t.dstSel[3] << 18);
		w[2] = t.vtxOffset;
		w[3] = 0;
		return;
	}
	w[0] = (uint32)t.opcode | ((uint32)t.unit << 8) | ((uint32)t.srcGpr << 16);
	w[1] = (uint32)t.dstGpr | ((uint32)t.dstSel[0] << 9) | ((uint32)t.dstSel[1] << 12) | ((uint32)t.dstSel[2] << 15) |
		((uint32)t.dstSel[3] << 18) | (t.unnormalized ? 0u : 0xF0000000u);
	w[2] = ((uint32)t.offset[0] & 0x1F) | (((uint32)t.offset[1] & 0x1F) << 5) | (((uint32)t.offset[2] & 0x1F) << 10) |
		((uint32)t.unit << 15) | ((uint32)t.srcSel[0] << 20) | ((uint32)t.srcSel[1] << 23) | ((uint32)t.srcSel[2] << 26) |
		((uint32)t.srcSel[3] << 29);
	w[3] = 0;
}

// the units' dimension and number format into the pixel shader's texture
// registers, as the decompiler reads them
static void SetTextureRegisters(uint32* regs, const Program& p)
{
	LatteContextRegister& lcr = *(LatteContextRegister*)regs;
	auto* units = p.geometry ? lcr.SQ_TEX_START_GS : p.vertex ? lcr.SQ_TEX_START_VS : lcr.SQ_TEX_START_PS;
	for (int u = 0; u < 4; u++)
	{
		Latte::E_DIM dim = Latte::E_DIM::DIM_2D;
		switch (p.texKind[u])
		{
		case TEX_2D_ARRAY: dim = Latte::E_DIM::DIM_2D_ARRAY; break;
		case TEX_3D: dim = Latte::E_DIM::DIM_3D; break;
		case TEX_CUBE: dim = Latte::E_DIM::DIM_CUBEMAP; break;
		case TEX_1D:
		case TEX_1D_UINT: dim = Latte::E_DIM::DIM_1D; break;
		default: break;
		}
		units[u].word0.set_DIM(dim);
		units[u].word4.set_NUM_FORM_ALL(p.texKind[u] == TEX_2D_UINT || p.texKind[u] == TEX_1D_UINT ?
			Latte::LATTE_SQ_TEX_RESOURCE_WORD4_N::E_NUM_FORMAT_ALL::NUM_FORMAT_INT :
			Latte::LATTE_SQ_TEX_RESOURCE_WORD4_N::E_NUM_FORMAT_ALL::NUM_FORMAT_NORM);
	}
}

static void EncodeAlu(const AluOp& op, bool last, uint32& w0, uint32& w1)
{
	w0 = (op.src[0] & 0x1FF) | ((uint32)op.rel[0] << 9) | ((uint32)op.chan[0] << 10) | ((uint32)op.neg[0] << 12) |
		((op.src[1] & 0x1FF) << 13) | ((uint32)op.rel[1] << 22) | ((uint32)op.chan[1] << 23) | ((uint32)op.neg[1] << 25) |
		((uint32)op.indexMode << 26) | (last ? 0x80000000u : 0);
	if (op.op3)
	{
		w1 = (op.src[2] & 0x1FF) | ((uint32)op.rel[2] << 9) | ((uint32)op.chan[2] << 10) | ((uint32)op.neg[2] << 12) | (op.opcode << 13) |
			((uint32)op.destGpr << 21) | ((uint32)op.destRel << 28) | ((uint32)op.destElem << 29) | ((uint32)op.clamp << 31);
	}
	else
	{
		w1 = (uint32)op.abs[0] | ((uint32)op.abs[1] << 1) | ((uint32)op.exec << 2) | ((uint32)op.writeMask << 4) | ((uint32)op.omod << 5) |
			(op.opcode << 7) | ((uint32)op.destGpr << 21) | ((uint32)op.destRel << 28) | ((uint32)op.destElem << 29) | ((uint32)op.clamp << 31);
	}
}

// CF list, then the clauses' words (64 bit units)
static std::vector<uint32> Assemble(const Program& p)
{
	std::vector<uint32> alu;
	std::vector<uint32> cf;
	std::vector<CFEntry> list = p.cf;
	if (list.empty())
	{
		for (size_t i = 0; i < p.clauses.size(); i++)
			list.push_back({ GPU7_CF_INST_ALU, (int)i, 0 });
	}
	const uint32 cfCount = (uint32)list.size() + (p.vertex ? (uint32)p.vsExports.size() : (p.depthGpr >= 0 ? 2 : 1)) +
		(p.sub.empty() ? 0 : (uint32)p.sub.size() + 1);
	uint32 aluBase = (cfCount + 1) & ~1u; // in 64 bit words, after the CF list
	uint32 at = aluBase;
	std::vector<uint32> clauseStart, clauseWords;
	for (auto& clause : p.clauses)
	{
		uint32 start = at;
		uint32 words = 0;
		for (auto& g : clause)
		{
			bool usesLiteral = false, usesZW = false;
			for (size_t i = 0; i < g.ops.size(); i++)
			{
				uint32 w0, w1;
				EncodeAlu(g.ops[i], i + 1 == g.ops.size(), w0, w1);
				alu.push_back(w0);
				alu.push_back(w1);
				words++;
				const AluOp& o = g.ops[i];
				for (int s = 0; s < (o.op3 ? 3 : 2); s++)
				{
					if (o.src[s] == 253)
					{
						usesLiteral = true;
						usesZW |= o.chan[s] >= 2;
					}
				}
			}
			if (usesLiteral)
			{
				alu.push_back(g.literals[0]);
				alu.push_back(g.literals[1]);
				words++;
				if (usesZW)
				{
					alu.push_back(g.literals[2]);
					alu.push_back(g.literals[3]);
					words++;
				}
			}
		}
		at += words;
		clauseStart.push_back(start);
		clauseWords.push_back(words);
	}
	// the TEX clauses, 128 bit instructions, after the ALU clauses
	std::vector<uint32> tex;
	std::vector<uint32> texStart;
	at = (at + 1) & ~1u;
	if (alu.size() & 2)
	{
		alu.push_back(0);
		alu.push_back(0);
	}
	for (auto& clause : p.texClauses)
	{
		texStart.push_back(at);
		for (auto& t : clause)
		{
			uint32 w[4];
			EncodeTex(t, w);
			tex.insert(tex.end(), w, w + 4);
			at += 2;
		}
	}
	std::vector<size_t> calls; // the CALLs' first words, the address to come
	auto entry = [&](const CFEntry& e) {
		if (e.type == GPU7_CF_INST_CALL)
		{
			// CALL: addr (the subroutine's), call count 0
			calls.push_back(cf.size());
			cf.push_back(0);
			cf.push_back(((uint32)GPU7_CF_INST_CALL << 23) | (1u << 31));
		}
		else if (e.type == GPU7_CF_INST_TEX)
		{
			// CF_TEX: addr, count-1 (bits 10-12 and 19)
			const uint32 count = (uint32)p.texClauses[e.clause].size() - 1;
			cf.push_back(texStart[e.clause]);
			cf.push_back(((count & 7) << 10) | (((count >> 3) & 1) << 19) | ((uint32)GPU7_CF_INST_TEX << 23) | (1u << 31));
		}
		else if (e.type & GPU7_CF_INST_ALU_MASK)
		{
			// CF_ALU_*: addr, the kcache banks, count-1, their bases, the kind
			Program::Kcache kc{};
			if ((size_t)e.clause < p.kcache.size())
				kc = p.kcache[e.clause];
			cf.push_back(clauseStart[e.clause] | ((uint32)kc.bank[0] << 22) | ((uint32)kc.bank[1] << 26));
			cf.push_back(((uint32)kc.base[0] << 2) | ((uint32)kc.base[1] << 10) | ((clauseWords[e.clause] - 1) << 18) |
				((e.type & 0xF) << 26) | (1u << 31));
		}
		else if (e.type == GPU7_CF_INST_MEM_RING_WRITE)
		{
			const Program::VsExport& r = p.cfRing[e.clause];
			cf.push_back((uint32)r.base | ((uint32)r.gpr << 15) | (3u << 30));
			cf.push_back(((uint32)r.sel[0] << 12) | ((uint32)r.burst << 17) | ((uint32)GPU7_CF_INST_MEM_RING_WRITE << 23) | (1u << 31));
		}
		else
		{
			cf.push_back(0);
			cf.push_back((uint32)e.popCount | (e.type << 23) | (1u << 31));
		}
	};
	for (auto& e : list)
		entry(e);
	// a program of the CF list alone ends with its last entry
	if (p.vertex && p.vsExports.empty() && !cf.empty())
		cf.back() |= 1u << 21;
	if (p.vertex)
	{
		for (size_t i = 0; i < p.vsExports.size(); i++)
		{
			const Program::VsExport& e = p.vsExports[i];
			const bool last = i + 1 == p.vsExports.size();
			if (e.type == 5)
			{
				cf.push_back((uint32)e.base | ((uint32)e.gpr << 15) | (3u << 30));
				cf.push_back((uint32)e.sel[2] | ((uint32)e.sel[0] << 12) | (last ? (1u << 21) : 0) |
					((uint32)(e.sel[1] ? GPU7_CF_INST_MEM_STREAM1_WRITE : GPU7_CF_INST_MEM_STREAM0_WRITE) << 23) | (1u << 31));
				continue;
			}
			if (e.type == 4)
			{
				cf.push_back(0);
				cf.push_back((last ? (1u << 21) : 0) | ((uint32)GPU7_CF_INST_EMIT_VERTEX << 23) | (1u << 31));
				continue;
			}
			if (e.type == 3)
			{
				cf.push_back((uint32)e.base | ((uint32)e.gpr << 15) | (3u << 30));
				cf.push_back(((uint32)e.sel[0] << 12) | ((uint32)e.burst << 17) | (last ? (1u << 21) : 0) |
					((uint32)GPU7_CF_INST_MEM_RING_WRITE << 23) | (1u << 31));
				continue;
			}
			cf.push_back((uint32)e.base | ((uint32)e.type << 13) | ((uint32)e.gpr << 15));
			cf.push_back((uint32)e.sel[0] | ((uint32)e.sel[1] << 3) | ((uint32)e.sel[2] << 6) | ((uint32)e.sel[3] << 9) |
				((uint32)e.burst << 17) | (last ? (1u << 21) : 0) |
				((uint32)(last ? GPU7_CF_INST_EXPORT_DONE : GPU7_CF_INST_EXPORT) << 23) | (1u << 31));
		}
	}
	else
	{
		// EXPORT_DONE, pixel, color 0, end of program; or an EXPORT of it,
		// then the depth (array base 61, its channel then none)
		const bool depth = p.depthGpr >= 0;
		cf.push_back(0 | (0 << 13) | (p.exportGpr << 15));
		cf.push_back((uint32)p.exportSel[0] | ((uint32)p.exportSel[1] << 3) | ((uint32)p.exportSel[2] << 6) | ((uint32)p.exportSel[3] << 9) |
			(depth ? 0 : (1u << 21)) | ((uint32)(depth ? GPU7_CF_INST_EXPORT : GPU7_CF_INST_EXPORT_DONE) << 23) | (1u << 31));
		if (depth)
		{
			cf.push_back(61u | ((uint32)p.depthGpr << 15));
			cf.push_back((uint32)p.depthSel | (7u << 3) | (7u << 6) | (7u << 9) | (1u << 21) | ((uint32)GPU7_CF_INST_EXPORT_DONE << 23) | (1u << 31));
		}
	}
	// the subroutine past the end of the program
	if (!p.sub.empty())
	{
		const uint32 subAddr = (uint32)cf.size() / 2;
		for (size_t c : calls)
			cf[c] = subAddr;
		for (auto& e : p.sub)
			entry(e);
		entry({ GPU7_CF_INST_RETURN, -1, 0 });
	}
	std::vector<uint32> out(aluBase * 2, 0);
	std::memcpy(out.data(), cf.data(), cf.size() * 4);
	out.insert(out.end(), alu.begin(), alu.end());
	out.insert(out.end(), tex.begin(), tex.end());
	out.push_back(0);
	out.push_back(0);
	return out;
}

// --- random programs ---------------------------------------------------------

static const uint32 kVectorOps[] = {
	ALU_OP2_INST_ADD, ALU_OP2_INST_MUL, ALU_OP2_INST_MUL_IEEE, ALU_OP2_INST_MAX, ALU_OP2_INST_MIN, ALU_OP2_INST_MAX_DX10,
	ALU_OP2_INST_MIN_DX10, ALU_OP2_INST_SETE, ALU_OP2_INST_SETGT, ALU_OP2_INST_SETGE, ALU_OP2_INST_SETNE, ALU_OP2_INST_SETE_DX10,
	ALU_OP2_INST_SETGT_DX10, ALU_OP2_INST_SETGE_DX10, ALU_OP2_INST_SETNE_DX10, ALU_OP2_INST_FLOOR, ALU_OP2_INST_FRACT,
	ALU_OP2_INST_TRUNC, ALU_OP2_INST_RNDNE, ALU_OP2_INST_MOV, ALU_OP2_INST_MOV, ALU_OP2_INST_MOV, ALU_OP2_INST_NOP,
	ALU_OP2_INST_AND_INT, ALU_OP2_INST_OR_INT, ALU_OP2_INST_XOR_INT, ALU_OP2_INST_NOT_INT, ALU_OP2_INST_ADD_INT,
	ALU_OP2_INST_SUB_INT, ALU_OP2_INST_MAX_INT, ALU_OP2_INST_MIN_INT, ALU_OP2_INST_MAX_UINT, ALU_OP2_INST_MIN_UINT,
	ALU_OP2_INST_SETE_INT, ALU_OP2_INST_SETGT_INT, ALU_OP2_INST_SETGE_INT, ALU_OP2_INST_SETNE_INT, ALU_OP2_INST_SETGT_UINT,
	ALU_OP2_INST_SETGE_UINT, ALU_OP2_INST_ASHR_INT, ALU_OP2_INST_LSHR_INT, ALU_OP2_INST_LSHL_INT, ALU_OP2_INST_KILLGT,
	ALU_OP2_INST_KILLGE, ALU_OP2_INST_KILLE, ALU_OP2_INST_KILLE_INT, ALU_OP2_INST_KILLGT_INT, ALU_OP2_INST_KILLNE_INT,
};
static const uint32 kTransOps[] = {
	ALU_OP2_INST_EXP_IEEE, ALU_OP2_INST_LOG_CLAMPED, ALU_OP2_INST_LOG_IEEE, ALU_OP2_INST_SQRT_IEEE, ALU_OP2_INST_SIN,
	ALU_OP2_INST_COS, ALU_OP2_INST_RECIP_CLAMPED, ALU_OP2_INST_RECIP_FF, ALU_OP2_INST_RECIP_IEEE,
	ALU_OP2_INST_RECIPSQRT_CLAMPED, ALU_OP2_INST_RECIPSQRT_FF, ALU_OP2_INST_RECIPSQRT_IEEE, ALU_OP2_INST_MULLO_INT,
	ALU_OP2_INST_MULLO_UINT, ALU_OP2_INST_FLT_TO_INT, ALU_OP2_INST_FLT_TO_UINT, ALU_OP2_INST_INT_TO_FLOAT,
	ALU_OP2_INST_UINT_TO_FLOAT,
};
static const uint32 kOp3s[] = {
	ALU_OP3_INST_MULADD, ALU_OP3_INST_MULADD_M2, ALU_OP3_INST_MULADD_M4, ALU_OP3_INST_MULADD_D2, ALU_OP3_INST_MULADD_IEEE,
	ALU_OP3_INST_CMOVE, ALU_OP3_INST_CMOVGT, ALU_OP3_INST_CMOVGE, ALU_OP3_INST_CNDE_INT, ALU_OP3_INST_CNDGT_INT,
	ALU_OP3_INST_CMOVGE_INT,
};
static const uint32 kMova[] = { ALU_OP2_INST_MOVA_INT, ALU_OP2_INST_MOVA_FLOOR };
static const uint32 kPredOps[] = {
	ALU_OP2_INST_PRED_SETE, ALU_OP2_INST_PRED_SETGT, ALU_OP2_INST_PRED_SETGE, ALU_OP2_INST_PRED_SETNE,
	ALU_OP2_INST_PRED_SETE_INT, ALU_OP2_INST_PRED_SETGT_INT, ALU_OP2_INST_PRED_SETGE_INT, ALU_OP2_INST_PRED_SETNE_INT,
};

static bool IsKill(uint32 opcode)
{
	return opcode == ALU_OP2_INST_KILLE || opcode == ALU_OP2_INST_KILLGT || opcode == ALU_OP2_INST_KILLGE ||
		opcode == ALU_OP2_INST_KILLE_INT || opcode == ALU_OP2_INST_KILLGT_INT || opcode == ALU_OP2_INST_KILLNE_INT;
}

static const int kGPRs = 8;
static const int kCounterGPR = kGPRs - 1; // a loop's counter, written by nothing else
static const int kUniforms = 12;

// the type an operation reads an operand as (what a constant may be)
enum { T_F, T_I, T_U };
static int OperandType(const AluOp& op, int s)
{
	if (op.op3)
	{
		switch (op.opcode)
		{
		case ALU_OP3_INST_CMOVE: case ALU_OP3_INST_CMOVGT: case ALU_OP3_INST_CMOVGE:
			return s == 0 ? T_F : T_I;
		case ALU_OP3_INST_CNDE_INT: case ALU_OP3_INST_CNDGT_INT: case ALU_OP3_INST_CMOVGE_INT:
			return T_I;
		default:
			return T_F;
		}
	}
	switch (op.opcode)
	{
	case ALU_OP2_INST_AND_INT: case ALU_OP2_INST_OR_INT: case ALU_OP2_INST_XOR_INT: case ALU_OP2_INST_NOT_INT:
	case ALU_OP2_INST_ADD_INT: case ALU_OP2_INST_SUB_INT: case ALU_OP2_INST_MAX_INT: case ALU_OP2_INST_MIN_INT:
	case ALU_OP2_INST_SETE_INT: case ALU_OP2_INST_SETGT_INT: case ALU_OP2_INST_SETGE_INT: case ALU_OP2_INST_SETNE_INT:
	case ALU_OP2_INST_ASHR_INT: case ALU_OP2_INST_LSHL_INT: case ALU_OP2_INST_MULLO_INT: case ALU_OP2_INST_INT_TO_FLOAT:
	case ALU_OP2_INST_KILLE_INT: case ALU_OP2_INST_KILLGT_INT: case ALU_OP2_INST_KILLNE_INT: case ALU_OP2_INST_MOVA_INT:
	case ALU_OP2_INST_PRED_SETE_INT: case ALU_OP2_INST_PRED_SETGT_INT: case ALU_OP2_INST_PRED_SETGE_INT: case ALU_OP2_INST_PRED_SETNE_INT:
		return T_I;
	case ALU_OP2_INST_MAX_UINT: case ALU_OP2_INST_MIN_UINT: case ALU_OP2_INST_SETGT_UINT: case ALU_OP2_INST_SETGE_UINT:
	case ALU_OP2_INST_LSHR_INT: case ALU_OP2_INST_MULLO_UINT: case ALU_OP2_INST_UINT_TO_FLOAT:
		return T_U;
	default:
		return T_F;
	}
}

struct Gen
{
	std::mt19937 rng;
	int destGPRs = kGPRs; // without the counter's, in a program with loops
	// no kill in a loop: lavapipe (Mesa 25.2) keeps a discarded pixel in
	// the loop, where it never reaches the counted break, and crashes -
	// the module rslc makes of the GLSL as well as the direct one
	bool noKills = false;
	// one loop at most, and no kill (nor alpha test) after it: lavapipe
	// (Mesa 25.2) crashes or hangs on some shaders with a loop in a loop,
	// a loop after one some pixels broke out of, or a discard after such
	// a loop - the module rslc makes as well as the direct one (directed
	// programs keep a loop in a loop known to draw). A vertex shader: no
	// kill
	bool vertexMode = false;
	// the uniforms from uniform buffers (the clauses' kcache banks), not
	// the uniform registers; dynamic reads of them (relative to AR, or
	// VFETCH) make the whole buffers the shader's
	bool bufferMode = false;
	bool loopMade = false;
	// clauses inside a loop: no register written relative to AR there, which
	// could land on the loop's counter and keep it from ending
	bool inLoop = false;
	bool shadowUnit[4]{}; // the units compared with: a 2D's or a 2D array's
	// the register exported, written more often where control flow decides
	// which clauses run, so that it shows
	int favorDest = -1;
	uint8 Dest() { return (uint8)((favorDest >= 0 && Chance(35)) ? favorDest : (int)R(destGPRs)); }
	uint32 R(uint32 n) { return rng() % n; }
	bool Chance(uint32 percent) { return R(100) < percent; }
	// the pixel shader's system values (gl_FragCoord, the front face, a
	// depth export) from a stream of their own: the programs and every
	// other choice stay as each seed made them
	std::mt19937 sys;
	uint32 SysR(uint32 n) { return sys() % n; }
	bool SysChance(uint32 percent) { return SysR(100) < percent; }

	// a source: what was written in the previous group (PV/PS), a GPR, a
	// constant, a literal or a uniform
	void Source(AluOp& op, int s, const bool pvWritten[5], bool firstGroup)
	{
		uint32 kind = R(100);
		op.chan[s] = (uint8)R(4);
		// a register or uniform relative to AR's x to w, now and then
		if ((kind < 45 || (kind >= 68 && kind < 85)) && Chance(6))
		{
			op.rel[s] = 1;
			op.indexMode = (uint8)R(4);
		}
		if (kind < 45)
			op.src[s] = R(kGPRs);
		else if (kind < 55)
		{
			// the constants the operand's type takes (an integer one read as
			// a float is a GLSL error, which the emitter declines)
			static const uint32 f[] = { 248, 249, 252 };
			static const uint32 i[] = { 248, 249, 250, 251, 252 };
			static const uint32 u[] = { 248, 249, 250, 252 };
			int t = OperandType(op, s);
			if (op.neg[s] || (s < 2 && !op.op3 && op.abs[s]))
				t = T_F;
			op.src[s] = t == T_F ? f[R(3)] : (t == T_I ? i[R(5)] : u[R(4)]);
		}
		else if (kind < 68)
			op.src[s] = 253;
		else if (kind < 85)
			op.src[s] = bufferMode ? (Chance(50) ? 128 : 160) + R(32) : 256 + R(kUniforms);
		else
		{
			uint8 units[5];
			int n = 0;
			for (int u = 0; u < 5; u++)
				if (pvWritten[u])
					units[n++] = (uint8)u;
			if (firstGroup || n == 0)
			{
				op.src[s] = R(kGPRs);
				return;
			}
			uint8 u = units[R(n)];
			if (u == 4)
				op.src[s] = 255;
			else
			{
				op.src[s] = 254;
				op.chan[s] = u;
			}
		}
	}

	uint32 Literal()
	{
		static const uint32 special[] = { 0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000,
			0x00000001, 0x7F7FFFFF, 0x3F000000, 0x40000000, 0xFFFFFFFF, 0x7FFFFFFF, 0x00000003, 0x41200000 };
		if (Chance(40))
			return special[R(sizeof(special) / 4)];
		if (Chance(50))
		{
			float f = (float)((int)R(2001) - 1000) / (float)(1 + R(64));
			uint32 b;
			std::memcpy(&b, &f, 4);
			return b;
		}
		return rng();
	}

	AluGroup Group(bool firstGroup, bool pvWritten[5], bool allowDeclined)
	{
		AluGroup g{};
		for (auto& l : g.literals)
			l = Literal();
		bool usedElem[4] = {};
		bool written[5] = {};
		// a DOT4 or a CUBE now and then: four ops, x to w
		if (Chance(10))
		{
			uint32 opc = Chance(30) ? ALU_OP2_INST_CUBE : Chance(50) ? ALU_OP2_INST_DOT4 : ALU_OP2_INST_DOT4_IEEE;
			uint8 dst = Dest();
			for (uint8 e = 0; e < 4; e++)
			{
				AluOp op{};
				op.opcode = opc;
				Source(op, 0, pvWritten, firstGroup);
				Source(op, 1, pvWritten, firstGroup);
				op.destGpr = dst;
				op.destElem = e;
				op.writeMask = (e == 0 || Chance(30)) ? 1 : 0;
				if (e == 0)
				{
					op.clamp = Chance(15);
					op.omod = Chance(15) ? (uint8)(1 + R(3)) : 0;
				}
				else
				{
					op.clamp = g.ops[0].clamp;
					op.omod = g.ops[0].omod;
				}
				g.ops.push_back(op);
				written[e] = true;
			}
			std::memcpy(pvWritten, written, sizeof(written));
			return g;
		}
		uint32 vectorCount = 1 + R(4);
		for (uint32 i = 0; i < vectorCount; i++)
		{
			uint8 e;
			do
				e = (uint8)R(4);
			while (usedElem[e]);
			usedElem[e] = true;
			AluOp op{};
			uint32 kind = R(100);
			if (kind < 3)
				op.opcode = kMova[R(2)]; // AR's x to w
		else if (kind < 6)
			op.opcode = kPredOps[R(8)];
			else if (kind < 25)
			{
				op.op3 = true;
				op.opcode = kOp3s[R(sizeof(kOp3s) / 4)];
			}
			else
				op.opcode = kVectorOps[R(sizeof(kVectorOps) / 4)];
			if (noKills && !op.op3 && IsKill(op.opcode))
				op.opcode = ALU_OP2_INST_MOV;
			for (int s = 0; s < (op.op3 ? 3 : 2); s++)
				op.neg[s] = Chance(12);
			if (!op.op3)
			{
				op.abs[0] = Chance(10);
				op.abs[1] = Chance(10);
			}
			for (int s = 0; s < (op.op3 ? 3 : 2); s++)
				Source(op, s, pvWritten, firstGroup);
			op.destGpr = Dest();
			op.destElem = e;
			op.writeMask = op.op3 ? 1 : (Chance(75) ? 1 : 0);
			// a register written relative to AR (the registers then one array),
			// now and then
			op.destRel = Chance(4) && !inLoop;
			if (op.destRel && !op.rel[0] && !op.rel[1] && !op.rel[2])
				op.indexMode = (uint8)R(4);
			const bool mova = !op.op3 && (op.opcode == ALU_OP2_INST_MOVA_INT || op.opcode == ALU_OP2_INST_MOVA_FLOOR);
			if (mova)
				op.writeMask = 0; // into PV, as the hardware writes
			if (!op.op3 && ((op.opcode >= ALU_OP2_INST_PRED_SETE && op.opcode <= ALU_OP2_INST_PRED_SETNE) ||
				(op.opcode >= ALU_OP2_INST_PRED_SETE_INT && op.opcode <= ALU_OP2_INST_PRED_SETNE_INT)))
			{
				op.writeMask = 0;
				op.exec = Chance(50);
			}
			op.clamp = mova ? 0 : Chance(10);
			op.omod = (!op.op3 && !mova && Chance(10)) ? (uint8)(1 + R(3)) : 0;
			g.ops.push_back(op);
			if (!(op.op3 == false && op.opcode == ALU_OP2_INST_NOP))
				written[e] = true;
		}
		if (Chance(40))
		{
			AluOp op{};
			op.opcode = kTransOps[R(sizeof(kTransOps) / 4)];
			op.neg[0] = Chance(10);
			op.abs[0] = Chance(10);
			Source(op, 0, pvWritten, firstGroup);
			Source(op, 1, pvWritten, firstGroup);
			op.destGpr = Dest();
			op.destElem = (uint8)R(4);
			op.writeMask = Chance(75) ? 1 : 0;
			op.clamp = Chance(10);
			op.omod = Chance(10) ? (uint8)(1 + R(3)) : 0;
			g.ops.push_back(op);
			written[4] = true;
		}
		std::memcpy(pvWritten, written, sizeof(written));
		return g;
	}

	// an ALU clause of up to 'maxGroups' groups, and a predicate at its end
	// for a push
	int Clause(Program& p, uint32 maxGroups, bool allowDeclined, bool predicate = false)
	{
		bool pv[5] = {};
		std::vector<AluGroup> groups;
		uint32 n = 1 + R(maxGroups);
		for (uint32 i = 0; i < n; i++)
			groups.push_back(Group(i == 0, pv, allowDeclined));
		if (predicate)
		{
			AluGroup g{};
			for (auto& l : g.literals)
				l = Literal();
			AluOp op{};
			op.opcode = kPredOps[R(8)];
			op.neg[0] = Chance(10);
			op.neg[1] = Chance(10);
			op.abs[0] = Chance(10);
			Source(op, 0, pv, false);
			Source(op, 1, pv, false);
			if (Chance(60))
			{
				// an input that varies over the target: pixels split
				static const uint32 varying[] = { 0, 1, 3 };
				op.src[0] = varying[R(3)];
				op.chan[0] = (uint8)R(2);
			}
			op.destElem = (uint8)R(4);
			op.exec = 1;
			g.ops.push_back(op);
			groups.push_back(g);
		}
		p.clauses.push_back(std::move(groups));
		// the clause's kcache: two banks, each at 0, 16 or 32
		Program::Kcache kc{};
		kc.bank[0] = (uint8)R(3);
		kc.bank[1] = (uint8)R(4);
		kc.base[0] = (uint8)R(3);
		kc.base[1] = (uint8)R(3);
		p.kcache.resize(p.clauses.size());
		p.kcache.back() = kc;
		return (int)p.clauses.size() - 1;
	}
	// a TEX clause: samples of every unit, register, select, written
	// channel, offset, level and bias; compares of the shadow units,
	// gathers, texel fetches and the queries; kinds the emitter declines,
	// and kinds of a unit it does not take, now and then
	int TexClause(Program& p, bool allowDeclined)
	{
		std::vector<TexOp> clause;
		uint32 n = 1 + R(3);
		for (uint32 i = 0; i < n; i++)
		{
			TexOp t{};
			if (SysChance(6))
			{
				// a scatter buffer read: one to three floats, into as many
				// channels or fewer (more: the GLSL emitter's vector is too
				// short, declined)
				static const uint8 formats[3] = { 0x0E, 0x1E, 0x30 }; // HWFMT_32, 32_32, 32_32_32 float
				const uint32 size = 1 + SysR(3);
				t.memFormat = formats[size - 1];
				t.opcode = GPU7_TEX_INST_MEM;
				t.srcGpr = (uint8)SysR(kGPRs);
				t.srcSel[0] = (uint8)SysR(4);
				t.dstGpr = (uint8)SysR(destGPRs); // not a loop's counter
				for (int c = 0; c < 4; c++)
					t.dstSel[c] = 7;
				const uint32 want = (allowDeclined && SysChance(20)) ? 4 : 1 + SysR(size);
				for (uint32 c = 0, placed = 0; placed < want && c < 4; c++)
				{
					if (SysChance(70) || 4 - c <= want - placed)
					{
						t.dstSel[c] = (uint8)SysR(4);
						placed++;
					}
				}
				clause.push_back(t);
				continue;
			}
			if (bufferMode && Chance(15))
			{
				// a uniform buffer's vec4 at a register's index
				t.vfetch = true;
				t.opcode = GPU7_TEX_INST_VFETCH;
				t.unit = (uint8)(0x80 + R(3));
				t.srcGpr = (uint8)R(kGPRs);
				t.srcSel[0] = (uint8)R(4);
				for (int c = 0; c < 4; c++)
					t.dstSel[c] = (uint8)(Chance(85) ? R(4) : 7);
				if (t.dstSel[0] == 7 && t.dstSel[1] == 7 && t.dstSel[2] == 7 && t.dstSel[3] == 7)
					t.dstSel[R(4)] = (uint8)R(4);
				t.dstGpr = Dest();
				clause.push_back(t);
				continue;
			}
			t.unit = (uint8)R(4);
			const uint8 kind = p.texKind[t.unit];
			static const uint8 samples[] = { GPU7_TEX_INST_SAMPLE, GPU7_TEX_INST_SAMPLE, GPU7_TEX_INST_SAMPLE, GPU7_TEX_INST_SAMPLE_L,
				GPU7_TEX_INST_SAMPLE_LZ, GPU7_TEX_INST_SAMPLE_LB };
			static const uint8 compares[] = { GPU7_TEX_INST_SAMPLE_C, GPU7_TEX_INST_SAMPLE_C, GPU7_TEX_INST_SAMPLE_C_L, GPU7_TEX_INST_SAMPLE_C_LZ };
			static const uint8 any[] = { GPU7_TEX_INST_SAMPLE, GPU7_TEX_INST_SAMPLE_C, GPU7_TEX_INST_SAMPLE_C_L, GPU7_TEX_INST_SAMPLE_C_LZ,
				GPU7_TEX_INST_FETCH4, GPU7_TEX_INST_LD, GPU7_TEX_INST_GET_TEXTURE_RESINFO, GPU7_TEX_INST_GET_COMP_TEX_LOD, GPU7_TEX_INST_SAMPLE_G,
				GPU7_TEX_INST_GET_GRADIENTS_H };
			const bool flat2D = kind == TEX_2D || kind == TEX_2D_UINT;
			const bool line = kind == TEX_1D || kind == TEX_1D_UINT;
			const bool cube = kind == TEX_CUBE;
			uint32 k = R(100);
			if (allowDeclined && Chance(15))
				t.opcode = any[R(10)];
			else if (shadowUnit[t.unit])
			{
				// an array's: no level, but in a vertex shader's zero
				t.opcode = k < 75 ? compares[R(4)] : k < 90 ? GPU7_TEX_INST_GET_TEXTURE_RESINFO :
					vertexMode ? GPU7_TEX_INST_SAMPLE_C : GPU7_TEX_INST_GET_COMP_TEX_LOD;
				if (kind == TEX_2D_ARRAY && (t.opcode == GPU7_TEX_INST_SAMPLE_C_L || (t.opcode == GPU7_TEX_INST_SAMPLE_C_LZ && !vertexMode)))
					t.opcode = GPU7_TEX_INST_SAMPLE_C;
			}
			// no compare at a level in a vertex shader: lavapipe (Mesa 25.2)
			// crashes on some, rslc's module as well as the direct one
			if (vertexMode && t.opcode == GPU7_TEX_INST_SAMPLE_C_L)
				t.opcode = GPU7_TEX_INST_SAMPLE_C;
			else if (k < 10 && kind != TEX_3D && kind != TEX_1D_UINT)
				t.opcode = GPU7_TEX_INST_FETCH4; // a 1D's: zeros
			else if (k < 18 && (flat2D || line) && !vertexMode)
				t.opcode = GPU7_TEX_INST_LD;
			else if (k < 24)
				t.opcode = GPU7_TEX_INST_GET_TEXTURE_RESINFO;
			else if (k < 30 && !vertexMode)
				t.opcode = GPU7_TEX_INST_GET_COMP_TEX_LOD;
			else if (k < 37 && (flat2D || line))
				t.opcode = GPU7_TEX_INST_SAMPLE_G;
			else if (k < 42 && !vertexMode)
				t.opcode = Chance(50) ? GPU7_TEX_INST_GET_GRADIENTS_H : GPU7_TEX_INST_GET_GRADIENTS_V;
			else
				t.opcode = samples[R(6)];
			static const uint8 varying[] = { 0, 1, 3 };
			t.srcGpr = Chance(60) ? varying[R(3)] : (uint8)R(kGPRs);
			const bool compare = t.opcode == GPU7_TEX_INST_SAMPLE_C || t.opcode == GPU7_TEX_INST_SAMPLE_C_L || t.opcode == GPU7_TEX_INST_SAMPLE_C_LZ;
			for (int c = 0; c < 4; c++)
			{
				// a compare reads its register's channels as they are, a 1D
				// sample its coordinate
				const bool lineSample = line && !compare && t.opcode != GPU7_TEX_INST_FETCH4 && t.opcode != GPU7_TEX_INST_LD &&
					t.opcode != GPU7_TEX_INST_GET_TEXTURE_RESINFO && t.opcode != GPU7_TEX_INST_GET_COMP_TEX_LOD &&
					t.opcode != GPU7_TEX_INST_GET_GRADIENTS_H && t.opcode != GPU7_TEX_INST_GET_GRADIENTS_V && !t.unnormalized;
				const bool cubeSample = cube && t.opcode != GPU7_TEX_INST_GET_TEXTURE_RESINFO && t.opcode != GPU7_TEX_INST_GET_COMP_TEX_LOD &&
					t.opcode != GPU7_TEX_INST_GET_GRADIENTS_H && t.opcode != GPU7_TEX_INST_GET_GRADIENTS_V;
				const bool asRead = (compare && (kind == TEX_2D ? c != (t.opcode == GPU7_TEX_INST_SAMPLE_C_L ? 3 : 2) :
					kind == TEX_1D ? c == 0 || c == 3 : c == 3)) || (lineSample && c == 0) || (cubeSample && c < 2);
				t.srcSel[c] = Chance(asRead && !allowDeclined ? 100 : 85) ? (uint8)R(4) : (uint8)(4 + R(2));
			}
			bool anyWritten = false;
			for (int c = 0; c < 4; c++)
			{
				// a compare's one result: x (others, now and then)
				t.dstSel[c] = Chance(70) ? (compare && !Chance(5) ? 0 : (uint8)R(4)) : Chance(10) ? (uint8)(4 + R(2)) : 7;
				anyWritten |= t.dstSel[c] < 4;
			}
			if (!anyWritten)
				t.dstSel[R(4)] = compare ? 0 : (uint8)R(4);
			t.dstGpr = Dest();
			// offsets of the samples and texel fetches (in texels, -8 to 7),
			// the gathers' now and then
			const bool sampleKind = t.opcode != GPU7_TEX_INST_FETCH4 &&
				t.opcode != GPU7_TEX_INST_GET_GRADIENTS_H && t.opcode != GPU7_TEX_INST_GET_GRADIENTS_V &&
				!(compare && (kind == TEX_2D_ARRAY || t.opcode == GPU7_TEX_INST_SAMPLE_C_L));
			if (Chance(sampleKind ? 30 : 3))
			{
				for (auto& o : t.offset)
					o = (sint8)((int)R(32) - 16);
			}
			t.unnormalized = t.opcode == GPU7_TEX_INST_SAMPLE && (((flat2D || line) && !vertexMode && !shadowUnit[t.unit] && Chance(10)) ||
				(allowDeclined && Chance(5)));
			if (t.opcode == GPU7_TEX_INST_GET_GRADIENTS_H || t.opcode == GPU7_TEX_INST_GET_GRADIENTS_V || (cube && !allowDeclined))
				t.offset[0] = t.offset[1] = t.offset[2] = 0;
			// no offset on a 3D texture outside a pixel shader: lavapipe
			// (Mesa 25.2) calls through a null pointer on some textureLodOffset
			// of one there, rslc's module as well as the direct one
			if (vertexMode && kind == TEX_3D)
				t.offset[0] = t.offset[1] = t.offset[2] = 0;
			if (cube && Chance(40))
			{
				// the cube array's layer first, now and then
				TexOp set{};
				set.opcode = GPU7_TEX_INST_SET_CUBEMAP_INDEX;
				set.unit = t.unit;
				set.srcGpr = Chance(60) ? varying[R(3)] : (uint8)R(kGPRs);
				for (int c = 0; c < 4; c++)
				{
					set.srcSel[c] = Chance(85) ? (uint8)R(4) : (uint8)(4 + R(2)); // constants too
					set.dstSel[c] = 7;
				}
				clause.push_back(set);
			}
			if (t.opcode == GPU7_TEX_INST_SAMPLE_G)
			{
				// gradH and gradV set first, in the clause: none read unset
				for (uint8 g : { (uint8)GPU7_TEX_INST_SET_GRADIENTS_H, (uint8)GPU7_TEX_INST_SET_GRADIENTS_V })
				{
					TexOp set{};
					set.opcode = g;
					set.unit = t.unit;
					set.srcGpr = Chance(60) ? varying[R(3)] : (uint8)R(kGPRs);
					for (int c = 0; c < 4; c++)
					{
						set.srcSel[c] = Chance(85) ? (uint8)R(4) : (uint8)(4 + R(2));
						set.dstSel[c] = 7;
					}
					clause.push_back(set);
				}
			}
			clause.push_back(t);
		}
		p.texClauses.push_back(std::move(clause));
		return (int)p.texClauses.size() - 1;
	}

	static bool IsPlainALU(const Program& p)
	{
		return !p.cf.empty() && p.cf.back().type == GPU7_CF_INST_ALU;
	}

	// statements at a stack depth: clauses, ifs with an else now and then,
	// loops, breaks out of them - ended by each kind of pop
	// loopBase: the depth of the innermost loop's body
	void Statements(Program& p, int depth, int loops, int loopBase, int& budget, bool allowDeclined)
	{
		noKills = vertexMode || loops > 0 || loopMade;
		inLoop = inLoop || loops > 0;
		uint32 n = 1 + R(3);
		for (uint32 i = 0; i < n && budget > 0; i++)
		{
			uint32 k = R(100);
			budget--;
			if (loops > 0 && (k < 4 || (depth > loopBase && k < 20)))
				p.cf.push_back({ GPU7_CF_INST_LOOP_BREAK, -1, 0 });
			else if (loops > 0 && depth == loopBase && depth > 0 && k < 30)
			{
				// the if around the loop turned over inside it: the loop's
				// own mask goes off, and back to what it was at its end
				p.cf.push_back({ GPU7_CF_INST_ELSE, -1, 0 });
			}
			else if (k < 50 || depth >= 3)
			{
				if (Chance(30))
					p.cf.push_back({ GPU7_CF_INST_TEX, TexClause(p, allowDeclined), 0 });
				else
					p.cf.push_back({ GPU7_CF_INST_ALU, Clause(p, 4, allowDeclined), 0 });
			}
			else if (k < 82 || loops >= 1 || loopMade)
			{
				p.cf.push_back({ GPU7_CF_INST_ALU_PUSH_BEFORE, Clause(p, 3, allowDeclined, true), 0 });
				Statements(p, depth + 1, loops, loopBase, budget, allowDeclined);
				if (Chance(45))
				{
					if (IsPlainALU(p) && Chance(50))
						p.cf.back().type = GPU7_CF_INST_ALU_ELSE_AFTER;
					else
						p.cf.push_back({ GPU7_CF_INST_ELSE, -1, 0 });
					Statements(p, depth + 1, loops, loopBase, budget, allowDeclined);
				}
				// the pop: after the last clause, or its own - or the inner
				// if's, taking this one's along
				if (IsPlainALU(p) && Chance(50))
					p.cf.back().type = GPU7_CF_INST_ALU_POP_AFTER;
				else if (!p.cf.empty() && p.cf.back().type == GPU7_CF_INST_ALU_POP_AFTER && Chance(40))
					p.cf.back().type = GPU7_CF_INST_ALU_POP2_AFTER;
				else if (!p.cf.empty() && p.cf.back().type == GPU7_CF_INST_POP && p.cf.back().popCount == 1 && Chance(40))
					p.cf.back().popCount = 2;
				else
					p.cf.push_back({ GPU7_CF_INST_POP, -1, 1 });
			}
			else
			{
				// counter = 0; loop { counter += 1.0; break unless N > counter; ... }
				const uint8 chan = (uint8)loops;
				AluGroup init{};
				AluOp mov{};
				mov.opcode = ALU_OP2_INST_MOV;
				mov.src[0] = 253;
				mov.destGpr = kCounterGPR;
				mov.destElem = chan;
				mov.writeMask = 1;
				init.ops.push_back(mov);
				p.clauses.push_back({ init });
				p.cf.push_back({ GPU7_CF_INST_ALU, (int)p.clauses.size() - 1, 0 });
				p.cf.push_back({ Chance(50) ? (uint32)GPU7_CF_INST_LOOP_START_DX10 : (uint32)GPU7_CF_INST_LOOP_START_NO_AL, -1, 0 });
				AluGroup inc{};
				AluOp add{};
				add.opcode = ALU_OP2_INST_ADD;
				add.src[0] = kCounterGPR;
				add.chan[0] = chan;
				add.src[1] = 253;
				add.destGpr = kCounterGPR;
				add.destElem = chan;
				add.writeMask = 1;
				inc.ops.push_back(add);
				inc.literals[0] = 0x3F800000u;
				AluGroup test{};
				AluOp pred{};
				pred.opcode = ALU_OP2_INST_PRED_SETGT;
				pred.src[0] = 253;
				pred.src[1] = kCounterGPR;
				pred.chan[1] = chan;
				pred.exec = 1;
				test.ops.push_back(pred);
				float limit = (float)(1 + R(4));
				std::memcpy(&test.literals[0], &limit, 4);
				noKills = true;
				inLoop = true;
				int c = Clause(p, 2, allowDeclined);
				auto& groups = p.clauses[c];
				groups.insert(groups.begin(), inc);
				groups.push_back(test);
				p.cf.push_back({ GPU7_CF_INST_ALU_BREAK, c, 0 });
				Statements(p, depth, loops + 1, depth, budget, allowDeclined);
				loopMade = true;
				noKills = true;
				p.cf.push_back({ GPU7_CF_INST_LOOP_END, -1, 0 });
			}
		}
	}

	// a run of plain ALU and TEX entries moved into a subroutine, a CALL in
	// their place; another CALL of it elsewhere now and then
	void MakeCall(Program& p)
	{
		if (p.cf.empty())
		{
			for (size_t i = 0; i < p.clauses.size(); i++)
				p.cf.push_back({ GPU7_CF_INST_ALU, (int)i, 0 });
		}
		std::vector<size_t> plain;
		for (size_t i = 0; i < p.cf.size(); i++)
		{
			if (p.cf[i].type == GPU7_CF_INST_ALU || p.cf[i].type == GPU7_CF_INST_TEX)
				plain.push_back(i);
		}
		if (plain.empty())
			return;
		const size_t at = plain[R((uint32)plain.size())];
		size_t end = at + 1;
		const size_t want = 1 + R(3);
		while (end < p.cf.size() && end - at < want && (p.cf[end].type == GPU7_CF_INST_ALU || p.cf[end].type == GPU7_CF_INST_TEX))
			end++;
		p.sub.assign(p.cf.begin() + at, p.cf.begin() + end);
		p.cf.erase(p.cf.begin() + at, p.cf.begin() + end);
		p.cf.insert(p.cf.begin() + at, CFEntry{ GPU7_CF_INST_CALL, -1, 0 });
		// the second outside any loop: the subroutine may set a loop's
		// counter, which a call in its body would reset each time round
		std::vector<size_t> outside;
		int loops = 0;
		for (size_t i = 0; i <= p.cf.size(); i++)
		{
			if (loops == 0)
				outside.push_back(i);
			if (i == p.cf.size())
				break;
			if (p.cf[i].type == GPU7_CF_INST_LOOP_START_DX10 || p.cf[i].type == GPU7_CF_INST_LOOP_START_NO_AL)
				loops++;
			else if (p.cf[i].type == GPU7_CF_INST_LOOP_END)
				loops--;
		}
		if (Chance(30))
			p.cf.insert(p.cf.begin() + outside[R((uint32)outside.size())], CFEntry{ GPU7_CF_INST_CALL, -1, 0 });
	}

	Program Make(bool allowDeclined)
	{
		Program p{};
		bufferMode = Chance(30);
		destGPRs = kGPRs;
		noKills = vertexMode; // a vertex shader has no discard
		loopMade = false;
		inLoop = false;
		favorDest = -1;
		p.exportGpr = R(kGPRs - 1);
		// the units' textures: mostly 2D floats, a cube map (declined) now
		// and then
		for (auto& kind : p.texKind)
		{
			uint32 k = R(100);
			kind = k < 32 ? TEX_2D : k < 45 ? TEX_2D_ARRAY : k < 57 ? TEX_3D : k < 70 ? TEX_2D_UINT : k < 81 ? TEX_1D : k < 88 ? TEX_1D_UINT : TEX_CUBE;
		}
		// a 2D or 2D array compared with, now and then
		for (int u = 0; u < 4; u++)
			shadowUnit[u] = (p.texKind[u] == TEX_2D || p.texKind[u] == TEX_2D_ARRAY || p.texKind[u] == TEX_1D) && Chance(30);
		if (Chance(55))
		{
			// control flow; the counters' register written by them only
			destGPRs = kCounterGPR;
			favorDest = (int)p.exportGpr;
			int budget = 2 + (int)R(9);
			Statements(p, 0, 0, 0, budget, allowDeclined);
		}
		else
		{
			// ALU clauses, with TEX clauses among them now and then
			const bool withTex = Chance(45);
			favorDest = withTex ? (int)p.exportGpr : -1;
			uint32 clauses = 1 + R(3);
			for (uint32 c = 0; c < clauses; c++)
			{
				if (withTex && Chance(50))
					p.cf.push_back({ GPU7_CF_INST_TEX, TexClause(p, allowDeclined), 0 });
				int a = Clause(p, 8, allowDeclined);
				if (withTex)
					p.cf.push_back({ GPU7_CF_INST_ALU, a, 0 });
			}
		}
		if (Chance(15))
			MakeCall(p);
		// a depth export now and then: a register's channel, or a constant
		if (!vertexMode && SysChance(20))
		{
			p.depthGpr = (sint8)(SysChance(50) ? p.exportGpr : SysR(kGPRs - 1));
			p.depthSel = (uint8)(SysChance(85) ? SysR(4) : 4 + SysR(2));
		}
		for (int i = 0; i < 4; i++)
		{
			uint32 k = R(10);
			p.exportSel[i] = (uint8)(k < 7 ? R(4) : (k == 7 ? 4 : (k == 8 ? 5 : 7)));
		}
		return p;
	}
};

// --- the two outputs ---------------------------------------------------------

static uint32 s_regs[LATTE_MAX_REGISTER];

struct Decompiled
{
	std::vector<uint32> spirv;   // the module the renderer gets
	LatteDecompilerOutputUniformOffsets offsets;
	std::vector<LatteDecompilerRemappedUniformEntry_t> remapped;
	uint8 uniformMode = 0;
	sint32 setIndex = 0, binding = -1;
	bool ok = false;
	const char* declined = nullptr;
	sint32 texBinding[LATTE_NUM_MAX_TEX_UNITS]; // -1: the unit is not used
	bool shadow[LATTE_NUM_MAX_TEX_UNITS]; // a shadow sampler: the depth texture
	sint8 attrLoc[LATTE_NUM_MAX_ATTRIBUTE_LOCATIONS]; // a vertex shader's
	sint32 tfBinding = -1; // the streamout storage buffer's, where written
	sint32 ubBinding[LATTE_NUM_MAX_UNIFORM_BUFFERS]; // uniform buffer mode's, -1 for none
	std::string glsl;
};

static bool CompileGlsl(const std::string& source, std::vector<uint32>& out, bool rte, rslc_stage stage = RSLC_STAGE_FRAGMENT)
{
	const std::string glsl = RslcGlsl(source, rte);
	char err[512] = {};
	rslc_opts opts{};
	opts.dst = RSLC_DST_SPIRV;
	opts.err = err;
	opts.err_cap = sizeof(err);
	rslc_result res{};
	if (!rslc_compile(glsl.data(), glsl.size(), stage, &opts, &res))
		return false;
	out.assign(res.spirv, res.spirv + res.spirv_len);
	rslc_result_free(&res, &opts);
	return true;
}

// the pixel shader drawn after a geometry shader (its inputs by semantic)
static bool s_usesGS = false;

// four inputs from the vertex shader, R0 to R3, at locations 0 to 3: by
// index, semantics 0 to 3; after a geometry shader, at the semantic's
// location, semantics 3 to 0 (so the two ways name different locations)
// fragCoord: input 1 the position (gl_FragCoord), as the SPI imports it;
// psControl1: SPI_PS_IN_CONTROL_1, its front face bits; paramGenGpr: the
// register param gen writes gl_PointCoord into (-1: none)
static void SetPSInputs(bool usesGS, bool fragCoord = false, uint32 psControl1 = 0, sint32 paramGenGpr = -1)
{
	g_testPSInputTable.paramGen = paramGenGpr >= 0 ? 1 : 0;
	g_testPSInputTable.paramGenGPR = (uint8)(paramGenGpr >= 0 ? paramGenGpr : 0);
	s_usesGS = usesGS;
	g_testPSInputTable.count = 4;
	for (int i = 0; i < 4; i++)
	{
		g_testPSInputTable.import[i].semanticId = (uint32)(usesGS ? 3 - i : i);
		g_testPSInputTable.import[i].isFlat = false;
		g_testPSInputTable.import[i].isNoPerspective = (usesGS ? 3 - i : i) == 3;
	}
	if (fragCoord && !usesGS)
		g_testPSInputTable.import[1].semanticId = LATTE_ANALYZER_IMPORT_INDEX_SPIPOSITION;
	s_regs[mmSPI_PS_IN_CONTROL_1] = psControl1;
}

static Decompiled Decompile(const std::vector<uint32>& program, bool direct, bool strictMul, bool rte)
{
	Decompiled d;
	LatteDecompilerOptions options;
	options.usesGeometryShader = s_usesGS;
	options.strictMul = strictMul;
	options.directSPIRV = direct;
	options.spirvInstrinsics.hasRoundingModeRTEFloat32 = rte;
	LatteDecompilerOutput_t out{};
	LatteDecompiler_DecompilePixelShader(0x1234, s_regs, (uint8*)program.data(), (uint32)program.size() * 4, options, &out);
	LatteDecompilerShader* shader = out.shader;
	d.offsets = out.uniformOffsetsVK;
	d.remapped = shader->list_remappedUniformEntries;
	d.uniformMode = shader->uniformMode;
	d.setIndex = out.resourceMappingVK.setIndex;
	d.binding = out.resourceMappingVK.uniformVarsBufferBindingPoint;
	for (int b = 0; b < LATTE_NUM_MAX_UNIFORM_BUFFERS; b++)
		d.ubBinding[b] = out.resourceMappingVK.uniformBuffersBindingPoint[b];
	for (int i = 0; i < LATTE_NUM_MAX_TEX_UNITS; i++)
	{
		d.texBinding[i] = out.textureUnitMask[i] ? out.resourceMappingVK.textureUnitToBindingPoint[i] : -1;
		d.shadow[i] = shader->textureUsesDepthCompare[i];
	}
	d.declined = LatteDecompiler_SPIRVDeclineReason();
	if (shader->hasError)
	{
		delete shader;
		return d;
	}
	if (!shader->spirv.empty())
	{
		d.spirv = shader->spirv;
		d.ok = true;
	}
	else
	{
		d.glsl = shader->shaderSource;
		d.ok = CompileGlsl(shader->shaderSource, d.spirv, rte);
	}
	delete shader;
	return d;
}

// --- drawing -----------------------------------------------------------------

static const uint32 W = 4, H = 4;

// the uniform buffers' contents, the same for every program: buffer b's
// vec4s (4096, a whole bound range), floats of either sign mostly and raw
// bits now and then
static uint32 s_cbuf[LATTE_NUM_MAX_UNIFORM_BUFFERS][4096 * 4];

static void FillUniformBuffers()
{
	uint32 h = 0x9E3779B9u;
	for (auto& buffer : s_cbuf)
	{
		for (auto& word : buffer)
		{
			h ^= h << 13;
			h ^= h >> 17;
			h ^= h << 5;
			if (h % 4 != 0)
			{
				const float f = (float)((int)(h >> 8) % 2000 - 1000) / 64.0f;
				std::memcpy(&word, &f, 4);
			}
			else
				word = h;
		}
	}
}

// a remapped uniform's vec4: the register file's, or a uniform buffer's
static const uint32* RemappedSource(const LatteDecompilerRemappedUniformEntry_t& e, const uint32* cfile)
{
	if (e.isRegister)
		return cfile + e.index * 4;
	return s_cbuf[e.kcacheBankId % LATTE_NUM_MAX_UNIFORM_BUFFERS] + (e.index % 4096) * 4;
}

struct Context
{
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice gpu = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32 family = 0;
	VkCommandPool pool = VK_NULL_HANDLE;
	VkRenderPass renderPass = VK_NULL_HANDLE;
	VkImage color = VK_NULL_HANDLE;
	VkDeviceMemory colorMem = VK_NULL_HANDLE, readMem = VK_NULL_HANDLE, uboMem = VK_NULL_HANDLE;
	static const uint32 kStreamoutBytes = 32768;
	VkImageView colorView = VK_NULL_HANDLE;
	VkFramebuffer framebuffer = VK_NULL_HANDLE;
	VkBuffer readback = VK_NULL_HANDLE, ubo = VK_NULL_HANDLE;
	// a pixel shader exporting depth: the color and a D32 target, the depth
	// read back after the color
	VkRenderPass renderPassDepth = VK_NULL_HANDLE;
	VkImage depthTarget = VK_NULL_HANDLE;
	VkDeviceMemory depthTargetMem = VK_NULL_HANDLE, depthReadMem = VK_NULL_HANDLE;
	VkImageView depthTargetView = VK_NULL_HANDLE;
	VkFramebuffer framebufferDepth = VK_NULL_HANDLE;
	VkBuffer depthReadback = VK_NULL_HANDLE;
	uint8* uboData = nullptr;
	// the streamout storage buffer (kStreamoutBytes), written by vertex
	// and geometry shaders where the device lets them store
	VkBuffer sbuf = VK_NULL_HANDLE;
	VkDeviceMemory sbufMem = VK_NULL_HANDLE;
	// the uniform buffers (s_cbuf), 64 KiB each, bound where a shader reads
	// them whole
	VkBuffer cbuf = VK_NULL_HANDLE;
	VkDeviceMemory cbufMem = VK_NULL_HANDLE;
	uint8* sbufData = nullptr;
	bool vertexStores = false;
	VkShaderModule vs = VK_NULL_HANDLE;
	VkShaderModule vsPoint = VK_NULL_HANDLE; // one point over the target
	bool rte = false;
	bool largePoints = false; // points of sizes other than 1
	bool geometryShader = false;
	bool geometryPointSize = false;
	// a texture of each kind, with mips, its view and sampler
	struct Texture
	{
		VkImage image = VK_NULL_HANDLE;
		VkDeviceMemory mem = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkSampler sampler = VK_NULL_HANDLE;
	} tex[TEX_KINDS], depth[TEX_KINDS]; // depth: a 2D's and a 2D array's, compared
	// a vertex shader drawn: a fragment shader passing on the flat
	// parameter at each location, one the depth; the vertex data
	VkShaderModule fsParam[8]{}, fsDepth = VK_NULL_HANDLE;
	VkBuffer vbuf = VK_NULL_HANDLE;
	VkDeviceMemory vbufMem = VK_NULL_HANDLE;
	uint8* vbufData = nullptr;
};

static const char* kVS = R"(#version 450
layout(push_constant) uniform PC { vec4 v[4]; } pc;
layout(location = 0) out vec4 p0;
layout(location = 1) out vec4 p1;
layout(location = 2) out vec4 p2;
layout(location = 3) out vec4 p3;
void main()
{
	vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
	// a depth and a w that vary (gl_FragCoord's z and w)
	float w = 1.0 + pos.y * 0.75;
	gl_Position = vec4((pos * 2.0 - 1.0) * w, pos.x * 0.375 * w, w);
	// inputs that vary over the target, and some that stay put
	p0 = pc.v[0] * pos.x + pc.v[1];
	p1 = pc.v[1] * pos.y - pc.v[2];
	p2 = pc.v[2];
	p3 = pc.v[3] * (pos.x - pos.y);
}
)";

// one point of the target's size over it (gl_PointCoord over its pixels),
// the inputs the push constants
static const char* kVSPoint = R"(#version 450
layout(push_constant) uniform PC { vec4 v[4]; } pc;
layout(location = 0) out vec4 p0;
layout(location = 1) out vec4 p1;
layout(location = 2) out vec4 p2;
layout(location = 3) out vec4 p3;
void main()
{
	gl_Position = vec4(0.0, 0.0, 0.25, 1.0);
	gl_PointSize = 4.0;
	p0 = pc.v[0];
	p1 = pc.v[1];
	p2 = pc.v[2];
	p3 = pc.v[3];
}
)";

static uint32 MemoryType(const Context& c, uint32 bits, VkMemoryPropertyFlags want)
{
	VkPhysicalDeviceMemoryProperties mp;
	vkGetPhysicalDeviceMemoryProperties(c.gpu, &mp);
	for (uint32 i = 0; i < mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
			return i;
	std::printf("FAIL: no memory type\n");
	std::exit(1);
}

static VkDeviceMemory Bind(const Context& c, VkMemoryRequirements req, VkMemoryPropertyFlags want)
{
	VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	ai.allocationSize = req.size;
	ai.memoryTypeIndex = MemoryType(c, req.memoryTypeBits, want);
	VkDeviceMemory m;
	VKCHECK(vkAllocateMemory(c.device, &ai, nullptr, &m));
	return m;
}

static VkShaderModule Module(const Context& c, const std::vector<uint32>& words)
{
	VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	si.codeSize = words.size() * 4;
	si.pCode = words.data();
	VkShaderModule m;
	VKCHECK(vkCreateShaderModule(c.device, &si, nullptr, &m));
	return m;
}

// 8x8 with 4 levels: a 2D, a 2D array of 4 layers, an 8 deep 3D, an 8 wide
// 1D, an array of two cubes (floats, filtered linearly, repeating), a 2D
// and a 1D of uints
// (nearest); random texels from a fixed seed, uploaded through a staging
// buffer. depth: a 2D's, a 2D array's or a 1D's depths, 0 to 1, for a
// sampler comparing with LESS
static void CreateTexture(Context& c, TexKind kind, bool depth = false)
{
	Context::Texture& t = depth ? c.depth[kind] : c.tex[kind];
	const bool isUint = kind == TEX_2D_UINT || kind == TEX_1D_UINT;
	const bool line = kind == TEX_1D || kind == TEX_1D_UINT;
	const uint32 layers = kind == TEX_2D_ARRAY ? 4 : kind == TEX_CUBE ? 12 : 1; // a cube array: two cubes
	const uint32 levels = 4;
	const uint32 texelBytes = depth ? 4 : 16;
	const VkImageAspectFlags aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ii.imageType = kind == TEX_3D ? VK_IMAGE_TYPE_3D : line ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
	ii.format = depth ? VK_FORMAT_D32_SFLOAT : isUint ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT;
	ii.extent = {8, line ? 1u : 8u, kind == TEX_3D ? 8u : 1u};
	ii.mipLevels = levels;
	ii.arrayLayers = layers;
	ii.samples = VK_SAMPLE_COUNT_1_BIT;
	ii.tiling = VK_IMAGE_TILING_OPTIMAL;
	ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	if (kind == TEX_CUBE)
		ii.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	VKCHECK(vkCreateImage(c.device, &ii, nullptr, &t.image));
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(c.device, t.image, &req);
	t.mem = Bind(c, req, 0);
	VKCHECK(vkBindImageMemory(c.device, t.image, t.mem, 0));
	// every level's texels, one after the other
	std::vector<VkBufferImageCopy> regions;
	size_t bytes = 0;
	for (uint32 l = 0; l < levels; l++)
	{
		uint32 e = 8u >> l;
		VkBufferImageCopy r{};
		r.bufferOffset = bytes;
		r.imageSubresource = {aspect, l, 0, layers};
		r.imageExtent = {e, line ? 1u : e, kind == TEX_3D ? e : 1u};
		regions.push_back(r);
		bytes += (size_t)e * (line ? 1u : e) * (kind == TEX_3D ? e : 1u) * layers * texelBytes;
	}
	VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bi.size = bytes;
	bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkBuffer staging;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &staging));
	VkMemoryRequirements breq;
	vkGetBufferMemoryRequirements(c.device, staging, &breq);
	VkDeviceMemory stagingMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, staging, stagingMem, 0));
	void* map;
	VKCHECK(vkMapMemory(c.device, stagingMem, 0, bytes, 0, &map));
	std::mt19937 rng(1000 + kind + (depth ? 100 : 0));
	uint32* texels = (uint32*)map;
	for (size_t i = 0; i < bytes / 4; i++)
	{
		if (depth)
		{
			float f = (float)(rng() % 1001) / 1000.0f;
			std::memcpy(texels + i, &f, 4);
		}
		else if (isUint)
			texels[i] = rng();
		else
		{
			float f = (float)((int)(rng() % 4001) - 2000) / 500.0f;
			std::memcpy(texels + i, &f, 4);
		}
	}
	vkUnmapMemory(c.device, stagingMem);
	VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cai.commandPool = c.pool;
	cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cai.commandBufferCount = 1;
	VkCommandBuffer cmd;
	VKCHECK(vkAllocateCommandBuffers(c.device, &cai, &cmd));
	VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VKCHECK(vkBeginCommandBuffer(cmd, &cbi));
	VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image = t.image;
	b.subresourceRange = {aspect, 0, levels, 0, layers};
	b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
	vkCmdCopyBufferToImage(cmd, staging, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32)regions.size(), regions.data());
	b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
	VKCHECK(vkEndCommandBuffer(cmd));
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	VKCHECK(vkCreateFence(c.device, &fi, nullptr, &fence));
	VKCHECK(vkQueueSubmit(c.queue, 1, &si, fence));
	VKCHECK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
	vkDestroyFence(c.device, fence, nullptr);
	vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
	vkDestroyBuffer(c.device, staging, nullptr);
	vkFreeMemory(c.device, stagingMem, nullptr);
	VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	vi.image = t.image;
	vi.viewType = kind == TEX_3D ? VK_IMAGE_VIEW_TYPE_3D : kind == TEX_2D_ARRAY ? VK_IMAGE_VIEW_TYPE_2D_ARRAY :
		kind == TEX_CUBE ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY :
		line ? VK_IMAGE_VIEW_TYPE_1D : VK_IMAGE_VIEW_TYPE_2D;
	vi.format = ii.format;
	vi.subresourceRange = {aspect, 0, levels, 0, layers};
	VKCHECK(vkCreateImageView(c.device, &vi, nullptr, &t.view));
	VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
	sci.magFilter = sci.minFilter = isUint ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
	sci.mipmapMode = isUint ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sci.maxLod = (float)levels;
	sci.compareEnable = depth ? VK_TRUE : VK_FALSE;
	sci.compareOp = VK_COMPARE_OP_LESS;
	VKCHECK(vkCreateSampler(c.device, &sci, nullptr, &t.sampler));
}

static bool CreateContext(Context& c)
{
	if (!InitializeGlobalVulkan())
		return false;
	VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_1;
	VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	if (vkCreateInstance(&ici, nullptr, &c.instance) != VK_SUCCESS || !InitializeInstanceVulkan(c.instance))
		return false;
	uint32 count = 0;
	vkEnumeratePhysicalDevices(c.instance, &count, nullptr);
	if (count == 0)
		return false;
	std::vector<VkPhysicalDevice> gpus(count);
	vkEnumeratePhysicalDevices(c.instance, &count, gpus.data());
	c.gpu = gpus[0];
	// the round-to-even mode, where the device takes it (as the renderer asks)
	uint32 n = 0;
	vkEnumerateDeviceExtensionProperties(c.gpu, nullptr, &n, nullptr);
	std::vector<VkExtensionProperties> exts(n);
	vkEnumerateDeviceExtensionProperties(c.gpu, nullptr, &n, exts.data());
	bool hasFloatControls = false, hasImageRobustness = false;
	for (auto& e : exts)
	{
		hasFloatControls |= !strcmp(e.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
		hasImageRobustness |= !strcmp(e.extensionName, VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME);
	}
	if (hasFloatControls)
	{
		VkPhysicalDeviceFloatControlsPropertiesKHR fc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES_KHR};
		VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
		p2.pNext = &fc;
		vkGetPhysicalDeviceProperties2(c.gpu, &p2);
		c.rte = fc.shaderRoundingModeRTEFloat32 != VK_FALSE;
	}
	uint32 qn = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(c.gpu, &qn, nullptr);
	std::vector<VkQueueFamilyProperties> qf(qn);
	vkGetPhysicalDeviceQueueFamilyProperties(c.gpu, &qn, qf.data());
	for (uint32 i = 0; i < qn; i++)
		if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			c.family = i;
			break;
		}
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qi.queueFamilyIndex = c.family;
	qi.queueCount = 1;
	qi.pQueuePriorities = &prio;
	// texel fetches out of the texture or its levels read zeros: robust
	// image access, where there is
	const char* devExts[2];
	uint32 nExts = 0;
	if (c.rte)
		devExts[nExts++] = VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME;
	VkPhysicalDeviceImageRobustnessFeaturesEXT robust{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT};
	VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	if (hasImageRobustness)
	{
		devExts[nExts++] = VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME;
		robust.robustImageAccess = VK_TRUE;
		dci.pNext = &robust;
	}
	// cube map arrays, the decompiler's cube maps
	VkPhysicalDeviceFeatures features{};
	features.imageCubeArray = VK_TRUE;
	// points of the size a vertex shader gives
	VkPhysicalDeviceFeatures2 has{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	vkGetPhysicalDeviceFeatures2(c.gpu, &has);
	c.largePoints = has.features.largePoints != VK_FALSE;
	features.largePoints = has.features.largePoints;
	// the test's geometry shader, after a vertex shader writing the ring
	c.geometryShader = has.features.geometryShader != VK_FALSE;
	features.geometryShader = has.features.geometryShader;
	// and its points' size, where it emits points
	c.geometryPointSize = has.features.shaderTessellationAndGeometryPointSize != VK_FALSE;
	features.shaderTessellationAndGeometryPointSize = has.features.shaderTessellationAndGeometryPointSize;
	// streamout: the vertex and geometry shaders' stores to the buffer
	c.vertexStores = has.features.vertexPipelineStoresAndAtomics != VK_FALSE;
	features.vertexPipelineStoresAndAtomics = has.features.vertexPipelineStoresAndAtomics;
	// uniforms read past the array (an AR index out of it) read zeros
	features.robustBufferAccess = VK_TRUE;
	dci.pEnabledFeatures = &features;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qi;
	dci.enabledExtensionCount = nExts;
	dci.ppEnabledExtensionNames = devExts;
	if (vkCreateDevice(c.gpu, &dci, nullptr, &c.device) != VK_SUCCESS)
		return false;
	InitializeDeviceVulkan(c.device);
	vkGetDeviceQueue(c.device, c.family, 0, &c.queue);

	VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pci.queueFamilyIndex = c.family;
	VKCHECK(vkCreateCommandPool(c.device, &pci, nullptr, &c.pool));

	VkAttachmentDescription att{};
	att.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	att.samples = VK_SAMPLE_COUNT_1_BIT;
	att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkSubpassDescription sub{};
	sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	sub.colorAttachmentCount = 1;
	sub.pColorAttachments = &cref;
	VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
	rpi.attachmentCount = 1;
	rpi.pAttachments = &att;
	rpi.subpassCount = 1;
	rpi.pSubpasses = &sub;
	VKCHECK(vkCreateRenderPass(c.device, &rpi, nullptr, &c.renderPass));

	VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ii.imageType = VK_IMAGE_TYPE_2D;
	ii.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	ii.extent = {W, H, 1};
	ii.mipLevels = 1;
	ii.arrayLayers = 1;
	ii.samples = VK_SAMPLE_COUNT_1_BIT;
	ii.tiling = VK_IMAGE_TILING_OPTIMAL;
	ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	VKCHECK(vkCreateImage(c.device, &ii, nullptr, &c.color));
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(c.device, c.color, &req);
	c.colorMem = Bind(c, req, 0);
	VKCHECK(vkBindImageMemory(c.device, c.color, c.colorMem, 0));
	VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	vi.image = c.color;
	vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vi.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	VKCHECK(vkCreateImageView(c.device, &vi, nullptr, &c.colorView));
	VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
	fbi.renderPass = c.renderPass;
	fbi.attachmentCount = 1;
	fbi.pAttachments = &c.colorView;
	fbi.width = W;
	fbi.height = H;
	fbi.layers = 1;
	VKCHECK(vkCreateFramebuffer(c.device, &fbi, nullptr, &c.framebuffer));
	{
		VkAttachmentDescription atts[2] = { att, att };
		atts[1].format = VK_FORMAT_D32_SFLOAT;
		VkAttachmentReference dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
		VkSubpassDescription subD = sub;
		subD.pDepthStencilAttachment = &dref;
		VkRenderPassCreateInfo rpd = rpi;
		rpd.attachmentCount = 2;
		rpd.pAttachments = atts;
		rpd.pSubpasses = &subD;
		VKCHECK(vkCreateRenderPass(c.device, &rpd, nullptr, &c.renderPassDepth));
		VkImageCreateInfo di = ii;
		di.format = VK_FORMAT_D32_SFLOAT;
		di.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		VKCHECK(vkCreateImage(c.device, &di, nullptr, &c.depthTarget));
		VkMemoryRequirements dreq;
		vkGetImageMemoryRequirements(c.device, c.depthTarget, &dreq);
		c.depthTargetMem = Bind(c, dreq, 0);
		VKCHECK(vkBindImageMemory(c.device, c.depthTarget, c.depthTargetMem, 0));
		VkImageViewCreateInfo dvi = vi;
		dvi.image = c.depthTarget;
		dvi.format = VK_FORMAT_D32_SFLOAT;
		dvi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
		VKCHECK(vkCreateImageView(c.device, &dvi, nullptr, &c.depthTargetView));
		VkImageView views[2] = { c.colorView, c.depthTargetView };
		VkFramebufferCreateInfo fbd = fbi;
		fbd.renderPass = c.renderPassDepth;
		fbd.attachmentCount = 2;
		fbd.pAttachments = views;
		VKCHECK(vkCreateFramebuffer(c.device, &fbd, nullptr, &c.framebufferDepth));
	}

	VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bi.size = W * H * 16;
	bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &c.readback));
	VkMemoryRequirements breq;
	vkGetBufferMemoryRequirements(c.device, c.readback, &breq);
	c.readMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, c.readback, c.readMem, 0));
	bi.size = W * H * 4;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &c.depthReadback));
	vkGetBufferMemoryRequirements(c.device, c.depthReadback, &breq);
	c.depthReadMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, c.depthReadback, c.depthReadMem, 0));
	bi.size = 8192;
	bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &c.ubo));
	vkGetBufferMemoryRequirements(c.device, c.ubo, &breq);
	c.uboMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, c.ubo, c.uboMem, 0));
	void* map;
	VKCHECK(vkMapMemory(c.device, c.uboMem, 0, 8192, 0, &map));
	c.uboData = (uint8*)map;
	bi.size = sizeof(s_cbuf);
	bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &c.cbuf));
	vkGetBufferMemoryRequirements(c.device, c.cbuf, &breq);
	c.cbufMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, c.cbuf, c.cbufMem, 0));
	VKCHECK(vkMapMemory(c.device, c.cbufMem, 0, sizeof(s_cbuf), 0, &map));
	std::memcpy(map, s_cbuf, sizeof(s_cbuf));
	vkUnmapMemory(c.device, c.cbufMem);
	bi.size = Context::kStreamoutBytes;
	bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &c.sbuf));
	vkGetBufferMemoryRequirements(c.device, c.sbuf, &breq);
	c.sbufMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, c.sbuf, c.sbufMem, 0));
	VKCHECK(vkMapMemory(c.device, c.sbufMem, 0, Context::kStreamoutBytes, 0, &map));
	c.sbufData = (uint8*)map;

	char err[512] = {};
	rslc_opts opts{};
	opts.dst = RSLC_DST_SPIRV;
	opts.err = err;
	opts.err_cap = sizeof(err);
	rslc_result res{};
	if (!rslc_compile(kVS, strlen(kVS), RSLC_STAGE_VERTEX, &opts, &res))
	{
		std::printf("FAIL: rslc: %s\n", err);
		std::exit(1);
	}
	c.vs = Module(c, std::vector<uint32>(res.spirv, res.spirv + res.spirv_len));
	rslc_result_free(&res, &opts);
	if (!rslc_compile(kVSPoint, strlen(kVSPoint), RSLC_STAGE_VERTEX, &opts, &res))
	{
		std::printf("FAIL: rslc: %s\n", err);
		std::exit(1);
	}
	c.vsPoint = Module(c, std::vector<uint32>(res.spirv, res.spirv + res.spirv_len));
	rslc_result_free(&res, &opts);
	for (int k = TEX_2D; k <= TEX_CUBE; k++)
		CreateTexture(c, (TexKind)k);
	CreateTexture(c, TEX_2D, true);
	CreateTexture(c, TEX_2D_ARRAY, true);
	CreateTexture(c, TEX_1D, true);
	// the vertex tests' fragment shaders and vertex buffer
	auto fragment = [&](const std::string& src) {
		rslc_result r{};
		if (!rslc_compile(src.data(), src.size(), RSLC_STAGE_FRAGMENT, &opts, &r))
		{
			std::printf("FAIL: rslc: %s\n", err);
			std::exit(1);
		}
		VkShaderModule m = Module(c, std::vector<uint32>(r.spirv, r.spirv + r.spirv_len));
		rslc_result_free(&r, &opts);
		return m;
	};
	for (int l = 0; l < 8; l++)
		c.fsParam[l] = fragment(fmt::format("#version 450\nlayout(location = {}) flat in vec4 p;\nlayout(location = 0) out vec4 o;\nvoid main() {{ o = p; }}\n", l));
	c.fsDepth = fragment("#version 450\nlayout(location = 0) out vec4 o;\nvoid main() { o = vec4(gl_FragCoord.z, gl_FragCoord.w, gl_FragCoord.x, gl_FragCoord.y); }\n");
	bi.size = 1 << 16;
	bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &c.vbuf));
	vkGetBufferMemoryRequirements(c.device, c.vbuf, &breq);
	c.vbufMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, c.vbuf, c.vbufMem, 0));
	VKCHECK(vkMapMemory(c.device, c.vbufMem, 0, 1 << 16, 0, &map));
	c.vbufData = (uint8*)map;
	return true;
}

static void DestroyContext(Context& c)
{
	if (!c.device)
		return;
	vkDeviceWaitIdle(c.device);
	vkDestroyShaderModule(c.device, c.vs, nullptr);
	vkDestroyShaderModule(c.device, c.vsPoint, nullptr);
	for (auto m : c.fsParam)
		vkDestroyShaderModule(c.device, m, nullptr);
	vkDestroyShaderModule(c.device, c.fsDepth, nullptr);
	vkUnmapMemory(c.device, c.vbufMem);
	vkDestroyBuffer(c.device, c.vbuf, nullptr);
	vkFreeMemory(c.device, c.vbufMem, nullptr);
	for (auto& t : c.tex)
	{
		if (!t.image)
			continue;
		if (c.depth[&t - c.tex].image)
		{
			Context::Texture& z = c.depth[&t - c.tex];
			vkDestroySampler(c.device, z.sampler, nullptr);
			vkDestroyImageView(c.device, z.view, nullptr);
			vkDestroyImage(c.device, z.image, nullptr);
			vkFreeMemory(c.device, z.mem, nullptr);
		}
		vkDestroySampler(c.device, t.sampler, nullptr);
		vkDestroyImageView(c.device, t.view, nullptr);
		vkDestroyImage(c.device, t.image, nullptr);
		vkFreeMemory(c.device, t.mem, nullptr);
	}
	vkUnmapMemory(c.device, c.uboMem);
	vkDestroyBuffer(c.device, c.ubo, nullptr);
	vkUnmapMemory(c.device, c.sbufMem);
	vkDestroyBuffer(c.device, c.sbuf, nullptr);
	vkFreeMemory(c.device, c.sbufMem, nullptr);
	vkDestroyBuffer(c.device, c.cbuf, nullptr);
	vkFreeMemory(c.device, c.cbufMem, nullptr);
	vkDestroyBuffer(c.device, c.readback, nullptr);
	vkFreeMemory(c.device, c.uboMem, nullptr);
	vkFreeMemory(c.device, c.readMem, nullptr);
	vkDestroyBuffer(c.device, c.depthReadback, nullptr);
	vkFreeMemory(c.device, c.depthReadMem, nullptr);
	vkDestroyFramebuffer(c.device, c.framebufferDepth, nullptr);
	vkDestroyImageView(c.device, c.depthTargetView, nullptr);
	vkDestroyImage(c.device, c.depthTarget, nullptr);
	vkFreeMemory(c.device, c.depthTargetMem, nullptr);
	vkDestroyRenderPass(c.device, c.renderPassDepth, nullptr);
	vkDestroyFramebuffer(c.device, c.framebuffer, nullptr);
	vkDestroyImageView(c.device, c.colorView, nullptr);
	vkDestroyImage(c.device, c.color, nullptr);
	vkFreeMemory(c.device, c.colorMem, nullptr);
	vkDestroyRenderPass(c.device, c.renderPass, nullptr);
	vkDestroyCommandPool(c.device, c.pool, nullptr);
	vkDestroyDevice(c.device, nullptr);
	vkDestroyInstance(c.instance, nullptr);
}

// the shader drawn over the target: what each pixel holds, as bits
// uniform buffer mode's buffers: each a binding of the stage's
static void UniformBufferBindings(const Decompiled& d, VkShaderStageFlags stage, VkDescriptorSetLayoutBinding* b, uint32& nb)
{
	for (int i = 0; i < LATTE_NUM_MAX_UNIFORM_BUFFERS; i++)
	{
		if (d.ubBinding[i] < 0)
			continue;
		b[nb].binding = (uint32)d.ubBinding[i];
		b[nb].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		b[nb].descriptorCount = 1;
		b[nb].stageFlags = stage;
		nb++;
	}
}

// and their writes: buffer i at i * 64 KiB of the test's, 64 KiB long
static void WriteUniformBuffers(const Context& c, const Decompiled& d, VkDescriptorSet ds)
{
	VkDescriptorBufferInfo infos[LATTE_NUM_MAX_UNIFORM_BUFFERS];
	VkWriteDescriptorSet writes[LATTE_NUM_MAX_UNIFORM_BUFFERS]{};
	uint32 n = 0;
	for (int i = 0; i < LATTE_NUM_MAX_UNIFORM_BUFFERS; i++)
	{
		if (d.ubBinding[i] < 0)
			continue;
		infos[n] = { c.cbuf, (VkDeviceSize)i * sizeof(s_cbuf[0]), sizeof(s_cbuf[0]) };
		writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[n].dstSet = ds;
		writes[n].dstBinding = (uint32)d.ubBinding[i];
		writes[n].descriptorCount = 1;
		writes[n].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writes[n].pBufferInfo = &infos[n];
		n++;
	}
	if (n)
		vkUpdateDescriptorSets(c.device, n, writes, 0, nullptr);
}

// cw: the triangle's front face clockwise (so drawn as a back face);
// depth: into the depth target too, its values after the pixels'
static std::vector<uint32> Draw(const Context& c, const Decompiled& d, const uint8 texKind[4], const float inputs[16], const uint32* uniforms, size_t uniformBytes,
	bool cw = false, bool depth = false, bool points = false)
{
	// the uniform block and the texture units at the shader's set and
	// bindings (set 0 empty, as the renderer's vertex set)
	VkDescriptorSetLayoutBinding b[2 + LATTE_NUM_MAX_TEX_UNITS + LATTE_NUM_MAX_UNIFORM_BUFFERS]{};
	VkWriteDescriptorSet texWrites[LATTE_NUM_MAX_TEX_UNITS]{};
	VkDescriptorImageInfo texInfos[LATTE_NUM_MAX_TEX_UNITS]{};
	uint32 nb = 0, nTex = 0;
	if (d.binding >= 0)
	{
		b[nb].binding = (uint32)d.binding;
		b[nb].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		b[nb].descriptorCount = 1;
		b[nb].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		nb++;
	}
	for (int i = 0; i < LATTE_NUM_MAX_TEX_UNITS; i++)
	{
		if (d.texBinding[i] < 0)
			continue;
		if (i >= 4 || texKind[i] == TEX_NONE || texKind[i] >= TEX_KINDS || (d.shadow[i] && !c.depth[texKind[i]].image))
		{
			std::printf("FAIL: texture unit %d drawn with no texture\n", i);
			s_failures++;
			continue;
		}
		const Context::Texture& t = d.shadow[i] ? c.depth[texKind[i]] : c.tex[texKind[i]];
		b[nb].binding = (uint32)d.texBinding[i];
		b[nb].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		b[nb].descriptorCount = 1;
		b[nb].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		nb++;
		texInfos[nTex] = {t.sampler, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		texWrites[nTex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		texWrites[nTex].dstBinding = (uint32)d.texBinding[i];
		texWrites[nTex].descriptorCount = 1;
		texWrites[nTex].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		texWrites[nTex].pImageInfo = &texInfos[nTex];
		nTex++;
	}
	VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	VkDescriptorSetLayout empty, set;
	VKCHECK(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &empty));
	UniformBufferBindings(d, VK_SHADER_STAGE_FRAGMENT_BIT, b, nb);
	dli.bindingCount = nb;
	dli.pBindings = b;
	VKCHECK(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &set));
	VkDescriptorSetLayout sets[2] = {empty, set};
	VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 64};
	VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pli.setLayoutCount = 2;
	pli.pSetLayouts = sets;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges = &pcr;
	VkPipelineLayout layout;
	VKCHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &layout));
	VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + LATTE_NUM_MAX_UNIFORM_BUFFERS}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, LATTE_NUM_MAX_TEX_UNITS}};
	VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	dpi.maxSets = 1;
	dpi.poolSizeCount = 2;
	dpi.pPoolSizes = ps;
	VkDescriptorPool dpool;
	VKCHECK(vkCreateDescriptorPool(c.device, &dpi, nullptr, &dpool));
	VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	dai.descriptorPool = dpool;
	dai.descriptorSetCount = 1;
	dai.pSetLayouts = &set;
	VkDescriptorSet ds;
	VKCHECK(vkAllocateDescriptorSets(c.device, &dai, &ds));
	WriteUniformBuffers(c, d, ds);
	std::memset(c.uboData, 0, 8192);
	std::memcpy(c.uboData, uniforms, uniformBytes);
	if (d.binding >= 0)
	{
		VkDescriptorBufferInfo dbi{c.ubo, 0, 8192};
		VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		wds.dstSet = ds;
		wds.dstBinding = (uint32)d.binding;
		wds.descriptorCount = 1;
		wds.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		wds.pBufferInfo = &dbi;
		vkUpdateDescriptorSets(c.device, 1, &wds, 0, nullptr);
	}
	for (uint32 i = 0; i < nTex; i++)
		texWrites[i].dstSet = ds;
	if (nTex)
		vkUpdateDescriptorSets(c.device, nTex, texWrites, 0, nullptr);

	VkShaderModule fs = Module(c, d.spirv);
	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = points ? c.vsPoint : c.vs;
	stages[0].pName = "main";
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fs;
	stages[1].pName = "main";
	VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	ia.topology = points ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkViewport vp{0, 0, (float)W, (float)H, 0, 1};
	VkRect2D sc{{0, 0}, {W, H}};
	VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	vps.viewportCount = 1;
	vps.pViewports = &vp;
	vps.scissorCount = 1;
	vps.pScissors = &sc;
	VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rs.cullMode = VK_CULL_MODE_NONE;
	rs.frontFace = cw ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rs.lineWidth = 1.0f;
	VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineColorBlendAttachmentState cba{};
	cba.colorWriteMask = 0xF;
	VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	cb.attachmentCount = 1;
	cb.pAttachments = &cba;
	VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
	dss.depthTestEnable = VK_TRUE;
	dss.depthWriteEnable = VK_TRUE;
	dss.depthCompareOp = VK_COMPARE_OP_ALWAYS;
	VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	gpi.stageCount = 2;
	gpi.pStages = stages;
	gpi.pVertexInputState = &vin;
	gpi.pInputAssemblyState = &ia;
	gpi.pViewportState = &vps;
	gpi.pRasterizationState = &rs;
	gpi.pMultisampleState = &ms;
	gpi.pColorBlendState = &cb;
	gpi.pDepthStencilState = depth ? &dss : nullptr;
	gpi.layout = layout;
	gpi.renderPass = depth ? c.renderPassDepth : c.renderPass;
	VkPipeline pipeline;
	VKCHECK(vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline));

	VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cai.commandPool = c.pool;
	cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cai.commandBufferCount = 1;
	VkCommandBuffer cmd;
	VKCHECK(vkAllocateCommandBuffers(c.device, &cai, &cmd));
	VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VKCHECK(vkBeginCommandBuffer(cmd, &cbi));
	VkClearValue clear[2];
	clear[0].color = {{-7.0f, 13.0f, -21.0f, 42.0f}}; // a discarded pixel keeps it
	clear[1].depthStencil = {0.625f, 0};
	VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
	rbi.renderPass = depth ? c.renderPassDepth : c.renderPass;
	rbi.framebuffer = depth ? c.framebufferDepth : c.framebuffer;
	rbi.renderArea = {{0, 0}, {W, H}};
	rbi.clearValueCount = depth ? 2 : 1;
	rbi.pClearValues = clear;
	vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	if (nb)
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &ds, 0, nullptr);
	vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, inputs);
	vkCmdDraw(cmd, points ? 1 : 3, 1, 0, 0);
	vkCmdEndRenderPass(cmd);
	VkBufferImageCopy region{};
	region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	region.imageExtent = {W, H, 1};
	vkCmdCopyImageToBuffer(cmd, c.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.readback, 1, &region);
	if (depth)
	{
		region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
		vkCmdCopyImageToBuffer(cmd, c.depthTarget, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.depthReadback, 1, &region);
	}
	VKCHECK(vkEndCommandBuffer(cmd));
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	VKCHECK(vkCreateFence(c.device, &fi, nullptr, &fence));
	VKCHECK(vkQueueSubmit(c.queue, 1, &si, fence));
	// a shader that never ends (a loop that does not break) fails here,
	// rather than hanging the test
	if (vkWaitForFences(c.device, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000) != VK_SUCCESS)
	{
		std::printf("FAIL: a draw did not finish within 30 s (a loop that never ends?)\n");
		std::fflush(stdout);
		std::_Exit(1);
	}
	vkDestroyFence(c.device, fence, nullptr);
	std::vector<uint32> pixels(W * H * 4);
	void* map;
	VKCHECK(vkMapMemory(c.device, c.readMem, 0, pixels.size() * 4, 0, &map));
	std::memcpy(pixels.data(), map, pixels.size() * 4);
	vkUnmapMemory(c.device, c.readMem);
	if (depth)
	{
		pixels.resize(W * H * 5);
		VKCHECK(vkMapMemory(c.device, c.depthReadMem, 0, W * H * 4, 0, &map));
		std::memcpy(pixels.data() + W * H * 4, map, W * H * 4);
		vkUnmapMemory(c.device, c.depthReadMem);
	}
	vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
	vkDestroyPipeline(c.device, pipeline, nullptr);
	vkDestroyShaderModule(c.device, fs, nullptr);
	vkDestroyDescriptorPool(c.device, dpool, nullptr);
	vkDestroyPipelineLayout(c.device, layout, nullptr);
	vkDestroyDescriptorSetLayout(c.device, set, nullptr);
	vkDestroyDescriptorSetLayout(c.device, empty, nullptr);
	return pixels;
}

static bool SameOffsets(const LatteDecompilerOutputUniformOffsets& a, const LatteDecompilerOutputUniformOffsets& b)
{
	if (a.offset_remapped != b.offset_remapped || a.offset_uniformRegister != b.offset_uniformRegister ||
		a.count_uniformRegister != b.count_uniformRegister || a.offset_alphaTestRef != b.offset_alphaTestRef ||
		a.offset_pointSize != b.offset_pointSize || a.offset_fragCoordScale != b.offset_fragCoordScale ||
		a.offset_windowSpaceToClipSpaceTransform != b.offset_windowSpaceToClipSpaceTransform || a.offset_endOfBlock != b.offset_endOfBlock ||
		a.offset_verticesPerInstance != b.offset_verticesPerInstance)
		return false;
	for (int i = 0; i < LATTE_NUM_STREAMOUT_BUFFER; i++)
		if (a.offset_streamoutBufferBase[i] != b.offset_streamoutBufferBase[i])
			return false;
	for (int i = 0; i < LATTE_NUM_MAX_TEX_UNITS; i++)
		if (a.offset_texScale[i] != b.offset_texScale[i])
			return false;
	return true;
}

struct Case
{
	Program p;
	uint32 alphaRaw;
	bool strictMul, rte, preventLoops, usesGS;
	bool roundPrecision; // a texel edge title's (ForceSamplerRoundToPrecision)
	float inputs[16];
	uint32 cfile[256 * 4];
	float alphaRef;
	float texScale[4][2]; // uf_tex<n>Scale
	// input 1 gl_FragCoord, uf_fragCoordScale; SPI_PS_IN_CONTROL_1 (front
	// face into a register); the triangle drawn as a back face
	bool fragCoord, cw;
	float fragScale[4];
	uint32 psControl1;
	// drawn as one point over the target; param gen's register (-1: none)
	bool points;
	sint32 paramGenGpr;
};

// the same bits, or both a NaN: which NaN an operation on a NaN gives is
// the driver's (its optimizer reorders, folds and fuses as the code's
// shape lets it), as on any GPU, and the R600's own differ again
static bool IsNaN(uint32 b)
{
	return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0;
}
static bool SamePixels(const std::vector<uint32>& a, const std::vector<uint32>& b)
{
	if (a.size() != b.size())
		return false;
	for (size_t i = 0; i < a.size(); i++)
		if (a[i] != b[i] && !(IsNaN(a[i]) && IsNaN(b[i])))
			return false;
	return true;
}

// What a module computes and how it branches: its instructions counted by
// opcode (and GLSL.std.450 instruction), leaving out how values move
// through memory and vectors, which rslc and the emitter do differently
// and a driver makes the same of (loads, stores, access chains,
// extracting and building vectors, variables, names, types, constants).
// The direct module mirrors rslc's in all the rest: the same operations,
// the same constants folded, the same selects, branches and calls - which
// is what makes a driver's optimizer treat the two alike.
static std::map<uint32, int> Shape(const std::vector<uint32>& m)
{
	std::map<uint32, int> h;
	for (size_t i = 5; i < m.size();)
	{
		uint32 n = m[i] >> 16, op = m[i] & 0xFFFF;
		if (n == 0)
			break;
		switch (op)
		{
		case 59: case 61: case 62: case 65: case 80: case 81: case 82: case 79: // memory and vectors
		case 3: case 4: case 7: case 8: case 5: case 6: case 71: case 72: case 248: // debug, names, decorations, labels
			break;
		case 12: // OpExtInst: by its instruction
			h[0x10000 | m[i + 4]]++;
			break;
		default:
			if (op < 19 || op > 46) // not a type or constant
				h[op]++;
		}
		i += n;
	}
	return h;
}

static std::string ShapeDiff(const std::map<uint32, int>& a, const std::map<uint32, int>& b)
{
	std::string out;
	std::map<uint32, int> all = a;
	for (auto& it : b)
		all[it.first];
	for (auto& it : all)
	{
		auto x = a.find(it.first);
		auto y = b.find(it.first);
		int na = x == a.end() ? 0 : x->second, nb = y == b.end() ? 0 : y->second;
		if (na != nb)
			out += fmt::format(" {}{}: {} via GLSL, {} directly;", it.first & 0x10000 ? "GLSL.std.450 " : "op ", it.first & 0xFFFF, na, nb);
	}
	return out;
}

// 1: drawn differently, 0: the same, -1: not drawn both ways
static int Differs(const Context& c, const Case& k, std::vector<uint32>* outA = nullptr, std::vector<uint32>* outB = nullptr)
{
	std::vector<uint32> words = Assemble(k.p);
	s_regs[Latte::REGADDR::SX_ALPHA_TEST_CONTROL] = k.alphaRaw;
	g_testPreventInfiniteLoops = k.preventLoops;
	g_testRoundToPrecision = k.roundPrecision;
	SetPSInputs(k.usesGS, k.fragCoord, k.psControl1, k.paramGenGpr);
	SetTextureRegisters(s_regs, k.p);
	Decompiled viaGlsl = Decompile(words, false, k.strictMul, k.rte);
	Decompiled viaSpirv = Decompile(words, true, k.strictMul, k.rte);
	if (viaSpirv.declined || !viaGlsl.ok || !viaSpirv.ok)
		return -1;
	uint32 uniforms[2048] = {}; // a full register file (4 KiB) and what follows it
	if (viaGlsl.uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_REMAPPED)
	{
		for (auto& e : viaGlsl.remapped)
			std::memcpy(uniforms + viaGlsl.offsets.offset_remapped / 4 + e.mappedIndex * 4, RemappedSource(e, k.cfile), 16);
	}
	else if (viaGlsl.uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CFILE)
		std::memcpy(uniforms + viaGlsl.offsets.offset_uniformRegister / 4, k.cfile, viaGlsl.offsets.count_uniformRegister * 16);
	if (viaGlsl.offsets.offset_alphaTestRef >= 0)
		std::memcpy(uniforms + viaGlsl.offsets.offset_alphaTestRef / 4, &k.alphaRef, 4);
	for (int t = 0; t < 4; t++)
	{
		if (viaGlsl.offsets.offset_texScale[t] >= 0)
			std::memcpy(uniforms + viaGlsl.offsets.offset_texScale[t] / 4, k.texScale[t], 8);
	}
	if (viaGlsl.offsets.offset_fragCoordScale >= 0)
		std::memcpy(uniforms + viaGlsl.offsets.offset_fragCoordScale / 4, k.fragScale, 16);
	const bool depth = k.p.depthGpr >= 0;
	std::vector<uint32> a = Draw(c, viaGlsl, k.p.texKind, k.inputs, uniforms, sizeof(uniforms), k.cw, depth, k.points);
	std::vector<uint32> b = Draw(c, viaSpirv, k.p.texKind, k.inputs, uniforms, sizeof(uniforms), k.cw, depth, k.points);
	if (k.p.expectPairs)
	{
		for (const std::vector<uint32>* px : { &a, &b })
		{
			for (int ch = 0; ch < 2; ch++)
			{
				if ((*px)[ch] != (*px)[ch + 2])
				{
					std::printf("FAIL: pixel 0's channel %d is %08x, channel %d %08x\n", ch, (*px)[ch], ch + 2, (*px)[ch + 2]);
					s_failures++;
				}
			}
		}
	}
	if (k.p.sysInputs)
	{
		// pixel 0's centre (0.5, 0.5) scaled; the triangle's signed area is
		// negative in framebuffer space, front facing where clockwise is
		const float want[3] = { 1.0f, 2.0f, k.cw ? 1.0f : 0.0f };
		for (int ch = 0; ch < 3; ch++)
		{
			for (const std::vector<uint32>* px : { &a, &b })
			{
				float v;
				std::memcpy(&v, px->data() + ch, 4);
				if (v != want[ch])
				{
					std::printf("FAIL: pixel 0's system input %d is %f, not %f\n", ch, v, want[ch]);
					s_failures++;
				}
			}
		}
	}
	if (k.p.expectDepth)
	{
		for (const std::vector<uint32>* px : { &a, &b })
		{
			float d0 = 0.0f;
			if (px->size() > W * H * 4)
				std::memcpy(&d0, px->data() + W * H * 4, 4);
			if (d0 != k.p.depth0)
			{
				std::printf("FAIL: pixel 0's depth is %f, not %f\n", d0, k.p.depth0);
				s_failures++;
			}
		}
	}
	if (k.p.expectPixel && (k.p.pointCoordGpr < 0 || k.points))
	{
		for (int ch = 0; ch < 4; ch++)
		{
			for (const std::vector<uint32>* px : { &a, &b })
			{
				if ((*px)[ch] != k.p.pixel0[ch])
				{
					std::printf("FAIL: pixel 0's channel %d is %08x, not %08x\n", ch, (*px)[ch], k.p.pixel0[ch]);
					s_failures++;
				}
			}
		}
	}
	if (k.p.expectFetch)
	{
		// pixel 0: buffer 1's vec4 5's z; buffer 2's vec4 9's w, x and y
		const uint32 want[4] = { s_cbuf[1][5 * 4 + 2], s_cbuf[2][9 * 4 + 3], s_cbuf[2][9 * 4 + 0], s_cbuf[2][9 * 4 + 1] };
		for (int ch = 0; ch < 4; ch++)
		{
			for (const std::vector<uint32>* px : { &a, &b })
			{
				if ((*px)[ch] != want[ch] && !(IsNaN((*px)[ch]) && IsNaN(want[ch])))
				{
					std::printf("FAIL: the uniform buffer fetches' channel %d read %08x, not %08x\n", ch, (*px)[ch], want[ch]);
					s_failures++;
				}
			}
		}
	}
	if (k.p.expectRemap)
	{
		// pixel 0: c3.x and buffer 0's vec4 3's y, both ways
		const uint32 want[2] = { k.cfile[3 * 4 + 0], s_cbuf[0][3 * 4 + 1] };
		for (int ch = 0; ch < 2; ch++)
		{
			for (const std::vector<uint32>* px : { &a, &b })
			{
				if ((*px)[ch] != want[ch] && !(IsNaN((*px)[ch]) && IsNaN(want[ch])))
				{
					std::printf("FAIL: the remapped uniforms' channel %d read %08x, not %08x\n", ch, (*px)[ch], want[ch]);
					s_failures++;
				}
			}
		}
	}
	if (outA)
		*outA = viaGlsl.spirv;
	if (outB)
		*outB = viaSpirv.spirv;
	if (outA && !SamePixels(a, b))
	{
		for (size_t i = 0; i < a.size(); i++)
			if (a[i] != b[i] && !(IsNaN(a[i]) && IsNaN(b[i])))
				std::printf("  pixel %zu channel %zu: %08x via GLSL, %08x directly\n", i / 4, i % 4, a[i], b[i]);
	}
	return SamePixels(a, b) ? 0 : 1;
}

// the smallest program that still draws differently: groups, then
// instructions, taken out one at a time while it does
static void Minimize(const Context& c, Case k, const char* outDir)
{
	bool shrunk = true;
	while (shrunk)
	{
		shrunk = false;
		for (size_t cl = 0; cl < k.p.clauses.size(); cl++)
		{
			for (size_t g = 0; g < k.p.clauses[cl].size(); g++)
			{
				Case t = k;
				t.p.clauses[cl].erase(t.p.clauses[cl].begin() + g);
				if (t.p.clauses[cl].empty())
				{
					if (!t.p.cf.empty())
						continue; // the CF list refers to it
					t.p.clauses.erase(t.p.clauses.begin() + cl);
				}
				if (t.p.clauses.empty())
					continue;
				if (Differs(c, t) == 1)
				{
					k = t;
					shrunk = true;
					break;
				}
				// or one instruction of it
				for (size_t o = 0; o < k.p.clauses[cl][g].ops.size(); o++)
				{
					Case u = k;
					auto& ops = u.p.clauses[cl][g].ops;
					if (ops[o].opcode == ALU_OP2_INST_DOT4 || ops[o].opcode == ALU_OP2_INST_DOT4_IEEE)
						continue;
					ops.erase(ops.begin() + o);
					if (ops.empty())
						continue;
					if (Differs(c, u) == 1)
					{
						k = u;
						shrunk = true;
						break;
					}
				}
				if (shrunk)
					break;
			}
			if (shrunk)
				break;
		}
	}
	std::vector<uint32> a, b;
	Differs(c, k, &a, &b);
	std::vector<uint32> words = Assemble(k.p);
	g_testPreventInfiniteLoops = k.preventLoops;
	g_testRoundToPrecision = k.roundPrecision;
	SetTextureRegisters(s_regs, k.p);
	std::string glsl = Decompile(words, false, k.strictMul, k.rte).glsl;
	std::printf("minimal program (strictMul %d, rte %d, alpha %08x):\n%s\n", k.strictMul, k.rte, k.alphaRaw, glsl.c_str());
	if (outDir)
	{
		char path[512];
		std::snprintf(path, sizeof(path), "%s/min_glsl.spv", outDir);
		FILE* f = std::fopen(path, "wb");
		std::fwrite(a.data(), 4, a.size(), f);
		std::fclose(f);
		std::snprintf(path, sizeof(path), "%s/min_direct.spv", outDir);
		f = std::fopen(path, "wb");
		std::fwrite(b.data(), 4, b.size(), f);
		std::fclose(f);
	}
}

// A literal the GLSL emitter writes as a decimal (its low mantissa byte
// zero, its exponent within +-10), and that decimal as rslc reads it
static bool WrittenAsDecimal(uint32 bits)
{
	sint32 exponent = (sint32)((bits >> 23) & 0xFF) - 127;
	return (bits & 0xFF) == 0 && exponent >= -10 && exponent <= 10;
}
static double AsRead(uint32 bits)
{
	float f;
	std::memcpy(&f, &bits, 4);
	char text[64];
	size_t len = fmt::format_to_n(text, 63, "{:#}", f).size;
	text[len] = '\0';
	return std::strtod(text, nullptr);
}

// ADD of two literals where folding the decimals in double, as rslc
// does, gives another float than adding the floats: found by search
static bool FindFoldingPair(std::mt19937& rng, uint32& a, uint32& b)
{
	for (int i = 0; i < 2000000; i++)
	{
		a = (rng() & 0x3FFFFF00u) | 0x3E000000u;
		b = (rng() & 0x3FFFFF00u) | 0x3E000000u;
		if (!WrittenAsDecimal(a) || !WrittenAsDecimal(b))
			continue;
		float fa, fb;
		std::memcpy(&fa, &a, 4);
		std::memcpy(&fb, &b, 4);
		float folded = (float)(AsRead(a) + AsRead(b));
		float plain = fa + fb;
		if (std::memcmp(&folded, &plain, 4) != 0)
			return true;
	}
	return false;
}

// What random programs reach only now and then: constant folding that
// differs from float arithmetic (the decimals rslc reads, sin and a
// division in double), on each side of a ternary
static std::vector<Program> DirectedPrograms()
{
	std::vector<Program> out;
	std::mt19937 rng(7);
	auto one = [](const AluOp& op, uint32 l0, uint32 l1) {
		Program p{};
		AluGroup g{};
		g.ops.push_back(op);
		g.literals[0] = l0;
		g.literals[1] = l1;
		p.clauses.push_back({ g });
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		return p;
	};
	AluOp add{};
	add.opcode = ALU_OP2_INST_ADD;
	add.src[0] = 253;
	add.src[1] = 253;
	add.chan[1] = 1;
	add.writeMask = 1;
	uint32 a, b;
	if (FindFoldingPair(rng, a, b))
		out.push_back(one(add, a, b));
	AluOp sinOp = add;
	sinOp.opcode = ALU_OP2_INST_SIN;
	sinOp.src[0] = 249; // sin(1.0/0.1591549367), folded in double
	sinOp.destElem = 1;
	out.push_back(one(sinOp, 0, 0));
	AluOp recip = add;
	recip.opcode = ALU_OP2_INST_RECIP_IEEE;
	recip.destElem = 2;
	out.push_back(one(recip, 0x40400000u /* 3.0 */, 0));
	// a loop in an if that turns the if over inside it: its mask is off
	// at its end and must be put back for the clause after it, which only
	// the pixels in the if run
	{
		Program p{};
		auto clause = [&](std::vector<AluOp> ops, uint32 literal) {
			AluGroup g{};
			g.ops = std::move(ops);
			g.literals[0] = literal;
			p.clauses.push_back({ g });
			return (int)p.clauses.size() - 1;
		};
		AluOp pred{};
		pred.opcode = ALU_OP2_INST_PRED_SETGT;
		pred.src[0] = 3;   // R3.x, of either sign over the target
		pred.src[1] = 248; // 0.0
		pred.exec = 1;
		AluOp zero{};
		zero.opcode = ALU_OP2_INST_MOV;
		zero.src[0] = 253;
		zero.destGpr = kCounterGPR;
		zero.writeMask = 1;
		AluOp inc{};
		inc.opcode = ALU_OP2_INST_ADD;
		inc.src[0] = kCounterGPR;
		inc.src[1] = 249; // 1.0
		inc.destGpr = kCounterGPR;
		inc.writeMask = 1;
		AluOp limit{};
		limit.opcode = ALU_OP2_INST_PRED_SETGT;
		limit.src[0] = 253;
		limit.src[1] = kCounterGPR;
		limit.exec = 1;
		AluOp mark{};
		mark.opcode = ALU_OP2_INST_MOV;
		mark.src[0] = 253;
		mark.destGpr = 1;
		mark.writeMask = 1;
		const int cPush = clause({ pred }, 0);
		const int cZero = clause({ zero }, 0);
		const int cBreak = clause({ inc }, 0);
		p.clauses[cBreak].push_back(AluGroup{});
		p.clauses[cBreak].back().ops.push_back(limit);
		p.clauses[cBreak].back().literals[0] = 0x40000000u; // 2.0
		const int cMark = clause({ mark }, 0x40A00000u); // 5.0
		p.cf = {
			{ GPU7_CF_INST_ALU_PUSH_BEFORE, cPush, 0 },
			{ GPU7_CF_INST_ALU, cZero, 0 },
			{ GPU7_CF_INST_LOOP_START_DX10, -1, 0 },
			{ GPU7_CF_INST_ALU_BREAK, cBreak, 0 },
			{ GPU7_CF_INST_ELSE, -1, 0 },
			{ GPU7_CF_INST_LOOP_END, -1, 0 },
			{ GPU7_CF_INST_ALU, cMark, 0 },
			{ GPU7_CF_INST_POP, -1, 1 },
		};
		p.exportGpr = 1;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// texel fetches at known texels, the scale 1.0: an LD of (3, 5) of
	// level 0 and of (1, 2) of level 1; a SAMPLE of texel coordinates
	// (2.0, 6.0). Random registers seldom hold a texel's coordinates
	auto fetch = [](uint8 opcode, const uint32 r4[4], const uint32 r5[4]) {
		Program p{};
		AluGroup g4{}, g5{};
		for (uint8 c = 0; c < 4; c++)
		{
			AluOp mov{};
			mov.opcode = ALU_OP2_INST_MOV;
			mov.src[0] = 253;
			mov.chan[0] = c;
			mov.destGpr = 4;
			mov.destElem = c;
			mov.writeMask = 1;
			g4.ops.push_back(mov);
			g4.literals[c] = r4[c];
			mov.destGpr = 5;
			g5.ops.push_back(mov);
			g5.literals[c] = r5[c];
		}
		p.clauses.push_back({ g4, g5 });
		TexOp t{};
		t.opcode = opcode;
		t.srcGpr = 4;
		t.dstGpr = 0;
		t.unnormalized = opcode == GPU7_TEX_INST_SAMPLE;
		for (uint8 c = 0; c < 4; c++)
		{
			t.srcSel[c] = c;
			t.dstSel[c] = c < 2 ? c : 7;
		}
		TexOp u = t;
		u.srcGpr = 5;
		for (uint8 c = 0; c < 4; c++)
			u.dstSel[c] = c < 2 ? 7 : c;
		p.texClauses.push_back({ t, u });
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 }, { GPU7_CF_INST_TEX, 0, 0 } };
		p.texKind[0] = TEX_2D;
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		return p;
	};
	// a loop in a loop, counted: R1.x summed over the inner one's runs,
	// 3 times 2, from R0.x; the loop stack's masks and counters nested
	{
		Program p{};
		auto clause = [&](std::vector<std::vector<AluOp>> groups, uint32 literal) {
			std::vector<AluGroup> gs;
			for (auto& ops : groups)
			{
				AluGroup g{};
				g.ops = ops;
				g.literals[0] = literal;
				gs.push_back(g);
			}
			p.clauses.push_back(gs);
			return (int)p.clauses.size() - 1;
		};
		auto mov0 = [](uint8 elem) {
			AluOp op{};
			op.opcode = ALU_OP2_INST_MOV;
			op.src[0] = 248; // 0.0
			op.destGpr = kCounterGPR;
			op.destElem = elem;
			op.writeMask = 1;
			return op;
		};
		auto inc = [](uint8 elem) {
			AluOp op{};
			op.opcode = ALU_OP2_INST_ADD;
			op.src[0] = kCounterGPR;
			op.chan[0] = elem;
			op.src[1] = 249; // 1.0
			op.destGpr = kCounterGPR;
			op.destElem = elem;
			op.writeMask = 1;
			return op;
		};
		auto limit = [](uint8 elem) {
			AluOp op{};
			op.opcode = ALU_OP2_INST_PRED_SETGT;
			op.src[0] = 253;
			op.src[1] = kCounterGPR;
			op.chan[1] = elem;
			op.exec = 1;
			return op;
		};
		AluOp acc{};
		acc.opcode = ALU_OP2_INST_ADD;
		acc.src[0] = 1;
		acc.src[1] = 0;
		acc.destGpr = 1;
		acc.writeMask = 1;
		AluOp clear{};
		clear.opcode = ALU_OP2_INST_MOV;
		clear.src[0] = 248;
		clear.destGpr = 1;
		clear.writeMask = 1;
		const int cStart = clause({ { mov0(0), clear } }, 0);
		const int cOuter = clause({ { inc(0) }, { limit(0) } }, 0x40800000u); // 4.0: 3 runs
		const int cInner0 = clause({ { mov0(1) } }, 0);
		const int cInner = clause({ { inc(1) }, { acc }, { limit(1) } }, 0x40400000u); // 3.0: 2 runs
		p.cf = {
			{ GPU7_CF_INST_ALU, cStart, 0 },
			{ GPU7_CF_INST_LOOP_START_DX10, -1, 0 },
			{ GPU7_CF_INST_ALU_BREAK, cOuter, 0 },
			{ GPU7_CF_INST_ALU, cInner0, 0 },
			{ GPU7_CF_INST_LOOP_START_DX10, -1, 0 },
			{ GPU7_CF_INST_ALU_BREAK, cInner, 0 },
			{ GPU7_CF_INST_LOOP_END, -1, 0 },
			{ GPU7_CF_INST_LOOP_END, -1, 0 },
		};
		p.exportGpr = 1;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// the level queries at a known level of detail: R0.x varies over the
	// target as 1.516 * x + 0.1, 3 texels of 8 a pixel (level 1.6), R0.y
	// stays; the lod and its rounding, then the constants of a RESINFO
	{
		Program p{};
		TexOp q{};
		q.opcode = GPU7_TEX_INST_GET_COMP_TEX_LOD;
		q.srcGpr = 0;
		q.dstGpr = 2;
		for (uint8 c = 0; c < 4; c++)
			q.srcSel[c] = c;
		q.dstSel[0] = 0;
		q.dstSel[1] = 1;
		q.dstSel[2] = 7;
		q.dstSel[3] = 7;
		TexOp r = q;
		r.opcode = GPU7_TEX_INST_GET_TEXTURE_RESINFO;
		r.srcSel[3] = 4; // level 0
		r.dstSel[0] = 7;
		r.dstSel[1] = 7;
		r.dstSel[2] = 5;
		r.dstSel[3] = 4;
		p.texClauses.push_back({ q, r });
		p.cf = { { GPU7_CF_INST_TEX, 0, 0 } };
		p.texKind[0] = TEX_2D;
		p.exportGpr = 2;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		p.inputs.assign(16, 0.0f);
		p.inputs[0] = 1.516f; // v[0].x
		p.inputs[4] = 0.1f;   // v[1].x
		p.inputs[5] = 0.4f;   // v[1].y
		out.push_back(p);
	}
	// compares of known coordinates and references with a 2D's depths:
	// at level 0, at level 1, with an offset, and of the implicit level
	{
		Program p{};
		AluGroup g{};
		const uint32 r4[4] = { 0x3E99999Au /* 0.3 */, 0x3F19999Au /* 0.6 */, 0x3F000000u /* 0.5 */, 0x3F800000u /* 1.0 */ };
		for (uint8 c = 0; c < 4; c++)
		{
			AluOp mov{};
			mov.opcode = ALU_OP2_INST_MOV;
			mov.src[0] = 253;
			mov.chan[0] = c;
			mov.destGpr = 4;
			mov.destElem = c;
			mov.writeMask = 1;
			g.ops.push_back(mov);
			g.literals[c] = r4[c];
		}
		p.clauses.push_back({ g });
		std::vector<TexOp> clause;
		const uint8 ops[4] = { GPU7_TEX_INST_SAMPLE_C_LZ, GPU7_TEX_INST_SAMPLE_C_L, GPU7_TEX_INST_SAMPLE_C, GPU7_TEX_INST_SAMPLE_C };
		for (uint8 c = 0; c < 4; c++)
		{
			TexOp t{};
			t.opcode = ops[c];
			t.srcGpr = 4;
			t.dstGpr = 0;
			t.srcSel[0] = 0;
			t.srcSel[1] = 1;
			t.srcSel[2] = 2; // C_L's reference
			t.srcSel[3] = ops[c] == GPU7_TEX_INST_SAMPLE_C_L ? 3 : 2; // its level, the others' reference
			for (uint8 d = 0; d < 4; d++)
				t.dstSel[d] = d == c ? 0 : 7;
			if (c == 2)
				t.offset[0] = t.offset[1] = 2; // a texel over
			clause.push_back(t);
		}
		p.texClauses.push_back(clause);
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 }, { GPU7_CF_INST_TEX, 0, 0 } };
		p.texKind[0] = TEX_2D;
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// CUBE of known (x, y, z), each major axis negative in turn (z, y, x):
	// the S and T of each and their faces summed into R0
	{
		Program p{};
		std::vector<AluGroup> groups;
		const float xyz[3][3] = { { 0.25f, 0.5f, -2.0f }, { 0.25f, -2.0f, 0.5f }, { -2.0f, 0.5f, 0.25f } };
		for (uint8 r = 0; r < 3; r++)
		{
			AluGroup g{};
			std::memcpy(&g.literals[0], &xyz[r][2], 4); // z: src0.x
			std::memcpy(&g.literals[1], &xyz[r][1], 4); // y: src1.x
			std::memcpy(&g.literals[2], &xyz[r][0], 4); // x: src1.y
			for (uint8 e = 0; e < 4; e++)
			{
				AluOp op{};
				op.opcode = ALU_OP2_INST_CUBE;
				op.src[0] = e == 0 ? 253 : 248;
				op.chan[0] = 0;
				op.src[1] = e < 2 ? 253 : 248;
				op.chan[1] = e == 0 ? 1 : 2;
				op.destGpr = (uint8)(1 + r);
				op.destElem = e;
				op.writeMask = 1;
				g.ops.push_back(op);
			}
			groups.push_back(g);
		}
		AluGroup sum{};
		auto add = [&](uint8 elem, uint8 a, uint8 ac, uint8 b, uint8 bc) {
			AluOp op{};
			op.opcode = ALU_OP2_INST_ADD;
			op.src[0] = a;
			op.chan[0] = ac;
			op.src[1] = b;
			op.chan[1] = bc;
			op.destGpr = 0;
			op.destElem = elem;
			op.writeMask = 1;
			sum.ops.push_back(op);
		};
		add(0, 1, 0, 1, 1); // z: S + T
		add(1, 1, 3, 2, 3); // the z and y faces
		add(2, 2, 0, 2, 1); // y: S + T
		add(3, 3, 1, 3, 3); // x: T and its face
		groups.push_back(sum);
		p.clauses.push_back(groups);
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 } };
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// uniform register c3 and buffer 0's vec4 3, remapped: each read from
	// its own entry, where both decompilers once took the register's for
	// the buffer's (the entries' indices alike, a register's bank 0)
	{
		Program p{};
		AluGroup g{};
		AluOp reg{};
		reg.opcode = ALU_OP2_INST_MOV;
		reg.src[0] = 256 + 3;
		reg.chan[0] = 0;
		reg.destGpr = 0;
		reg.destElem = 0;
		reg.writeMask = 1;
		AluOp buf = reg;
		buf.src[0] = 128 + 3;
		buf.chan[0] = 1;
		buf.destElem = 1;
		g.ops = { reg, buf };
		p.clauses.push_back({ g });
		p.kcache.push_back(Program::Kcache{});
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		p.expectRemap = true;
		out.push_back(p);
	}
	// uniform buffer VFETCHes at known indices (R1.y 5, R1.x 9): one
	// channel of another (z into x), then three swizzled (wxy into yzw)
	{
		Program p{};
		AluGroup g{};
		AluOp idx{};
		idx.opcode = ALU_OP2_INST_MOV;
		idx.src[0] = 253;
		idx.chan[0] = 0;
		idx.destGpr = 1;
		idx.destElem = 1;
		idx.writeMask = 1;
		AluOp idx2 = idx;
		idx2.chan[0] = 1;
		idx2.destElem = 0;
		g.ops = { idx, idx2 };
		g.literals[0] = 5;
		g.literals[1] = 9;
		p.clauses.push_back({ g });
		TexOp a{}, b{};
		a.vfetch = b.vfetch = true;
		a.opcode = b.opcode = GPU7_TEX_INST_VFETCH;
		a.unit = 0x81;
		a.srcGpr = 1;
		a.srcSel[0] = 1;
		a.dstGpr = 0;
		const uint8 selA[4] = { 2, 7, 7, 7 };
		b.unit = 0x82;
		b.srcGpr = 1;
		b.srcSel[0] = 0;
		b.dstGpr = 0;
		const uint8 selB[4] = { 7, 3, 0, 1 };
		for (int c = 0; c < 4; c++)
		{
			a.dstSel[c] = selA[c];
			b.dstSel[c] = selB[c];
		}
		p.texClauses.push_back({ a, b });
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 }, { GPU7_CF_INST_TEX, 0, 0 } };
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		p.expectFetch = true;
		out.push_back(p);
	}
	// a cube array's layer by a constant select (1.0: the second cube), then
	// a sample of it at known st and face; the GLSL emitter once indexed
	// its select table by the select, 5, out of the table. Then the layer
	// from w (0.0: the first cube) and a sample again
	{
		Program p{};
		AluGroup g{};
		const uint32 r4[4] = { 0x3F99999Au /* 1.2 */, 0x3FD9999Au /* 1.7 */, 0x40000000u /* 2.0 */, 0 };
		for (uint8 c = 0; c < 4; c++)
		{
			AluOp mov{};
			mov.opcode = ALU_OP2_INST_MOV;
			mov.src[0] = 253;
			mov.chan[0] = c;
			mov.destGpr = 4;
			mov.destElem = c;
			mov.writeMask = 1;
			g.ops.push_back(mov);
			g.literals[c] = r4[c];
		}
		p.clauses.push_back({ g });
		TexOp set{}, sample{};
		set.opcode = GPU7_TEX_INST_SET_CUBEMAP_INDEX;
		set.srcGpr = 4;
		set.srcSel[0] = 5;
		sample.opcode = GPU7_TEX_INST_SAMPLE_LZ;
		sample.srcGpr = 4;
		for (uint8 c = 0; c < 4; c++)
		{
			set.dstSel[c] = 7;
			sample.srcSel[c] = c;
			sample.dstSel[c] = c;
		}
		set.srcSel[1] = set.srcSel[2] = set.srcSel[3] = 0;
		for (uint8 c = 0; c < 4; c++)
			sample.dstSel[c] = c < 2 ? c : 7;
		TexOp set2 = set, sample2 = sample;
		set2.srcSel[0] = 3;
		for (uint8 c = 0; c < 4; c++)
			sample2.dstSel[c] = c < 2 ? 7 : c;
		p.texClauses.push_back({ set, sample, set2, sample2 });
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 }, { GPU7_CF_INST_TEX, 0, 0 } };
		p.texKind[0] = TEX_CUBE;
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
		// the same with the registers one array (a write relative to AR):
		// the layer read from R4.w by the array's name, where the GLSL
		// emitter once named R4i, undeclared, and did not compile
		AluGroup rel{};
		AluOp w{};
		w.opcode = ALU_OP2_INST_MOV;
		w.src[0] = 4;
		w.chan[0] = 0;
		w.destGpr = 5;
		w.destElem = 0;
		w.writeMask = 1;
		w.destRel = 1;
		w.indexMode = GPU7_INDEX_AR_X;
		rel.ops.push_back(w);
		p.clauses[0].push_back(rel);
		out.push_back(p);
	}
	// registers written relative to AR (the registers one array): MOVA_INT
	// 1, then R[2+AR.x].y and, by AR.y (still 0), R[4+AR.y].z set from
	// literals; R3 and R4 read back plainly, R[1+AR.x] (R2) relative
	{
		Program p{};
		auto op2 = [](uint32 opcode, uint8 dst, uint8 elem, uint32 src, uint8 chan) {
			AluOp op{};
			op.opcode = opcode;
			op.src[0] = src;
			op.chan[0] = chan;
			op.destGpr = dst;
			op.destElem = elem;
			op.writeMask = 1;
			return op;
		};
		AluGroup init{};
		for (uint8 r = 0; r < 4; r++)
		{
			init.ops.push_back(op2(ALU_OP2_INST_MOV, (uint8)(1 + r), r, 253, r));
			init.literals[r] = 0x100u * (r + 1);
		}
		AluGroup ar{};
		AluOp mova = op2(ALU_OP2_INST_MOVA_INT, 0, 0, 253, 0);
		mova.writeMask = 0;
		ar.ops.push_back(mova);
		ar.literals[0] = 1;
		AluGroup writes{};
		AluOp w1 = op2(ALU_OP2_INST_MOV, 2, 1, 253, 0);
		w1.destRel = 1;
		w1.indexMode = GPU7_INDEX_AR_X;
		writes.ops.push_back(w1);
		writes.literals[0] = 0x7777u;
		AluGroup writes2{};
		AluOp w2 = op2(ALU_OP2_INST_MOV, 4, 2, 253, 1);
		w2.destRel = 1;
		w2.indexMode = GPU7_INDEX_AR_Y;
		writes2.ops.push_back(w2);
		writes2.literals[1] = 0x8888u;
		AluGroup reads{};
		reads.ops.push_back(op2(ALU_OP2_INST_MOV, 0, 0, 3, 1)); // R3.y
		reads.ops.push_back(op2(ALU_OP2_INST_MOV, 0, 1, 4, 2)); // R4.z
		AluOp rr = op2(ALU_OP2_INST_MOV, 0, 2, 1, 2);           // R[1+AR.x].z
		rr.rel[0] = 1;
		rr.indexMode = GPU7_INDEX_AR_X;
		reads.ops.push_back(rr);
		reads.ops.push_back(op2(ALU_OP2_INST_MOV, 0, 3, 3, 3)); // R3.w
		p.clauses.push_back({ init, ar, writes, writes2, reads });
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 } };
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// AR set by MOVA_INT 2, MOVA_FLOOR 1.7 and MOVA_INT 300 (out of range:
	// -256), then registers and uniforms read relative to it: R[1+AR.x]
	// (R3), c[1+AR.y] (c2), R[1+AR.z] (none: 0), and the MOVAs' own outputs
	// summed from PV (2 + -256)
	{
		Program p{};
		auto mov = [](uint8 dst, uint8 elem, uint32 src, uint8 chan) {
			AluOp op{};
			op.opcode = ALU_OP2_INST_MOV;
			op.src[0] = src;
			op.chan[0] = chan;
			op.destGpr = dst;
			op.destElem = elem;
			op.writeMask = 1;
			return op;
		};
		AluGroup regs{};
		for (uint8 r = 0; r < 4; r++)
		{
			regs.ops.push_back(mov((uint8)(1 + r), 0, 253, r));
			regs.literals[r] = 0x1000u + r; // R1.x to R4.x
		}
		AluGroup ar{};
		const float floor17 = 1.7f;
		ar.literals[0] = 2;
		std::memcpy(&ar.literals[1], &floor17, 4);
		ar.literals[2] = 300;
		const uint32 movas[3] = { ALU_OP2_INST_MOVA_INT, ALU_OP2_INST_MOVA_FLOOR, ALU_OP2_INST_MOVA_INT };
		for (uint8 e = 0; e < 3; e++)
		{
			AluOp op = mov(0, e, 253, e);
			op.opcode = movas[e];
			op.writeMask = 0;
			ar.ops.push_back(op);
		}
		const uint32 srcs[3] = { 1, 256 + 1, 1 };
		const uint8 modes[3] = { GPU7_INDEX_AR_X, GPU7_INDEX_AR_Y, GPU7_INDEX_AR_Z };
		// one index mode an instruction: a group for each read
		std::vector<AluGroup> groups = { regs, ar };
		for (uint8 e = 0; e < 3; e++)
		{
			AluGroup g{};
			AluOp op = mov(0, e, srcs[e], 0);
			op.rel[0] = 1;
			op.indexMode = modes[e];
			g.ops.push_back(op);
			if (e == 0)
			{
				AluOp sum = mov(0, 3, 254, 0); // PV.x + PV.z
				sum.opcode = ALU_OP2_INST_ADD_INT;
				sum.src[1] = 254;
				sum.chan[1] = 2;
				g.ops.push_back(sum);
			}
			groups.push_back(g);
		}
		p.clauses.push_back(groups);
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 } };
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// a read relative to AR in a float shader (no MOVA, an integer one; AR
	// 0) without the register array: R1.x to R4.x floats, R[2+AR.x].x (R2.x)
	// into R0.x, a float sum into R0.y and R[20+AR.x].x (never written: the
	// chain's last 0.0) into R0.z; the chain of floats, where the
	// GLSL and MSL emitters once ended it with an int 0 and did not compile
	{
		Program p{};
		auto mov = [](uint8 dst, uint8 elem, uint32 src, uint8 chan) {
			AluOp op{};
			op.opcode = ALU_OP2_INST_MOV;
			op.src[0] = src;
			op.chan[0] = chan;
			op.destGpr = dst;
			op.destElem = elem;
			op.writeMask = 1;
			return op;
		};
		AluGroup regs{};
		for (uint8 r = 0; r < 4; r++)
		{
			regs.ops.push_back(mov((uint8)(1 + r), 0, 253, r));
			const float f = 1.5f + (float)r;
			std::memcpy(&regs.literals[r], &f, 4);
		}
		AluGroup rd{};
		AluOp op = mov(0, 0, 2, 0);
		op.rel[0] = 1;
		op.indexMode = GPU7_INDEX_AR_X;
		rd.ops.push_back(op);
		AluOp sum = mov(0, 1, 2, 0); // R2.x + R4.x
		sum.opcode = ALU_OP2_INST_ADD;
		sum.src[1] = 4;
		sum.chan[1] = 0;
		rd.ops.push_back(sum);
		AluOp none = mov(0, 2, 20, 0);
		none.rel[0] = 1;
		none.indexMode = GPU7_INDEX_AR_X;
		rd.ops.push_back(none);
		p.clauses.push_back({ regs, rd });
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 } };
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// the integer -1 read as a uint (UINT_TO_FLOAT: 4294967296.0) and as a
	// float (SETE of it with itself, a NaN: 0.0), and the integer 1 as a
	// float (a denormal, times 2^126), where the
	// GLSL and MSL emitters once wrote nothing for them and did not compile
	{
		Program p{};
		AluGroup g{};
		AluOp u{};
		u.opcode = ALU_OP2_INST_UINT_TO_FLOAT;
		u.src[0] = 251;
		u.destGpr = 0;
		u.destElem = 0;
		u.writeMask = 1;
		AluOp f{};
		f.opcode = ALU_OP2_INST_SETE;
		f.src[0] = 251;
		f.src[1] = 251;
		f.destGpr = 0;
		f.destElem = 1;
		f.writeMask = 1;
		AluOp one = f;
		one.opcode = ALU_OP2_INST_MUL_IEEE;
		one.src[0] = 250; // the integer 1
		one.src[1] = 253; // 2^126: 2^-23 where the denormal is kept
		one.chan[1] = 0;
		one.destElem = 2;
		g.literals[0] = 0x7E800000u;
		g.ops = { u, f, one };
		p.clauses.push_back({ g });
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 } };
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// a 1D at known coordinates (unit 1): a sample a texel over, the
	// level 1 texel 3, its level count, and a gradient lookup of gradients
	// (0.25, 1.0) and (0.5, 0.125) - level 2 by the x components
	{
		Program p{};
		const uint32 regs[4][4] = {
			{ 0x3E99999Au /* 0.3 */, 0, 0, 0 },
			{ 0x3E800000u /* 0.25 */, 0x3F800000u /* 1.0 */, 0, 0 },
			{ 3, 0, 0, 1 },
			{ 0x3F000000u /* 0.5 */, 0x3E000000u /* 0.125 */, 0, 0 } };
		std::vector<AluGroup> groups;
		for (uint8 r = 0; r < 4; r++)
		{
			AluGroup g{};
			for (uint8 c = 0; c < 4; c++)
			{
				AluOp mov{};
				mov.opcode = ALU_OP2_INST_MOV;
				mov.src[0] = 253;
				mov.chan[0] = c;
				mov.destGpr = (uint8)(4 + r);
				mov.destElem = c;
				mov.writeMask = 1;
				g.ops.push_back(mov);
				g.literals[c] = regs[r][c];
			}
			groups.push_back(g);
		}
		p.clauses.push_back(groups);
		auto op = [](uint8 opcode, uint8 src, uint8 dstChan, uint8 dstSel) {
			TexOp t{};
			t.opcode = opcode;
			t.unit = 1;
			t.srcGpr = src;
			t.dstGpr = 0;
			for (uint8 c = 0; c < 4; c++)
			{
				t.srcSel[c] = c;
				t.dstSel[c] = c == dstChan ? dstSel : 7;
			}
			return t;
		};
		TexOp sample = op(GPU7_TEX_INST_SAMPLE, 4, 0, 0);
		sample.offset[0] = 2;
		TexOp setH = op(GPU7_TEX_INST_SET_GRADIENTS_H, 5, 0, 7), setV = op(GPU7_TEX_INST_SET_GRADIENTS_V, 7, 0, 7);
		setH.dstSel[0] = setV.dstSel[0] = 7;
		p.texClauses.push_back({ sample, op(GPU7_TEX_INST_LD, 6, 1, 0), op(GPU7_TEX_INST_GET_TEXTURE_RESINFO, 6, 2, 3),
			setH, setV, op(GPU7_TEX_INST_SAMPLE_G, 4, 3, 0) });
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 }, { GPU7_CF_INST_TEX, 0, 0 } };
		p.texKind[1] = TEX_1D;
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		out.push_back(p);
	}
	// a subroutine adding 1.0 to R1.x, called three times: before an if,
	// in it (false: the call masked off) and after it; R1 (1, 2, 3, 4)
	// first, so pixel 0 (3, 2, 3, 4)
	{
		Program p{};
		AluGroup g{};
		for (uint8 c = 0; c < 4; c++)
		{
			AluOp mov{};
			mov.opcode = ALU_OP2_INST_MOV;
			mov.src[0] = 253;
			mov.chan[0] = c;
			mov.destGpr = 1;
			mov.destElem = c;
			mov.writeMask = 1;
			g.ops.push_back(mov);
		}
		g.literals[0] = 0x3F800000u;
		g.literals[1] = 0x40000000u;
		g.literals[2] = 0x40400000u;
		g.literals[3] = 0x40800000u;
		AluOp pred{};
		pred.opcode = ALU_OP2_INST_PRED_SETGT;
		pred.src[0] = 248; // 0.0 > 1.0
		pred.src[1] = 249;
		pred.exec = 1;
		AluOp inc{};
		inc.opcode = ALU_OP2_INST_ADD;
		inc.src[0] = 1;
		inc.src[1] = 249;
		inc.destGpr = 1;
		inc.writeMask = 1;
		AluGroup gp{}, gi{};
		gp.ops.push_back(pred);
		gi.ops.push_back(inc);
		p.clauses = { { g }, { gp }, { gi } };
		p.cf = {
			{ GPU7_CF_INST_ALU, 0, 0 },
			{ GPU7_CF_INST_CALL, -1, 0 },
			{ GPU7_CF_INST_ALU_PUSH_BEFORE, 1, 0 },
			{ GPU7_CF_INST_CALL, -1, 0 },
			{ GPU7_CF_INST_POP, -1, 1 },
			{ GPU7_CF_INST_CALL, -1, 0 },
		};
		p.sub = { { GPU7_CF_INST_ALU, 2, 0 } };
		p.exportGpr = 1;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		p.expectPixel = true;
		p.pixel0[0] = 0x40400000u;
		p.pixel0[1] = 0x40000000u;
		p.pixel0[2] = 0x40400000u;
		p.pixel0[3] = 0x40800000u;
		out.push_back(p);
	}
	// gl_FragCoord's x and y and the front face (R1.x, R1.y, R0.y) into R2
	{
		Program p{};
		AluGroup g{};
		const uint8 srcGpr[3] = { 1, 1, 0 }, srcChan[3] = { 0, 1, 1 };
		for (uint8 c = 0; c < 3; c++)
		{
			AluOp mov{};
			mov.opcode = ALU_OP2_INST_MOV;
			mov.src[0] = srcGpr[c];
			mov.chan[0] = srcChan[c];
			mov.destGpr = 2;
			mov.destElem = c;
			mov.writeMask = 1;
			g.ops.push_back(mov);
		}
		p.clauses.push_back({ g });
		p.exportGpr = 2;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		p.sysInputs = true;
		out.push_back(p);
	}
	// param gen: gl_PointCoord into R4 (past the four inputs, which are
	// written after it), exported, of one point over the 4x4 target; pixel
	// 0's centre is (0.5, 0.5) of 4, so 0.125 each
	{
		Program p{};
		AluGroup g{};
		AluOp mov{};
		mov.opcode = ALU_OP2_INST_MOV;
		mov.src[0] = 249; // 1.0
		mov.destGpr = 2;
		mov.writeMask = 1;
		g.ops.push_back(mov);
		p.clauses.push_back({ g });
		p.exportGpr = 4;
		for (int i = 0; i < 4; i++)
		{
			p.exportSel[i] = (uint8)i;
			p.pixel0[i] = 0x3E000000u; // 0.125
		}
		p.pointCoordGpr = 4;
		p.expectPixel = true;
		out.push_back(p);
	}
	// the texel edge nudge's size: a nearest sample of the 8x8 uint texture
	// at u (3 - 0.003) / 8, which a nudge of 1/8/512 keeps in texel 2 and
	// one of twice that (or of level 1's size) moves into texel 3
	{
		Program p{};
		AluGroup g{};
		for (uint8 c = 0; c < 2; c++)
		{
			AluOp mov{};
			mov.opcode = ALU_OP2_INST_MOV;
			mov.src[0] = 253;
			mov.chan[0] = c;
			mov.destGpr = 4;
			mov.destElem = c;
			mov.writeMask = 1;
			g.ops.push_back(mov);
		}
		g.literals[0] = 0x3EBFCED9u; // (3 - 0.003) / 8
		g.literals[1] = 0x3EA00000u; // 2.5 / 8
		p.clauses.push_back({ g });
		TexOp t{};
		t.opcode = GPU7_TEX_INST_SAMPLE_LZ;
		t.srcGpr = 4;
		t.dstGpr = 0;
		for (uint8 c = 0; c < 4; c++)
		{
			t.srcSel[c] = c;
			t.dstSel[c] = c;
		}
		p.texClauses.push_back({ t });
		p.cf = { { GPU7_CF_INST_ALU, 0, 0 }, { GPU7_CF_INST_TEX, 0, 0 } };
		p.texKind[0] = TEX_2D_UINT;
		p.exportGpr = 0;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		p.roundPrecision = true;
		out.push_back(p);
	}
	// a depth export of a literal (0.375 into R2.y): pixel 0's depth
	{
		Program p{};
		AluGroup g{};
		AluOp mov{};
		mov.opcode = ALU_OP2_INST_MOV;
		mov.src[0] = 253;
		mov.destGpr = 2;
		mov.destElem = 1;
		mov.writeMask = 1;
		g.ops.push_back(mov);
		g.literals[0] = 0x3EC00000u; // 0.375
		p.clauses.push_back({ g });
		p.exportGpr = 2;
		for (int i = 0; i < 4; i++)
			p.exportSel[i] = (uint8)i;
		p.depthGpr = 2;
		p.depthSel = 1;
		p.expectDepth = true;
		p.depth0 = 0.375f;
		out.push_back(p);
	}
	const uint32 ld4[4] = { 3, 5, 0, 0 }, ld5[4] = { 1, 2, 0, 1 };
	out.push_back(fetch(GPU7_TEX_INST_LD, ld4, ld5));
	// texel fetches with an offset: an LD of (3, 5) offset by (1, -2) and a
	// texel SAMPLE of (2, 6) offset by (1, -2) into xy, the same texels
	// without one, (4, 3) and (3, 4), into zw: pixel 0's pairs equal
	{
		const uint32 a4[4] = { 3, 5, 0, 0 }, a5[4] = { 4, 3, 0, 0 };
		Program p = fetch(GPU7_TEX_INST_LD, a4, a5);
		const uint8 zw[4] = { 7, 7, 0, 1 }; // the second texel's x and y into zw
		std::memcpy(p.texClauses[0][1].dstSel, zw, 4);
		p.texClauses[0][0].offset[0] = 2; // half texels
		p.texClauses[0][0].offset[1] = -4;
		p.expectPairs = true;
		out.push_back(p);
		const uint32 b4[4] = { 0x40000000u, 0x40C00000u, 0, 0 }, b5[4] = { 0x40400000u, 0x40800000u, 0, 0 }; // (2, 6), (3, 4)
		p = fetch(GPU7_TEX_INST_SAMPLE, b4, b5);
		std::memcpy(p.texClauses[0][1].dstSel, zw, 4);
		p.texClauses[0][0].offset[0] = 2;
		p.texClauses[0][0].offset[1] = -4;
		p.expectPairs = true;
		out.push_back(p);
	}
	const uint32 s4[4] = { 0x40000000u, 0x40C00000u, 0, 0 }, s5[4] = { 0x40E00000u, 0x3F800000u, 0, 0 }; // (2, 6), (7, 1)
	out.push_back(fetch(GPU7_TEX_INST_SAMPLE, s4, s5));
	return out;
}

// --- vertex shaders ------------------------------------------------------------

// A random vertex shader: the program body of a pixel shader's (no kill),
// its position from an attribute into R8 (which nothing else writes), a
// grid of small triangles each covering one pixel, its parameters
// exported flat from random registers, and attributes of the formats the
// fetch shader decodes, of random data. Drawn with a fragment shader per
// parameter (and one for the depth), every pixel must match.
static const int kVerts = (int)(4 * 4 * 3);
static const int kPosGpr = 8;

struct VSCase
{
	Program p;
	std::vector<LatteParsedFetchShaderAttribute> attribs; // the position's first
	uint32 vtxSemantic[32];
	uint32 outId[8];
	uint8 psSemantic[8];
	int psCount;
	bool dxClip, viewportScale, strictMul, rte, preventLoops;
	bool roundPrecision;
	// drawn as points: of uf_pointSize's size, or the exported one
	bool points;
	// with a geometry shader (the test's): the position and the parameters
	// written to the ring, the parameters drawn from it
	bool usesGS;
	// a geometry shader's case: p the geometry shader, fed one point a pixel
	// by the test's vertex shader with gsIn ring parameters (the position
	// first), writing gsOut (the position first) for each of three vertices
	// of a triangle strip (or points) through its copy shader
	bool gsCase, gsPointsOut;
	bool gsTrianglesIn; // fed triangles (each pixel's three vertices alike), not points
	bool noCoverage; // a directed case drawing some pixels only
	// streamout enabled: the strides of buffers 0 and 1 in dwords, the
	// writes (a geometry shader's through its copy shader's), compared
	// with the pixels; a geometry shader's draws nothing then
	bool streamout;
	bool streamoutOff; // the writes there but VGT_STRMOUT_EN clear: dropped
	uint32 strmStride[2];
	int gsIn, gsOut;
	std::vector<uint32> copyShader;
	float pointSize;
	uint32 cfile[256 * 4];
	std::vector<uint8> vertexData; // 16 bytes a vertex, each attribute in turn
};

// the formats the fetch shader decodes (and a few it does not)
struct AttrFormat
{
	LatteConst::VertexFetchEndianMode endian;
	Latte::E_HWFMT fmt;
	uint8 nfa, isSigned;
};
static const AttrFormat kAttrFormats[] = {
#define F(e, f, n, s) { LatteConst::VertexFetchEndianMode::e, Latte::E_HWFMT::f, n, s }
	F(SWAP_U32, HWFMT_32_32_32_32_FLOAT, 2, 0), F(SWAP_U32, HWFMT_32_32_32_FLOAT, 2, 0), F(SWAP_U32, HWFMT_32_32_FLOAT, 2, 0),
	F(SWAP_U32, HWFMT_32_FLOAT, 2, 0), F(SWAP_U32, HWFMT_2_10_10_10, 0, 0), F(SWAP_U32, HWFMT_2_10_10_10, 0, 1),
	F(SWAP_U32, HWFMT_32_32_32_32, 1, 0), F(SWAP_U32, HWFMT_32_32_32, 1, 0), F(SWAP_U32, HWFMT_32_32, 1, 0),
	F(SWAP_U32, HWFMT_32, 1, 0), F(SWAP_U32, HWFMT_32, 1, 1), F(SWAP_U32, HWFMT_8_8_8_8, 0, 0), F(SWAP_U32, HWFMT_8_8_8_8, 0, 1),
	F(SWAP_U32, HWFMT_8_8_8_8, 1, 0), F(SWAP_U32, HWFMT_8_8_8_8, 2, 0),
	F(SWAP_NONE, HWFMT_32_32_32_32_FLOAT, 2, 0), F(SWAP_NONE, HWFMT_32_32_32_FLOAT, 2, 0), F(SWAP_NONE, HWFMT_32_32_FLOAT, 2, 0),
	F(SWAP_NONE, HWFMT_32, 1, 0), F(SWAP_NONE, HWFMT_2_10_10_10, 0, 0), F(SWAP_NONE, HWFMT_16_16_16_16, 0, 1),
	F(SWAP_NONE, HWFMT_16_16_16_16, 2, 1), F(SWAP_NONE, HWFMT_16_16_16_16_FLOAT, 2, 0), F(SWAP_NONE, HWFMT_16_16, 0, 1),
	F(SWAP_NONE, HWFMT_16_16_FLOAT, 2, 0), F(SWAP_NONE, HWFMT_8_8_8_8, 0, 0), F(SWAP_NONE, HWFMT_8_8_8_8, 0, 1),
	F(SWAP_NONE, HWFMT_8_8_8_8, 1, 0), F(SWAP_NONE, HWFMT_8_8_8_8, 1, 1), F(SWAP_NONE, HWFMT_8_8_8_8, 2, 0),
	F(SWAP_NONE, HWFMT_8_8, 0, 0), F(SWAP_NONE, HWFMT_8_8, 2, 0), F(SWAP_NONE, HWFMT_8_8, 0, 1), F(SWAP_NONE, HWFMT_8_8, 1, 0),
	F(SWAP_NONE, HWFMT_8, 0, 0), F(SWAP_NONE, HWFMT_8, 1, 0),
	F(SWAP_U16, HWFMT_16_16_16_16_FLOAT, 2, 0), F(SWAP_U16, HWFMT_16_16_16_16, 0, 1), F(SWAP_U16, HWFMT_16_16_16_16, 0, 0),
	F(SWAP_U16, HWFMT_16_16_16_16, 2, 1), F(SWAP_U16, HWFMT_16_16_16_16, 1, 1), F(SWAP_U16, HWFMT_16_16_16_16, 1, 0),
	F(SWAP_U16, HWFMT_16_16_FLOAT, 2, 0), F(SWAP_U16, HWFMT_16_16, 0, 0), F(SWAP_U16, HWFMT_16_16, 0, 1),
	F(SWAP_U16, HWFMT_16_16, 1, 0), F(SWAP_U16, HWFMT_16_16, 1, 1), F(SWAP_U16, HWFMT_16_16, 2, 0), F(SWAP_U16, HWFMT_16_16, 2, 1),
	F(SWAP_U16, HWFMT_16, 1, 0), F(SWAP_U16, HWFMT_16, 0, 0),
};
static const AttrFormat kAttrFormatsDeclined[] = {
	F(SWAP_NONE, HWFMT_16_16_16_16, 1, 0), F(SWAP_U32, HWFMT_16_16, 1, 0),
#undef F
};

// VulkanPipelineKey_VertexFormat: the integers of the format's sizes
static VkFormat VertexFormat(Latte::E_HWFMT f)
{
	using E = Latte::E_HWFMT;
	switch (f)
	{
	case E::HWFMT_32_32_32_32_FLOAT: case E::HWFMT_32_32_32_32: return VK_FORMAT_R32G32B32A32_UINT;
	case E::HWFMT_32_32_32_FLOAT: case E::HWFMT_32_32_32: return VK_FORMAT_R32G32B32_UINT;
	case E::HWFMT_32_32_FLOAT: case E::HWFMT_32_32: return VK_FORMAT_R32G32_UINT;
	case E::HWFMT_32_FLOAT: case E::HWFMT_32: case E::HWFMT_2_10_10_10: return VK_FORMAT_R32_UINT;
	case E::HWFMT_8_8_8_8: return VK_FORMAT_R8G8B8A8_UINT;
	case E::HWFMT_8_8: return VK_FORMAT_R8G8_UINT;
	case E::HWFMT_8: return VK_FORMAT_R8_UINT;
	case E::HWFMT_16_16_16_16: case E::HWFMT_16_16_16_16_FLOAT: return VK_FORMAT_R16G16B16A16_UINT;
	case E::HWFMT_16_16: case E::HWFMT_16_16_FLOAT: return VK_FORMAT_R16G16_UINT;
	case E::HWFMT_16: return VK_FORMAT_R16_UINT;
	default: return VK_FORMAT_UNDEFINED;
	}
}

static LatteFetchShader s_fetchShader;

// the registers, the pixel shader's inputs and the fetch shader of a case
static void SetVertexState(const VSCase& k)
{
	for (int i = 0; i < 32; i++)
		s_regs[mmSQ_VTX_SEMANTIC_0 + i] = k.vtxSemantic[i];
	for (int i = 0; i < 8; i++)
		s_regs[mmSPI_VS_OUT_ID_0 + i] = k.outId[i];
	LatteContextRegister& lcr = *(LatteContextRegister*)s_regs;
	lcr.PA_CL_CLIP_CNTL.set_DX_CLIP_SPACE_DEF(k.dxClip);
	lcr.PA_CL_VTE_CNTL.set_VPORT_X_SCALE_ENA(k.viewportScale);
	lcr.PA_CL_VTE_CNTL.set_VPORT_Y_SCALE_ENA(true);
	lcr.PA_CL_VTE_CNTL.set_VPORT_Z_SCALE_ENA(true);
	// the draw's primitives: a geometry shader's input (points or
	// triangles, whatever it emits), else points or triangles
	const bool drawPoints = k.gsCase ? !k.gsTrianglesIn : k.points;
	lcr.VGT_PRIMITIVE_TYPE.set_PRIMITIVE_MODE(drawPoints ? Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::POINTS :
		Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::TRIANGLES);
	s_regs[mmVGT_GS_OUT_PRIM_TYPE] = k.gsPointsOut ? 0 : 2;
	s_regs[mmVGT_STRMOUT_EN] = (k.streamout && !k.streamoutOff) ? 1 : 0;
	for (int b = 0; b < 4; b++)
		s_regs[mmVGT_STRMOUT_VTX_STRIDE_0 + b * 4] = b < 2 ? k.strmStride[b] : 0;
	s_regs[mmSQ_GS_VERT_ITEMSIZE] = (uint32)(4 * k.gsOut);
	s_regs[mmSQ_GSVS_RING_ITEMSIZE] = (uint32)(3 * 4 * k.gsOut);
	g_testPSInputTable.count = k.psCount;
	for (int i = 0; i < k.psCount; i++)
	{
		g_testPSInputTable.import[i].semanticId = k.psSemantic[i];
		g_testPSInputTable.import[i].isFlat = true;
		g_testPSInputTable.import[i].isNoPerspective = false;
	}
	SetTextureRegisters(s_regs, k.p);
	g_testPreventInfiniteLoops = k.preventLoops;
	g_testRoundToPrecision = k.roundPrecision;
	s_fetchShader.bufferGroups.clear();
	for (size_t i = 0; i < k.attribs.size(); i++)
	{
		LatteParsedFetchShaderBufferGroup g{};
		g.attributeBufferIndex = (uint8)i;
		g.attribCount = 1;
		g.attrib = const_cast<LatteParsedFetchShaderAttribute*>(&k.attribs[i]);
		s_fetchShader.bufferGroups.push_back(g);
	}
}

static Decompiled DecompileVertex(const std::vector<uint32>& program, bool direct, const VSCase& k)
{
	Decompiled d;
	LatteDecompilerOptions options;
	options.strictMul = k.strictMul;
	options.directSPIRV = direct;
	options.usesGeometryShader = k.usesGS || k.gsCase; // a geometry shader's draw, all its stages
	options.useTFViaSSBO = true; // as the Vulkan renderer's
	options.spirvInstrinsics.hasRoundingModeRTEFloat32 = k.rte;
	LatteDecompilerOutput_t out{};
	SetVertexState(k);
	if (k.gsCase)
		LatteDecompiler_DecompileGeometryShader(0x9abc, s_regs, (uint8*)program.data(), (uint32)program.size() * 4,
			(uint8*)k.copyShader.data(), (uint32)k.copyShader.size() * 4, (uint32)k.gsIn, options, &out);
	else
		LatteDecompiler_DecompileVertexShader(0x5678, s_regs, (uint8*)program.data(), (uint32)program.size() * 4, &s_fetchShader, options, &out);
	LatteDecompilerShader* shader = out.shader;
	d.offsets = out.uniformOffsetsVK;
	d.remapped = shader->list_remappedUniformEntries;
	d.uniformMode = shader->uniformMode;
	d.setIndex = out.resourceMappingVK.setIndex;
	d.binding = out.resourceMappingVK.uniformVarsBufferBindingPoint;
	for (int b = 0; b < LATTE_NUM_MAX_UNIFORM_BUFFERS; b++)
		d.ubBinding[b] = out.resourceMappingVK.uniformBuffersBindingPoint[b];
	for (int i = 0; i < LATTE_NUM_MAX_TEX_UNITS; i++)
	{
		d.texBinding[i] = out.textureUnitMask[i] ? out.resourceMappingVK.textureUnitToBindingPoint[i] : -1;
		d.shadow[i] = shader->textureUsesDepthCompare[i];
	}
	std::memcpy(d.attrLoc, out.resourceMappingVK.attributeMapping, sizeof(d.attrLoc));
	d.tfBinding = out.resourceMappingVK.tfStorageBindingPoint;
	d.declined = LatteDecompiler_SPIRVDeclineReason();
	if (shader->hasError)
	{
		delete shader;
		return d;
	}
	if (!shader->spirv.empty())
	{
		d.spirv = shader->spirv;
		d.ok = true;
	}
	else
	{
		d.glsl = shader->shaderSource;
		const std::string glsl = RslcGlsl(shader->shaderSource, k.rte);
		char err[512] = {};
		rslc_opts opts{};
		opts.dst = RSLC_DST_SPIRV;
		opts.err = err;
		opts.err_cap = sizeof(err);
		rslc_result res{};
		if (rslc_compile(glsl.data(), glsl.size(), k.gsCase ? RSLC_STAGE_GEOMETRY : RSLC_STAGE_VERTEX, &opts, &res))
		{
			d.spirv.assign(res.spirv, res.spirv + res.spirv_len);
			rslc_result_free(&res, &opts);
			d.ok = true;
		}
	}
	delete shader;
	return d;
}

// the vertex shader over the grid, the fragment shader given passing on
// what it shows
static VkShaderModule GeometryFeeder(const Context& c, const VSCase& k);

static std::vector<uint32> DrawVertex(const Context& c, const Decompiled& d, const VSCase& k, VkShaderModule fs, const uint32* uniforms)
{
	VkDescriptorSetLayoutBinding b[2 + LATTE_NUM_MAX_TEX_UNITS + LATTE_NUM_MAX_UNIFORM_BUFFERS]{};
	VkWriteDescriptorSet texWrites[LATTE_NUM_MAX_TEX_UNITS]{};
	VkDescriptorImageInfo texInfos[LATTE_NUM_MAX_TEX_UNITS]{};
	uint32 nb = 0, nTex = 0;
	// the streamout buffer, where the shader writes one
	if (d.tfBinding >= 0)
	{
		b[nb].binding = (uint32)d.tfBinding;
		b[nb].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		b[nb].descriptorCount = 1;
		b[nb].stageFlags = k.gsCase ? VK_SHADER_STAGE_GEOMETRY_BIT : VK_SHADER_STAGE_VERTEX_BIT;
		nb++;
	}
	if (d.binding >= 0)
	{
		b[nb].binding = (uint32)d.binding;
		b[nb].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		b[nb].descriptorCount = 1;
		b[nb].stageFlags = k.gsCase ? VK_SHADER_STAGE_GEOMETRY_BIT : VK_SHADER_STAGE_VERTEX_BIT;
		nb++;
	}
	for (int i = 0; i < LATTE_NUM_MAX_TEX_UNITS; i++)
	{
		if (d.texBinding[i] < 0)
			continue;
		if (i >= 4 || k.p.texKind[i] == TEX_NONE || k.p.texKind[i] >= TEX_KINDS || (d.shadow[i] && !c.depth[k.p.texKind[i]].image))
		{
			std::printf("FAIL: texture unit %d drawn with no texture\n", i);
			s_failures++;
			continue;
		}
		const Context::Texture& t = d.shadow[i] ? c.depth[k.p.texKind[i]] : c.tex[k.p.texKind[i]];
		b[nb].binding = (uint32)d.texBinding[i];
		b[nb].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		b[nb].descriptorCount = 1;
		b[nb].stageFlags = k.gsCase ? VK_SHADER_STAGE_GEOMETRY_BIT : VK_SHADER_STAGE_VERTEX_BIT;
		nb++;
		texInfos[nTex] = {t.sampler, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		texWrites[nTex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		texWrites[nTex].dstBinding = (uint32)d.texBinding[i];
		texWrites[nTex].descriptorCount = 1;
		texWrites[nTex].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		texWrites[nTex].pImageInfo = &texInfos[nTex];
		nTex++;
	}
	// a vertex shader's resources in set 0, a geometry shader's in set 2
	// (the sets before it empty, as the renderer's other stages')
	const int setIndex = k.gsCase ? 2 : 0;
	if (d.setIndex != setIndex)
	{
		std::printf("FAIL: a %s shader's resources in set %d\n", k.gsCase ? "geometry" : "vertex", d.setIndex);
		s_failures++;
	}
	VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	VkDescriptorSetLayout empty;
	VKCHECK(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &empty));
	UniformBufferBindings(d, k.gsCase ? VK_SHADER_STAGE_GEOMETRY_BIT : VK_SHADER_STAGE_VERTEX_BIT, b, nb);
	dli.bindingCount = nb;
	dli.pBindings = b;
	VkDescriptorSetLayout set;
	VKCHECK(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &set));
	VkDescriptorSetLayout setLayouts[3] = { empty, empty, empty };
	setLayouts[setIndex] = set;
	VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pli.setLayoutCount = (uint32)setIndex + 1;
	pli.pSetLayouts = setLayouts;
	VkPipelineLayout layout;
	VKCHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &layout));
	VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + LATTE_NUM_MAX_UNIFORM_BUFFERS}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, LATTE_NUM_MAX_TEX_UNITS},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
	VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	dpi.maxSets = 1;
	dpi.poolSizeCount = 3;
	dpi.pPoolSizes = ps;
	VkDescriptorPool dpool;
	VKCHECK(vkCreateDescriptorPool(c.device, &dpi, nullptr, &dpool));
	VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	dai.descriptorPool = dpool;
	dai.descriptorSetCount = 1;
	dai.pSetLayouts = &set;
	VkDescriptorSet ds;
	VKCHECK(vkAllocateDescriptorSets(c.device, &dai, &ds));
	WriteUniformBuffers(c, d, ds);
	std::memset(c.uboData, 0, 8192);
	std::memcpy(c.uboData, uniforms, 8192);
	if (d.binding >= 0)
	{
		VkDescriptorBufferInfo dbi{c.ubo, 0, 8192};
		VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		wds.dstSet = ds;
		wds.dstBinding = (uint32)d.binding;
		wds.descriptorCount = 1;
		wds.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		wds.pBufferInfo = &dbi;
		vkUpdateDescriptorSets(c.device, 1, &wds, 0, nullptr);
	}
	// the streamout buffer filled with a pattern, the writes over it
	std::memset(c.sbufData, 0xCD, Context::kStreamoutBytes);
	if (d.tfBinding >= 0)
	{
		VkDescriptorBufferInfo sbi{c.sbuf, 0, Context::kStreamoutBytes};
		VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		wds.dstSet = ds;
		wds.dstBinding = (uint32)d.tfBinding;
		wds.descriptorCount = 1;
		wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		wds.pBufferInfo = &sbi;
		vkUpdateDescriptorSets(c.device, 1, &wds, 0, nullptr);
	}
	for (uint32 i = 0; i < nTex; i++)
		texWrites[i].dstSet = ds;
	if (nTex)
		vkUpdateDescriptorSets(c.device, nTex, texWrites, 0, nullptr);
	// each attribute a binding of its own, 16 bytes a vertex
	VkVertexInputBindingDescription vb[16]{};
	VkVertexInputAttributeDescription va[16]{};
	uint32 nAttr = 0;
	for (size_t i = 0; i < k.attribs.size(); i++)
	{
		sint8 loc = d.attrLoc[k.attribs[i].semanticId];
		if (loc < 0)
			continue;
		vb[nAttr] = {(uint32)i, 16, VK_VERTEX_INPUT_RATE_VERTEX};
		va[nAttr] = {(uint32)loc, (uint32)i, VertexFormat(k.attribs[i].format), 0};
		nAttr++;
	}
	if (!k.vertexData.empty())
		std::memcpy(c.vbufData, k.vertexData.data(), k.vertexData.size());
	VkShaderModule vs = k.gsCase ? GeometryFeeder(c, k) : Module(c, d.spirv);
	VkPipelineShaderStageCreateInfo stages[3]{};
	stages[0].sType = stages[1].sType = stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vs;
	stages[0].pName = "main";
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fs;
	stages[1].pName = "main";
	VkShaderModule gs = VK_NULL_HANDLE;
	if (k.gsCase)
	{
		gs = Module(c, d.spirv);
		stages[2].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
		stages[2].module = gs;
		stages[2].pName = "main";
	}
	else if (k.usesGS)
	{
		// the ring's parameters through: the position from 0, parameter l
		// at location l from l + 1
		const int n = k.psCount + 1;
		std::string g = "#version 450\nlayout(triangles) in;\nlayout(triangle_strip, max_vertices = 3) out;\n"
			"layout(location = 0) in Vertex\n{\n";
		for (int i = 0; i < n; i++)
			g += fmt::format(" ivec4 passV2GParameter{};\n", i);
		g += "}v2g[];\n";
		for (int l = 0; l < k.psCount; l++)
			g += fmt::format("layout(location = {}) flat out vec4 p{};\n", l, l);
		g += "void main()\n{\nfor (int i = 0; i < 3; i++)\n{\ngl_Position = intBitsToFloat(v2g[i].passV2GParameter0);\n";
		for (int l = 0; l < k.psCount; l++)
			g += fmt::format("p{} = intBitsToFloat(v2g[i].passV2GParameter{});\n", l, l + 1);
		g += "EmitVertex();\n}\nEndPrimitive();\n}\n";
		std::vector<uint32> words;
		if (!CompileGlsl(g, words, false, RSLC_STAGE_GEOMETRY))
		{
			std::printf("FAIL: the test's geometry shader does not compile\n");
			s_failures++;
			std::exit(1);
		}
		gs = Module(c, words);
		stages[2].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
		stages[2].module = gs;
		stages[2].pName = "main";
	}
	VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	vin.vertexBindingDescriptionCount = nAttr;
	vin.pVertexBindingDescriptions = vb;
	vin.vertexAttributeDescriptionCount = nAttr;
	vin.pVertexAttributeDescriptions = va;
	VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	ia.topology = (k.points && !k.gsCase) || (k.gsCase && !k.gsTrianglesIn) ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkViewport vp{0, 0, (float)W, (float)H, 0, 1};
	VkRect2D sc{{0, 0}, {W, H}};
	VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	vps.viewportCount = 1;
	vps.pViewports = &vp;
	vps.scissorCount = 1;
	vps.pScissors = &sc;
	VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rs.cullMode = VK_CULL_MODE_NONE;
	// a geometry shader writing the stream writes no vertex (its outputs
	// go to the buffer instead): nothing to rasterize
	rs.rasterizerDiscardEnable = (k.gsCase && k.streamout) ? VK_TRUE : VK_FALSE;
	rs.lineWidth = 1.0f;
	VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineColorBlendAttachmentState cba{};
	cba.colorWriteMask = 0xF;
	VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	cb.attachmentCount = 1;
	cb.pAttachments = &cba;
	VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	// (no fragment stage where nothing is rasterized)
	if (rs.rasterizerDiscardEnable)
		stages[1] = stages[2];
	gpi.stageCount = rs.rasterizerDiscardEnable ? 2 : gs ? 3 : 2;
	gpi.pStages = stages;
	gpi.pVertexInputState = &vin;
	gpi.pInputAssemblyState = &ia;
	gpi.pViewportState = &vps;
	gpi.pRasterizationState = &rs;
	gpi.pMultisampleState = &ms;
	gpi.pColorBlendState = &cb;
	gpi.layout = layout;
	gpi.renderPass = c.renderPass;
	VkPipeline pipeline;
	VKCHECK(vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline));
	VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cai.commandPool = c.pool;
	cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cai.commandBufferCount = 1;
	VkCommandBuffer cmd;
	VKCHECK(vkAllocateCommandBuffers(c.device, &cai, &cmd));
	VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VKCHECK(vkBeginCommandBuffer(cmd, &cbi));
	VkClearValue clear;
	clear.color = {{-7.0f, 13.0f, -21.0f, 42.0f}};
	VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
	rbi.renderPass = c.renderPass;
	rbi.framebuffer = c.framebuffer;
	rbi.renderArea = {{0, 0}, {W, H}};
	rbi.clearValueCount = 1;
	rbi.pClearValues = &clear;
	vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	if (nb)
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, (uint32)setIndex, 1, &ds, 0, nullptr);
	for (uint32 i = 0; i < nAttr; i++)
	{
		VkDeviceSize offset = (VkDeviceSize)vb[i].binding * kVerts * 16;
		vkCmdBindVertexBuffers(cmd, vb[i].binding, 1, &c.vbuf, &offset);
	}
	vkCmdDraw(cmd, k.gsCase ? W * H * (k.gsTrianglesIn ? 3 : 1) : kVerts, 1, 0, 0);
	vkCmdEndRenderPass(cmd);
	VkBufferImageCopy region{};
	region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	region.imageExtent = {W, H, 1};
	vkCmdCopyImageToBuffer(cmd, c.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.readback, 1, &region);
	VKCHECK(vkEndCommandBuffer(cmd));
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd;
	VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	VKCHECK(vkCreateFence(c.device, &fi, nullptr, &fence));
	VKCHECK(vkQueueSubmit(c.queue, 1, &si, fence));
	if (vkWaitForFences(c.device, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000) != VK_SUCCESS)
	{
		std::printf("FAIL: a draw did not finish within 30 s (a loop that never ends?)\n");
		std::fflush(stdout);
		std::_Exit(1);
	}
	vkDestroyFence(c.device, fence, nullptr);
	std::vector<uint32> pixels(W * H * 4);
	void* map;
	VKCHECK(vkMapMemory(c.device, c.readMem, 0, pixels.size() * 4, 0, &map));
	std::memcpy(pixels.data(), map, pixels.size() * 4);
	vkUnmapMemory(c.device, c.readMem);
	// and the streamout buffer after them, compared alike
	if (d.tfBinding >= 0)
	{
		const uint32* sb = (const uint32*)c.sbufData;
		pixels.insert(pixels.end(), sb, sb + Context::kStreamoutBytes / 4);
	}
	vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
	vkDestroyPipeline(c.device, pipeline, nullptr);
	vkDestroyShaderModule(c.device, vs, nullptr);
	if (gs)
		vkDestroyShaderModule(c.device, gs, nullptr);
	vkDestroyDescriptorPool(c.device, dpool, nullptr);
	vkDestroyPipelineLayout(c.device, layout, nullptr);
	vkDestroyDescriptorSetLayout(c.device, empty, nullptr);
	vkDestroyDescriptorSetLayout(c.device, set, nullptr);
	return pixels;
}

// a register written relative to AR somewhere in the program
static bool WritesRelative(const Program& p)
{
	for (auto& clause : p.clauses)
		for (auto& g : clause)
			for (auto& op : g.ops)
				if (op.destRel)
					return true;
	return false;
}

// a geometry shader case's vertex shader: a point at each pixel's centre,
// the ring's parameters from the vertex index (the position first, at a
// depth inside the volume either way)
static VkShaderModule GeometryFeeder(const Context& c, const VSCase& k)
{
	std::string g = "#version 450\nlayout(location = 0) out Vertex\n{\n";
	for (int i = 0; i < k.gsIn; i++)
		g += fmt::format(" ivec4 passV2GParameter{};\n", i);
	g += k.gsTrianglesIn ? "}v2g;\nvoid main()\n{\nint i = gl_VertexIndex / 3;\n" : "}v2g;\nvoid main()\n{\nint i = gl_VertexIndex;\n";
	g +=
		"vec2 c = vec2(float(i % 4) + 0.5, float(i / 4) + 0.5) * 0.5 - 1.0;\n"
		"v2g.passV2GParameter0 = floatBitsToInt(vec4(c, 0.2 + 0.03 * float(i), 1.0));\n";
	for (int q = 1; q < k.gsIn; q++)
		g += fmt::format("v2g.passV2GParameter{0} = floatBitsToInt(vec4(float(i) * 0.37 + {0}.0, float(i) * -1.3 + {0}.5, {0}.25 - float(i) * 0.21, float(i * {0}) * 0.11 + float(gl_VertexIndex % 3)));\n", q);
	// a position of its own too, the corner of a small triangle (a primitive
	// the rasterizer would draw, before the geometry shader replaces it)
	g += "int r = gl_VertexIndex % 3;\nfloat dx = 0.0;\nfloat dy = 0.0;\nif (r == 1)\n\tdx = 0.25;\nif (r == 2)\n\tdy = 0.25;\ngl_Position = vec4(c.x + dx, c.y + dy, 0.5, 1.0);\n}\n";
	std::vector<uint32> words;
	if (!CompileGlsl(g, words, false, RSLC_STAGE_VERTEX))
	{
		std::printf("FAIL: the test's vertex shader for a geometry shader does not compile\n");
		std::exit(1);
	}
	return Module(c, words);
}

// a geometry shader: the ring's parameters fetched (the position into R8,
// the rest into R0 to R7), a random program, the three corners of a
// triangle over the pixel into R9 to R11, then for each a ring write of
// the corner and of each parameter (registers of the program's, some
// written twice in part) and EMIT_VERTEX; and its copy shader
// a stream write of a register to buffer b: some channels, at a dword of
// the vertex the stride leaves room for
static Program::VsExport StreamWrite(Gen& gen, uint32 b, uint8 gpr, uint32 strideDwords)
{
	Program::VsExport e{};
	e.type = 5;
	e.gpr = gpr;
	e.sel[0] = (uint8)(1 + gen.R(15));
	e.sel[1] = (uint8)b;
	e.sel[2] = 3;
	uint32 top = 0;
	for (uint32 c = 0; c < 4; c++)
		if (e.sel[0] & (1u << c))
			top = c + 1;
	e.base = (uint16)gen.R(strideDwords - top + 1);
	return e;
}

static std::vector<uint32> MakeCopyShader(int gsOut, const std::vector<Program::VsExport>& streams = {});

static VSCase MakeGeometryCase(Gen& gen, bool allowDeclined)
{
	VSCase k{};
	k.gsCase = true;
	k.viewportScale = true;
	gen.vertexMode = true;
	k.p = gen.Make(allowDeclined);
	gen.vertexMode = false;
	Program& p = k.p;
	p.vertex = true;
	p.geometry = true;
	k.gsIn = 2 + (int)gen.R(4);
	k.gsOut = 2 + (int)gen.R(3);
	k.gsPointsOut = gen.Chance(20);
	k.gsTrianglesIn = gen.Chance(30);
	k.points = k.gsPointsOut;
	k.pointSize = (float)(1 + gen.R(4));
	k.psCount = k.gsOut - 1;
	k.dxClip = gen.Chance(50);
	if (p.cf.empty())
	{
		for (size_t i = 0; i < p.clauses.size(); i++)
			p.cf.push_back({ GPU7_CF_INST_ALU, (int)i, 0 });
	}
	// the fetches, first
	std::vector<TexOp> fetch;
	for (int q = 0; q < k.gsIn; q++)
	{
		TexOp t{};
		t.vfetch = true;
		t.opcode = GPU7_TEX_INST_VFETCH;
		t.unit = 0x9F;
		t.vtxOffset = (uint16)(16 * q);
		t.dstGpr = (uint8)(q == 0 ? kPosGpr : gen.R(kGPRs));
		for (int c = 0; c < 4; c++)
			t.dstSel[c] = (uint8)c;
		if (q != 0 && gen.Chance(25))
		{
			// some channels, swizzled, some masked
			for (int c = 0; c < 4; c++)
				t.dstSel[c] = (uint8)(gen.Chance(80) ? gen.R(4) : 7);
			if (gen.Chance(40))
			{
				// one channel, of another
				for (int c = 0; c < 4; c++)
					t.dstSel[c] = 7;
				t.dstSel[gen.R(4)] = (uint8)gen.R(4);
			}
			if (t.dstSel[0] == 7 && t.dstSel[1] == 7 && t.dstSel[2] == 7 && t.dstSel[3] == 7)
				t.dstSel[0] = 0;
		}
		fetch.push_back(t);
	}
	p.texClauses.push_back(fetch);
	p.cf.insert(p.cf.begin(), CFEntry{ GPU7_CF_INST_TEX, (int)p.texClauses.size() - 1, 0 });
	// the corners: R9 to R11 = R8 + (-d,-d), (2d,-d), (-d,2d), d half a pixel
	const float d = 1.0f / (float)W;
	const float off[3][2] = { { -d, -d }, { 2 * d, -d }, { -d, 2 * d } };
	std::vector<AluGroup> corners;
	for (int v = 0; v < 3; v++)
	{
		AluGroup g{};
		for (uint8 c = 0; c < 4; c++)
		{
			AluOp op{};
			op.opcode = c < 2 ? ALU_OP2_INST_ADD : ALU_OP2_INST_MOV;
			op.src[0] = (uint32)kPosGpr;
			op.chan[0] = c;
			if (c < 2)
			{
				op.src[1] = 253;
				op.chan[1] = c;
				std::memcpy(&g.literals[c], &off[v][c], 4);
			}
			op.destGpr = (uint8)(9 + v);
			op.destElem = c;
			op.writeMask = 1;
			g.ops.push_back(op);
		}
		corners.push_back(g);
	}
	p.clauses.push_back(corners);
	p.cf.push_back({ GPU7_CF_INST_ALU, (int)p.clauses.size() - 1, 0 });
	// the vertices
	const uint32 vertDwords = 4 * (uint32)k.gsOut;
	auto ring = [&](uint32 base, uint8 gpr, uint8 mask, uint8 burst = 0) {
		Program::VsExport r{};
		r.type = 3;
		r.base = (uint16)base;
		r.gpr = gpr;
		r.sel[0] = mask;
		r.burst = burst;
		p.vsExports.push_back(r);
	};
	for (uint32 v = 0; v < 3; v++)
	{
		// the corner, now and then without its w (the default 1.0 then)
		ring(v * vertDwords, (uint8)(9 + v), gen.Chance(20) ? 0x7 : 0xF);
		for (int o = 1; o < k.gsOut; o++)
		{
			if (o + 1 < k.gsOut && gen.Chance(25))
			{
				// two parameters from two registers at once
				ring(v * vertDwords + 4 * (uint32)o, (uint8)gen.R(kGPRs - 1), 0xF, 1);
				o++;
			}
			else
				ring(v * vertDwords + 4 * (uint32)o, (uint8)gen.R(kGPRs), 0xF);
		}
		if (gen.Chance(50))
			ring(v * vertDwords + 4 * (1 + gen.R((uint32)k.gsOut - 1)), (uint8)gen.R(kGPRs), (uint8)(1 + gen.R(14)));
		Program::VsExport e{};
		e.type = 4;
		p.vsExports.push_back(e);
	}
	// streamout now and then: the copy shader's stream writes of some
	// outputs, the ring writes of those to the buffer instead
	std::vector<Program::VsExport> streams;
	k.streamout = gen.Chance(25);
	if (k.streamout)
	{
		k.strmStride[0] = 4 + gen.R(5);
		k.strmStride[1] = 4 + gen.R(5);
		for (int o = 0; o < k.gsOut; o++)
		{
			if (o > 0 && !gen.Chance(70))
				continue;
			const uint32 b = gen.R(2);
			streams.push_back(StreamWrite(gen, b, (uint8)o, k.strmStride[b]));
		}
	}
	k.copyShader = MakeCopyShader(k.gsOut, streams);
	return k;
}

// the copy shader of gsOut outputs: a VTX clause fetching each from the
// ring into R<o>, then the position's export (R0) and the parameters'
// (R1 on, parameter o - 1)
static std::vector<uint32> MakeCopyShader(int gsOut, const std::vector<Program::VsExport>& streams)
{
	VSCase k{};
	k.gsOut = gsOut;
	std::vector<uint32> cs;
	const uint32 cfCount = 1 + (uint32)streams.size() + (uint32)k.gsOut;
	const uint32 vtxAddr = (cfCount + 1) & ~1u;
	const uint32 n = (uint32)k.gsOut - 1;
	cs.push_back(vtxAddr);
	cs.push_back(((n & 7) << 10) | (((n >> 3) & 1) << 19) | ((uint32)GPU7_CF_INST_VTX << 23) | (1u << 31));
	// the stream writes of the fetched registers, before the exports
	for (auto& e : streams)
	{
		cs.push_back((uint32)e.base | ((uint32)e.gpr << 15) | (3u << 30));
		cs.push_back((uint32)e.sel[2] | ((uint32)e.sel[0] << 12) |
			((uint32)(e.sel[1] ? GPU7_CF_INST_MEM_STREAM1_WRITE : GPU7_CF_INST_MEM_STREAM0_WRITE) << 23) | (1u << 31));
	}
	for (int o = 0; o < k.gsOut; o++)
	{
		const bool last = o + 1 == k.gsOut;
		const uint32 type = o == 0 ? 1 : 2;
		const uint32 base = o == 0 ? GPU7_DECOMPILER_CF_EXPORT_BASE_POSITION : (uint32)(o - 1);
		cs.push_back(base | (type << 13) | ((uint32)o << 15));
		cs.push_back(0 | (1 << 3) | (2 << 6) | (3 << 9) | (last ? (1u << 21) : 0) |
			((uint32)(last ? GPU7_CF_INST_EXPORT_DONE : GPU7_CF_INST_EXPORT) << 23) | (1u << 31));
	}
	cs.resize(vtxAddr * 2, 0);
	for (int o = 0; o < k.gsOut; o++)
	{
		TexOp t{};
		t.vfetch = true;
		t.unit = 0x9F;
		t.vtxOffset = (uint16)(16 * o);
		t.dstGpr = (uint8)o;
		for (int c = 0; c < 4; c++)
			t.dstSel[c] = (uint8)c;
		uint32 w[4];
		EncodeTex(t, w);
		cs.insert(cs.end(), w, w + 4);
	}
	return cs;
}

// a geometry shader emitting its third vertex under a predicate (a pixel's
// x past the middle): EMIT_VERTEX in a conditional block, read under its
// own level of the active mask; the left half's strips of two vertices
// draw nothing
static VSCase MakeDirectedGeometryCase()
{
	VSCase k{};
	k.gsCase = true;
	k.viewportScale = true;
	k.noCoverage = true;
	k.gsIn = 2;
	k.gsOut = 2;
	k.psCount = 1;
	Program& p = k.p;
	p.vertex = true;
	p.geometry = true;
	for (int u = 0; u < 4; u++)
		p.texKind[u] = TEX_NONE;
	std::vector<TexOp> fetch;
	for (int q = 0; q < 2; q++)
	{
		TexOp t{};
		t.vfetch = true;
		t.opcode = GPU7_TEX_INST_VFETCH;
		t.unit = 0x9F;
		t.vtxOffset = (uint16)(16 * q);
		t.dstGpr = (uint8)(q == 0 ? kPosGpr : 1);
		for (int c = 0; c < 4; c++)
			t.dstSel[c] = (uint8)c;
		fetch.push_back(t);
	}
	p.texClauses.push_back(fetch);
	p.cf.push_back({ GPU7_CF_INST_TEX, 0, 0 });
	// the corners into R9 to R11
	const float d = 1.0f / (float)W;
	const float off[3][2] = { { -d, -d }, { 2 * d, -d }, { -d, 2 * d } };
	std::vector<AluGroup> corners;
	for (int v = 0; v < 3; v++)
	{
		AluGroup g{};
		for (uint8 c = 0; c < 4; c++)
		{
			AluOp op{};
			op.opcode = c < 2 ? ALU_OP2_INST_ADD : ALU_OP2_INST_MOV;
			op.src[0] = (uint32)kPosGpr;
			op.chan[0] = c;
			if (c < 2)
			{
				op.src[1] = 253;
				op.chan[1] = c;
				std::memcpy(&g.literals[c], &off[v][c], 4);
			}
			op.destGpr = (uint8)(9 + v);
			op.destElem = c;
			op.writeMask = 1;
			g.ops.push_back(op);
		}
		corners.push_back(g);
	}
	p.clauses.push_back(corners);
	p.cf.push_back({ GPU7_CF_INST_ALU, 0, 0 });
	// x > 0, the right half
	AluGroup predGroup{};
	AluOp pred{};
	pred.opcode = ALU_OP2_INST_PRED_SETGT;
	pred.src[0] = (uint32)kPosGpr;
	pred.src[1] = 248; // 0.0
	pred.exec = 1;
	predGroup.ops.push_back(pred);
	p.clauses.push_back({ predGroup });
	auto vertex = [&](uint8 corner) {
		Program::VsExport r{};
		r.type = 3;
		r.gpr = corner;
		r.sel[0] = 0xF;
		r.base = 0;
		p.cfRing.push_back(r);
		p.cf.push_back({ GPU7_CF_INST_MEM_RING_WRITE, (int)p.cfRing.size() - 1, 0 });
		r.gpr = 1;
		r.base = 4;
		p.cfRing.push_back(r);
		p.cf.push_back({ GPU7_CF_INST_MEM_RING_WRITE, (int)p.cfRing.size() - 1, 0 });
		p.cf.push_back({ GPU7_CF_INST_EMIT_VERTEX, 0, 0 });
	};
	vertex(9);
	vertex(10);
	p.cf.push_back({ GPU7_CF_INST_ALU_PUSH_BEFORE, 1, 0 });
	vertex(11);
	p.cf.push_back({ GPU7_CF_INST_POP, 0, 1 });
	k.copyShader = MakeCopyShader(2);
	return k;
}

static VSCase MakeVertexCase(Gen& gen, bool allowDeclined, bool forceStreamoutOff = false)
{
	VSCase k{};
	gen.vertexMode = true;
	k.p = gen.Make(allowDeclined);
	gen.vertexMode = false;
	Program& p = k.p;
	p.vertex = true;
	// distinct semantics for the attributes and the parameters
	bool used[0x80] = {};
	auto semantic = [&]() {
		uint8 v;
		do
			v = (uint8)gen.R(0x7F);
		while (used[v]);
		used[v] = true;
		return v;
	};
	for (auto& v : k.vtxSemantic)
		v = 0xFF;
	// the position: four floats into R8
	LatteParsedFetchShaderAttribute pos{};
	pos.semanticId = semantic();
	pos.format = Latte::E_HWFMT::HWFMT_32_32_32_32_FLOAT;
	pos.nfa = 2;
	pos.endianSwap = LatteConst::VertexFetchEndianMode::SWAP_NONE;
	for (int i = 0; i < 4; i++)
		pos.ds[i] = (uint8)i;
	k.attribs.push_back(pos);
	k.vtxSemantic[kPosGpr - 1] = pos.semanticId;
	// up to four more, into R1 to R7, now and then into none
	uint32 attribs = gen.R(5);
	bool locUsed[7] = {};
	for (uint32 i = 0; i < attribs; i++)
	{
		const AttrFormat& f = (allowDeclined && gen.Chance(10)) ? kAttrFormatsDeclined[gen.R(2)] :
			kAttrFormats[gen.R(sizeof(kAttrFormats) / sizeof(kAttrFormats[0]))];
		LatteParsedFetchShaderAttribute a{};
		a.attributeBufferIndex = (uint8)(i + 1);
		a.semanticId = semantic();
		a.format = f.fmt;
		a.nfa = f.nfa;
		a.isSigned = f.isSigned;
		a.endianSwap = f.endian;
		for (int c = 0; c < 4; c++)
			a.ds[c] = (uint8)(gen.Chance(80) ? gen.R(4) : 4 + gen.R(2));
		k.attribs.push_back(a);
		if (gen.Chance(90))
		{
			uint32 loc;
			do
				loc = gen.R(7);
			while (locUsed[loc]);
			locUsed[loc] = true;
			k.vtxSemantic[loc] = a.semanticId;
		}
	}
	// the parameters: index i to a semantic the pixel shader mostly takes
	uint32 params = 1 + gen.R(3);
	uint8 paramSemantic[4];
	for (auto& o : k.outId)
		o = 0xFFFFFFFF;
	k.psCount = 0;
	for (uint32 i = 0; i < params; i++)
	{
		paramSemantic[i] = semantic();
		k.outId[i / 4] = (k.outId[i / 4] & ~(0xFFu << (8 * (i % 4)))) | ((uint32)paramSemantic[i] << (8 * (i % 4)));
		if (gen.Chance(90))
			k.psSemantic[k.psCount++] = paramSemantic[i];
	}
	// the exports: the position, then the parameters, two at once now and then
	Program::VsExport e{};
	e.type = 1;
	e.base = GPU7_DECOMPILER_CF_EXPORT_BASE_POSITION;
	e.gpr = kPosGpr;
	for (int i = 0; i < 4; i++)
		e.sel[i] = (uint8)i;
	p.vsExports.push_back(e);
	// points now and then, of the render state's size or, half the time,
	// one exported from the position's register (in window space, up to
	// the target's size)
	k.points = gen.Chance(15);
	k.pointSize = (float)(1 + gen.R(4));
	if (k.points && gen.Chance(50))
	{
		Program::VsExport ps{};
		ps.type = 1;
		ps.base = GPU7_DECOMPILER_CF_EXPORT_POINT_SIZE;
		ps.gpr = kPosGpr;
		for (int c = 0; c < 4; c++)
			ps.sel[c] = (uint8)(gen.Chance(80) ? gen.R(4) : 4 + gen.R(2));
		p.vsExports.push_back(ps);
	}
	for (uint32 i = 0; i < params; i++)
	{
		Program::VsExport x{};
		x.type = 2;
		x.base = (uint16)i;
		x.gpr = (uint8)(gen.Chance(50) ? p.exportGpr : gen.R(kGPRs - 1));
		if (i + 1 < params && x.gpr < kGPRs - 2 && gen.Chance(25))
		{
			x.burst = 1;
			i++;
		}
		for (int c = 0; c < 4; c++)
		{
			uint32 r = gen.R(10);
			x.sel[c] = (uint8)(r < 7 ? gen.R(4) : (r == 7 ? 4 : (r == 8 ? 5 : 7)));
		}
		p.vsExports.push_back(x);
	}
	// with a geometry shader now and then: the position to ring parameter
	// 0, the parameters to 1 on, each written whole (an output left
	// unwritten is undefined), then now and then some channels again
	k.usesGS = !k.points && gen.Chance(20);
	if (k.usesGS)
	{
		p.vsExports.clear();
		auto ring = [&](uint32 param, uint8 gpr, uint8 mask, uint8 burst) {
			Program::VsExport r{};
			r.type = 3;
			r.base = (uint16)(param * 4);
			r.gpr = gpr;
			r.sel[0] = mask;
			r.burst = burst;
			p.vsExports.push_back(r);
		};
		ring(0, (uint8)kPosGpr, 0xF, 0);
		for (uint32 i = 0; i < params; i++)
		{
			uint8 gpr = (uint8)gen.R(kGPRs - 1);
			if (i + 1 < params && gen.Chance(25))
			{
				ring(1 + i, gpr, 0xF, 1);
				i++;
			}
			else
				ring(1 + i, gpr, 0xF, 0);
		}
		if (gen.Chance(75))
			ring(1 + gen.R(params), (uint8)gen.R(kGPRs - 1), (uint8)(1 + gen.R(14)), 0);
		k.psCount = (int)params;
	}
	// streamout now and then (not through a geometry shader): stream writes
	// before the exports, to buffers 0 and 1 of 4 to 8 dwords a vertex
	k.streamout = !k.usesGS && (gen.Chance(25) || forceStreamoutOff);
	if (k.streamout)
	{
		k.streamoutOff = gen.Chance(20) || forceStreamoutOff;
		k.strmStride[0] = 4 + gen.R(5);
		k.strmStride[1] = 4 + gen.R(5);
		std::vector<Program::VsExport> writes;
		const uint32 n = 1 + gen.R(3);
		for (uint32 i = 0; i < n; i++)
		{
			const uint32 b = gen.R(2);
			writes.push_back(StreamWrite(gen, b, (uint8)(gen.Chance(30) ? kPosGpr : gen.R(kGPRs)), k.strmStride[b]));
		}
		p.vsExports.insert(p.vsExports.begin(), writes.begin(), writes.end());
	}
	k.dxClip = gen.Chance(50);
	k.viewportScale = !gen.Chance(15);
	// the vertex data: the position's a triangle over each pixel's centre,
	// at a depth inside the volume either way; the rest random. Without
	// the viewport scale the position is in window space, the shader
	// taking it to clip space by uf_windowSpaceToClipSpaceTransform
	k.vertexData.assign(k.attribs.size() * kVerts * 16, 0);
	for (int v = 0; v < kVerts; v++)
	{
		int pixel = v / 3, corner = v % 3;
		float d = 1.0f / W; // half a pixel
		float cx = ((pixel % W) + 0.5f) * 2.0f / W - 1.0f;
		float cy = ((pixel / W) + 0.5f) * 2.0f / H - 1.0f;
		float xyzw[4] = { cx - d + (corner == 1 ? 3 * d : 0.0f), cy - d + (corner == 2 ? 3 * d : 0.0f),
			0.05f + 0.9f * (float)gen.R(1000) / 1000.0f, 1.0f };
		if (!k.viewportScale && !k.usesGS)
		{
			// (through a geometry shader the position goes to the ring as
			// it is, the test's taking it as clip space)
			xyzw[0] = (xyzw[0] + 1.0f) * (float)W / 2.0f;
			xyzw[1] = (xyzw[1] + 1.0f) * (float)H / 2.0f;
		}
		std::memcpy(k.vertexData.data() + v * 16, xyzw, 16);
	}
	for (size_t a = 1; a < k.attribs.size(); a++)
	{
		uint32* words = (uint32*)(k.vertexData.data() + a * kVerts * 16);
		for (int i = 0; i < kVerts * 4; i++)
			words[i] = gen.Literal();
	}
	return k;
}

int main(int argc, char** argv)
{
	const char* outDir = argc > 1 ? argv[1] : nullptr;
	const int count = argc > 2 ? std::atoi(argv[2]) : 400;
	FillUniformBuffers();
	Context c;
	bool device = CreateContext(c);
	if (!device)
	{
		std::printf("skip: no Vulkan device, nothing was drawn\n");
		if (getenv("VN_REQUIRE_DEVICE"))
		{
			std::printf("FAIL: VN_REQUIRE_DEVICE is set\n");
			s_failures++;
		}
	}
	SetPSInputs(false);
	s_regs[mmCB_SHADER_MASK] = 0xF;
	Gen gen;
	gen.rng.seed(argc > 3 ? (uint32)std::strtoul(argv[3], nullptr, 10) : 20261116u);
	gen.sys.seed((argc > 3 ? (uint32)std::strtoul(argv[3], nullptr, 10) : 20261116u) ^ 0x5F5F0001u);
	int direct = 0, declined = 0, drawn = 0, differing = 0, glslFailed = 0, offsetsDiffer = 0, shapeDiffers = 0;
	int gsDrawn = 0; // of the drawn, after a geometry shader
	int cbankDrawn = 0; // of the drawn, reading uniform buffers whole
	int subDrawn = 0; // of the drawn, calling a subroutine
	// of the drawn: reading gl_FragCoord, the front face, exporting depth
	int fragCoordDrawn = 0, frontFaceDrawn = 0, depthDrawn = 0;
	int roundDrawn = 0; // of the drawn, sampling with the texel edge nudge
	int pointCoordDrawn = 0; // of the drawn, as a point reading gl_PointCoord
	int memDrawn = 0; // of the drawn, with a scatter buffer read
	std::map<std::string, int> reasons;
	const std::vector<Program> directed = DirectedPrograms();
	CHECK(directed.size() == 26);
	for (int i = 0; i < count + (int)directed.size(); i++)
	{
		const bool allowDeclined = (i % 5) == 0;
		Case k{};
		k.p = i < (int)directed.size() ? directed[i] : gen.Make(allowDeclined);
		Program& p = k.p;
		std::vector<uint32> words = Assemble(p);
		// alpha test now and then
		Latte::LATTE_SX_ALPHA_TEST_CONTROL alpha{};
		// a directed program's pixels all drawn; no alpha test after a loop
		bool hasLoop = false;
		for (auto& e : p.cf)
			hasLoop |= e.type == GPU7_CF_INST_LOOP_START_DX10 || e.type == GPU7_CF_INST_LOOP_START_NO_AL;
		if (gen.Chance(30) && i >= (int)directed.size() && !hasLoop)
		{
			alpha.set_ALPHA_TEST_ENABLE(true);
			alpha.set_ALPHA_FUNC((Latte::E_COMPAREFUNC)gen.R(8));
		}
		k.alphaRaw = alpha.getRawValue();
		s_regs[Latte::REGADDR::SX_ALPHA_TEST_CONTROL] = k.alphaRaw;
		const bool strictMul = k.strictMul = gen.Chance(50);
		const bool rte = k.rte = c.rte && gen.Chance(50);
		g_testPreventInfiniteLoops = k.preventLoops = gen.Chance(50);
		g_testRoundToPrecision = k.roundPrecision = i < (int)directed.size() ? p.roundPrecision : gen.SysChance(30);
		// now and then after a geometry shader: the same inputs, by semantic
		k.usesGS = gen.Chance(20);
		// gl_FragCoord as an input, the front face into a register (R0,
		// whose use the GLSL emitter requires), a back face drawn, now and
		// then; not for directed programs, whose registers are their own
		const bool own = i >= (int)directed.size();
		k.fragCoord = own ? gen.SysChance(25) : p.sysInputs;
		k.psControl1 = own && gen.SysChance(25) ? (1u << 8) | (gen.SysR(4) << 9) : 0;
		k.cw = gen.SysChance(50);
		for (int f = 0; f < 4; f++)
			k.fragScale[f] = (float)(1 + gen.SysR(16)) / 4.0f;
		// one point over the target now and then (where the device draws
		// points of its size), param gen's gl_PointCoord into a register
		// (the exported one, or an input's) on most of those
		k.points = own && gen.SysChance(20) && (!device || c.largePoints);
		k.paramGenGpr = k.points && gen.SysChance(75) ? (sint32)(gen.SysChance(50) ? p.exportGpr : gen.SysR(kGPRs)) : -1;
		if (p.pointCoordGpr >= 0)
		{
			k.points = !device || c.largePoints;
			k.paramGenGpr = p.pointCoordGpr;
		}
		if (p.sysInputs)
		{
			k.usesGS = false;
			k.psControl1 = (1u << 8) | (1u << 9);
			k.fragScale[0] = 2.0f;
			k.fragScale[1] = 4.0f;
		}
		SetPSInputs(k.usesGS, k.fragCoord, k.psControl1, k.paramGenGpr);
		SetTextureRegisters(s_regs, p);
		Decompiled viaGlsl = Decompile(words, false, strictMul, rte);
		Decompiled viaSpirv = Decompile(words, true, strictMul, rte);
		if (viaSpirv.declined)
		{
			declined++;
			reasons[viaSpirv.declined]++;
			// the analyzer refuses what the inlining can't do: a call the
			// emitter can't make is a fault
			if (std::strstr(viaSpirv.declined, "subroutine"))
			{
				std::printf("FAIL: program %d's subroutine declined (%s)\n", i, viaSpirv.declined);
				s_failures++;
			}
			// the GLSL way, as before: the same module
			if (viaSpirv.spirv != viaGlsl.spirv)
			{
				std::printf("FAIL: program %d declined (%s) but not compiled as before\n", i, viaSpirv.declined);
				if (outDir)
				{
					char path[512];
					std::snprintf(path, sizeof(path), "%s/declined_a.glsl", outDir);
					FILE* f = std::fopen(path, "wb");
					std::fwrite(viaGlsl.glsl.data(), 1, viaGlsl.glsl.size(), f);
					std::fclose(f);
					std::snprintf(path, sizeof(path), "%s/declined_b.glsl", outDir);
					f = std::fopen(path, "wb");
					std::fwrite(viaSpirv.glsl.data(), 1, viaSpirv.glsl.size(), f);
					std::fclose(f);
				}
				s_failures++;
			}
			continue;
		}
		if (!viaGlsl.ok)
		{
			// what the GLSL emitter writes but rslc refuses: the direct
			// emitter must have declined it
			glslFailed++;
			std::printf("FAIL: program %d: written directly where the GLSL does not compile\n", i);
			s_failures++;
			continue;
		}
		direct++;
		CHECK(viaSpirv.ok);
		{
			const std::string diff = ShapeDiff(Shape(viaGlsl.spirv), Shape(viaSpirv.spirv));
			if (!diff.empty())
			{
				shapeDiffers++;
				if (shapeDiffers <= 3)
				{
					std::printf("FAIL: program %d is not shaped as rslc's:%s\n", i, diff.c_str());
					if (outDir)
					{
						char path[512];
						std::snprintf(path, sizeof(path), "%s/shape_glsl_%d.spv", outDir, i);
						FILE* f = std::fopen(path, "wb");
						std::fwrite(viaGlsl.spirv.data(), 4, viaGlsl.spirv.size(), f);
						std::fclose(f);
						std::snprintf(path, sizeof(path), "%s/shape_direct_%d.spv", outDir, i);
						f = std::fopen(path, "wb");
						std::fwrite(viaSpirv.spirv.data(), 4, viaSpirv.spirv.size(), f);
						std::fclose(f);
						std::snprintf(path, sizeof(path), "%s/shape_%d.glsl", outDir, i);
						f = std::fopen(path, "wb");
						std::fwrite(viaGlsl.glsl.data(), 1, viaGlsl.glsl.size(), f);
						std::fclose(f);
					}
				}
			}
		}
		if (!SameOffsets(viaGlsl.offsets, viaSpirv.offsets) || viaGlsl.binding != viaSpirv.binding || viaGlsl.setIndex != viaSpirv.setIndex ||
			std::memcmp(viaGlsl.ubBinding, viaSpirv.ubBinding, sizeof(viaGlsl.ubBinding)) != 0)
		{
			offsetsDiffer++;
			std::printf("FAIL: program %d: uniform layout differs\n", i);
			s_failures++;
			continue;
		}
		if (outDir)
		{
			char path[512];
			std::snprintf(path, sizeof(path), "%s/direct_%04d.spv", outDir, i);
			FILE* f = std::fopen(path, "wb");
			if (f)
			{
				std::fwrite(viaSpirv.spirv.data(), 4, viaSpirv.spirv.size(), f);
				std::fclose(f);
			}
		}
		if (!device)
			continue;
		// uniforms: the register file the remapped array is filled from,
		// and the alpha test reference; the inputs
		for (auto& u : k.cfile)
			u = gen.Literal();
		k.alphaRef = (float)((int)gen.R(200) - 100) / 37.0f;
		for (auto& sc : k.texScale)
		{
			sc[0] = (float)(1 + gen.R(32)) / 8.0f;
			sc[1] = (float)(1 + gen.R(32)) / 8.0f;
			if (i < (int)directed.size())
			{
				// unit 0's 1.0, the others' x 1.0 and y 0.25
				sc[0] = 1.0f;
				sc[1] = &sc - k.texScale == 0 ? 1.0f : 0.25f;
			}
		}
		for (auto& f : k.inputs)
		{
			uint32 b = gen.Literal();
			std::memcpy(&f, &b, 4);
			if (!std::isfinite(f) && gen.Chance(70))
				f = (float)((int)gen.R(2000) - 1000) / 64.0f;
		}
		if (!p.inputs.empty())
			std::memcpy(k.inputs, p.inputs.data(), sizeof(k.inputs));
		drawn++;
		gsDrawn += k.usesGS;
		cbankDrawn += viaGlsl.uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CBANK;
		subDrawn += !p.sub.empty();
		fragCoordDrawn += viaGlsl.offsets.offset_fragCoordScale >= 0;
		frontFaceDrawn += k.psControl1 != 0;
		depthDrawn += p.depthGpr >= 0;
		pointCoordDrawn += k.paramGenGpr >= 0;
		bool mem = false;
		for (auto& clause : p.texClauses)
			for (auto& t : clause)
				mem |= t.memFormat != 0;
		memDrawn += mem;
		roundDrawn += viaGlsl.glsl.find("/512.0") != std::string::npos;
		if (Differs(c, k) == 1)
		{
			differing++;
			std::printf("FAIL: program %d draws differently\n", i);
			if (differing == 1 && getenv("DECOMPILER_MINIMIZE"))
				Minimize(c, k, outDir);
		}
	}
	std::printf("decompiler: %d random pixel shaders, %d written as SPIR-V directly (%d drawn both ways, %d after a geometry shader, %d differing), %d declined\n",
		count, direct, drawn, gsDrawn, differing, declined);
	// pixel shaders after a geometry shader are drawn, not declined; and
	// pixel shaders reading uniform buffers whole
	CHECK(!device || count < 100 || gsDrawn * 10 > drawn);
	std::printf("decompiler: %d pixel shaders drawn reading uniform buffers whole\n", cbankDrawn);
	CHECK(!device || count < 100 || cbankDrawn * 20 > drawn);
	std::printf("decompiler: %d pixel shaders drawn calling a subroutine\n", subDrawn);
	CHECK(!device || count < 100 || subDrawn * 30 > drawn);
	std::printf("decompiler: %d pixel shaders drawn reading gl_FragCoord, %d the front face, %d exporting depth\n", fragCoordDrawn,
		frontFaceDrawn, depthDrawn);
	CHECK(!device || count < 100 || (fragCoordDrawn * 30 > drawn && frontFaceDrawn * 30 > drawn && depthDrawn * 30 > drawn));
	std::printf("decompiler: %d pixel shaders drawn sampling with the texel edge nudge\n", roundDrawn);
	CHECK(!device || count < 100 || roundDrawn * 30 > drawn);
	std::printf("decompiler: %d pixel shaders drawn as points reading gl_PointCoord\n", pointCoordDrawn);
	CHECK(!device || !c.largePoints || count < 100 || pointCoordDrawn * 30 > drawn);
	std::printf("decompiler: %d pixel shaders drawn with a scatter buffer read\n", memDrawn);
	CHECK(!device || count < 100 || memDrawn * 60 > drawn);
	for (auto& it : reasons)
		std::printf("  declined: %s (%d)\n", it.first.c_str(), it.second);
	if (differing)
		s_failures++;
	if (shapeDiffers)
	{
		std::printf("FAIL: %d direct modules not shaped as rslc's\n", shapeDiffers);
		s_failures++;
	}
	// most of what is generated is covered
	CHECK(direct * 2 > count);
	// vertex shaders, half as many
	{
		const int vsCount = count / 2;
		int vDirect = 0, vDeclined = 0, vDrawn = 0, vDiffering = 0, vShape = 0;
		int vGsDrawn = 0; // of the drawn, writing the ring to a geometry shader
		int gsCaseDrawn = 0; // of the drawn, geometry shaders
		int soDrawn = 0; // of the drawn, writing the streamout buffer
		int vCbankDrawn = 0; // of the drawn, reading uniform buffers whole
		int vSubDrawn = 0; // of the drawn, calling a subroutine
		int vRoundDrawn = 0; // of the drawn, sampling with the texel edge nudge
		int soOffDrawn = 0; // of those, with streamout off (the writes dropped)
		std::map<std::string, int> vReasons;
		{
			// a copy shader the parser does not take (an ALU clause): the
			// geometry shader an error both ways, not a read of no copy shader
			VSCase k = MakeDirectedGeometryCase();
			k.copyShader = { 0, (8u << 26) | (1u << 21) | (1u << 31) };
			std::vector<uint32> words = Assemble(k.p);
			CHECK(!DecompileVertex(words, false, k).ok);
			CHECK(!DecompileVertex(words, true, k).ok);
		}
		for (int i = 0; i < vsCount; i++)
		{
			const bool allowDeclined = (i % 5) == 0;
			// every fourth a geometry shader's
			// every fourth a geometry shader's; every sixteenth else stream
			// writes with streamout off
			VSCase k = i == 3 ? MakeDirectedGeometryCase() : (i % 4) == 3 ? MakeGeometryCase(gen, allowDeclined) :
				MakeVertexCase(gen, allowDeclined, (i % 16) == 6);
			k.strictMul = gen.Chance(50);
			k.rte = c.rte && gen.Chance(50);
			k.preventLoops = gen.Chance(50);
			k.roundPrecision = gen.SysChance(30);
			std::vector<uint32> words = Assemble(k.p);
			Decompiled viaGlsl = DecompileVertex(words, false, k);
			Decompiled viaSpirv = DecompileVertex(words, true, k);
			if (viaSpirv.declined)
			{
				vDeclined++;
				vReasons[viaSpirv.declined]++;
				if (std::strstr(viaSpirv.declined, "subroutine"))
				{
					std::printf("FAIL: vertex program %d's subroutine declined (%s)\n", i, viaSpirv.declined);
					s_failures++;
				}
				if (viaSpirv.spirv != viaGlsl.spirv)
				{
					std::printf("FAIL: vertex program %d declined (%s) but not compiled as before\n", i, viaSpirv.declined);
					s_failures++;
				}
				continue;
			}
			if (!viaGlsl.ok)
			{
				std::printf("FAIL: vertex program %d: written directly where the GLSL does not compile\n", i);
				s_failures++;
				continue;
			}
			vDirect++;
			CHECK(viaSpirv.ok);
			const std::string diff = ShapeDiff(Shape(viaGlsl.spirv), Shape(viaSpirv.spirv));
			if (!diff.empty())
			{
				vShape++;
				if (vShape <= 3)
				{
					std::printf("FAIL: vertex program %d is not shaped as rslc's:%s\n", i, diff.c_str());
					if (outDir)
					{
						char path[512];
						std::snprintf(path, sizeof(path), "%s/vshape_glsl_%d.spv", outDir, i);
						FILE* f = std::fopen(path, "wb");
						std::fwrite(viaGlsl.spirv.data(), 4, viaGlsl.spirv.size(), f);
						std::fclose(f);
						std::snprintf(path, sizeof(path), "%s/vshape_direct_%d.spv", outDir, i);
						f = std::fopen(path, "wb");
						std::fwrite(viaSpirv.spirv.data(), 4, viaSpirv.spirv.size(), f);
						std::fclose(f);
						std::snprintf(path, sizeof(path), "%s/vshape_%d.glsl", outDir, i);
						f = std::fopen(path, "wb");
						std::fwrite(viaGlsl.glsl.data(), 1, viaGlsl.glsl.size(), f);
						std::fclose(f);
					}
				}
			}
			if (!SameOffsets(viaGlsl.offsets, viaSpirv.offsets) || viaGlsl.binding != viaSpirv.binding || viaGlsl.setIndex != viaSpirv.setIndex ||
				viaGlsl.tfBinding != viaSpirv.tfBinding || std::memcmp(viaGlsl.ubBinding, viaSpirv.ubBinding, sizeof(viaGlsl.ubBinding)) != 0 ||
				std::memcmp(viaGlsl.attrLoc, viaSpirv.attrLoc, sizeof(viaGlsl.attrLoc)) != 0)
			{
				std::printf("FAIL: vertex program %d: resource layout differs\n", i);
				s_failures++;
				continue;
			}
			if (outDir)
			{
				char path[512];
				std::snprintf(path, sizeof(path), "%s/vdirect_%04d.spv", outDir, i);
				FILE* f = std::fopen(path, "wb");
				if (f)
				{
					std::fwrite(viaSpirv.spirv.data(), 4, viaSpirv.spirv.size(), f);
					std::fclose(f);
				}
			}
			if (!device)
				continue;
			for (auto& u : k.cfile)
				u = gen.Literal();
			uint32 uniforms[2048] = {}; // a full register file (4 KiB) and what follows it
			if (viaGlsl.uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_REMAPPED)
			{
				for (auto& e : viaGlsl.remapped)
					std::memcpy(uniforms + viaGlsl.offsets.offset_remapped / 4 + e.mappedIndex * 4, RemappedSource(e, k.cfile), 16);
			}
			else if (viaGlsl.uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CFILE)
				std::memcpy(uniforms + viaGlsl.offsets.offset_uniformRegister / 4, k.cfile, viaGlsl.offsets.count_uniformRegister * 16);
			if (viaGlsl.offsets.offset_windowSpaceToClipSpaceTransform >= 0)
			{
				const float transform[2] = { 2.0f / (float)W, 2.0f / (float)H };
				std::memcpy(uniforms + viaGlsl.offsets.offset_windowSpaceToClipSpaceTransform / 4, transform, 8);
			}
			// uf_verticesPerInstance and the streamout bases (buffer b at
			// b * 16 KiB)
			if (viaGlsl.offsets.offset_verticesPerInstance >= 0)
			{
				const sint32 perInstance = k.gsCase ? (sint32)(W * H) : kVerts;
				std::memcpy(uniforms + viaGlsl.offsets.offset_verticesPerInstance / 4, &perInstance, 4);
			}
			for (int b = 0; b < LATTE_NUM_STREAMOUT_BUFFER; b++)
			{
				if (viaGlsl.offsets.offset_streamoutBufferBase[b] >= 0)
				{
					const sint32 base = b * 16384;
					std::memcpy(uniforms + viaGlsl.offsets.offset_streamoutBufferBase[b] / 4, &base, 4);
				}
			}
			if (viaGlsl.offsets.offset_pointSize >= 0)
				std::memcpy(uniforms + viaGlsl.offsets.offset_pointSize / 4, &k.pointSize, 4);
			if ((k.points && !c.largePoints) || ((k.usesGS || k.gsCase) && !c.geometryShader) || (k.gsPointsOut && !c.geometryPointSize) ||
				(k.streamout && !c.vertexStores))
				continue;
			vDrawn++;
			vGsDrawn += k.usesGS;
			gsCaseDrawn += k.gsCase;
			soDrawn += k.streamout;
			vCbankDrawn += viaGlsl.uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CBANK;
			vSubDrawn += !k.p.sub.empty();
			vRoundDrawn += viaGlsl.glsl.find("/512.0") != std::string::npos;
			soOffDrawn += k.streamout && k.streamoutOff;
			bool same = true;
			for (int l = -1; l < k.psCount; l++)
			{
				VkShaderModule fs = l < 0 ? c.fsDepth : c.fsParam[l];
				std::vector<uint32> a = DrawVertex(c, viaGlsl, k, fs, uniforms);
				std::vector<uint32> b = DrawVertex(c, viaSpirv, k, fs, uniforms);
				if (k.streamout && !k.streamoutOff && l < 0)
				{
					// the stream written: the buffer drawn back, not all the pattern
					bool written = false;
					for (size_t q = W * H * 4; q < a.size(); q++)
						written |= a[q] != 0xCDCDCDCDu;
					if (a.size() <= W * H * 4 || !written)
					{
						std::printf("FAIL: vertex program %d wrote no stream\n", i);
						s_failures++;
					}
				}
				if (l < 0 && !k.points && !k.noCoverage && !(k.gsCase && k.streamout) && !WritesRelative(k.p))
				{
					// every pixel's triangle drawn: the depth there, not the clear
					// (but for points, and where a write relative to AR may land
					// on the position's register)
					int covered = 0;
					const float clearX = -7.0f;
					uint32 clearBits;
					std::memcpy(&clearBits, &clearX, 4);
					for (uint32 q = 0; q < W * H; q++)
						covered += a[q * 4] != clearBits;
					if (covered != (int)(W * H))
					{
						std::printf("FAIL: vertex program %d covers %d of %u pixels\n", i, covered, W * H);
						s_failures++;
					}
				}
				if (!SamePixels(a, b))
				{
					if (same && vDiffering < 3)
					{
						for (size_t q = 0; q < a.size(); q++)
							if (a[q] != b[q] && !(IsNaN(a[q]) && IsNaN(b[q])))
								std::printf("  %s pixel %zu channel %zu: %08x via GLSL, %08x directly\n", l < 0 ? "depth" : "parameter", q / 4, q % 4, a[q], b[q]);
						if (outDir)
						{
							char path[512];
							std::snprintf(path, sizeof(path), "%s/vdiff_%d.glsl", outDir, i);
							FILE* f = std::fopen(path, "wb");
							std::string g = DecompileVertex(words, false, k).glsl;
							std::fwrite(g.data(), 1, g.size(), f);
							std::fclose(f);
						}
					}
					same = false;
				}
			}
			if (!same)
			{
				vDiffering++;
				std::printf("FAIL: vertex program %d draws differently\n", i);
			}
		}
		std::printf("decompiler: %d random vertex shaders, %d written as SPIR-V directly (%d drawn both ways, %d differing), %d declined\n",
			vsCount, vDirect, vDrawn, vDiffering, vDeclined);
		for (auto& it : vReasons)
			std::printf("  declined: %s (%d)\n", it.first.c_str(), it.second);
		if (vDiffering)
			s_failures++;
		if (vShape)
		{
			std::printf("FAIL: %d direct vertex modules not shaped as rslc's\n", vShape);
			s_failures++;
		}
		CHECK(vsCount == 0 || vDirect * 2 > vsCount);
		// vertex shaders writing the ring are drawn, not declined
		std::printf("decompiler: %d vertex shaders drawn writing the ring, %d geometry shaders drawn, %d writing the stream\n", vGsDrawn, gsCaseDrawn, soDrawn);
		CHECK(!device || !c.vertexStores || vsCount < 100 || soDrawn * 10 > vDrawn);
		std::printf("decompiler: %d vertex and geometry shaders drawn reading uniform buffers whole\n", vCbankDrawn);
		CHECK(!device || vsCount < 100 || vCbankDrawn * 20 > vDrawn);
		std::printf("decompiler: %d vertex and geometry shaders drawn calling a subroutine\n", vSubDrawn);
		CHECK(!device || vsCount < 100 || vSubDrawn * 30 > vDrawn);
		std::printf("decompiler: %d vertex and geometry shaders drawn sampling with the texel edge nudge\n", vRoundDrawn);
		CHECK(!device || vsCount < 100 || vRoundDrawn * 30 > vDrawn);
		// stream writes with streamout off are drawn (dropped), not declined
		CHECK(!device || !c.vertexStores || vsCount < 100 || soOffDrawn > 0);
		// geometry shaders are drawn, not declined
		CHECK(!device || !c.geometryShader || vsCount < 100 || gsCaseDrawn * 8 > vDrawn);
		CHECK(!device || !c.geometryShader || vsCount < 100 || vGsDrawn * 20 > vDrawn);
	}
	(void)glslFailed;
	(void)offsetsDiffer;
	DestroyContext(c);
	if (s_failures)
	{
		std::printf("decompiler_test: %d failure(s)\n", s_failures);
		return 1;
	}
	std::printf("decompiler_test: all passed\n");
	return 0;
}
