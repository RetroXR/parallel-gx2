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
	uint8 abs[2];
	uint8 destGpr, destElem, writeMask, clamp, omod;
};

struct AluGroup
{
	std::vector<AluOp> ops;
	uint32 literals[4];
};

struct Program
{
	std::vector<std::vector<AluGroup>> clauses; // ALU clauses, in order
	uint32 exportGpr;
	uint8 exportSel[4];
};

static void EncodeAlu(const AluOp& op, bool last, uint32& w0, uint32& w1)
{
	w0 = (op.src[0] & 0x1FF) | ((uint32)op.chan[0] << 10) | ((uint32)op.neg[0] << 12) |
		((op.src[1] & 0x1FF) << 13) | ((uint32)op.chan[1] << 23) | ((uint32)op.neg[1] << 25) | (last ? 0x80000000u : 0);
	if (op.op3)
	{
		w1 = (op.src[2] & 0x1FF) | ((uint32)op.chan[2] << 10) | ((uint32)op.neg[2] << 12) | (op.opcode << 13) |
			((uint32)op.destGpr << 21) | ((uint32)op.destElem << 29) | ((uint32)op.clamp << 31);
	}
	else
	{
		w1 = (uint32)op.abs[0] | ((uint32)op.abs[1] << 1) | ((uint32)op.writeMask << 4) | ((uint32)op.omod << 5) |
			(op.opcode << 7) | ((uint32)op.destGpr << 21) | ((uint32)op.destElem << 29) | ((uint32)op.clamp << 31);
	}
}

// CF list, then the clauses' words (64 bit units)
static std::vector<uint32> Assemble(const Program& p)
{
	std::vector<uint32> alu;
	std::vector<uint32> cf;
	const uint32 cfCount = (uint32)p.clauses.size() + 1;
	uint32 aluBase = (cfCount + 1) & ~1u; // in 64 bit words, after the CF list
	uint32 at = aluBase;
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
		// CF_ALU: addr, count-1
		cf.push_back(start);
		cf.push_back(((words - 1) << 18) | (8u << 26) | (1u << 31));
	}
	// EXPORT_DONE, pixel, color 0, end of program
	cf.push_back(0 | (0 << 13) | (p.exportGpr << 15));
	cf.push_back((uint32)p.exportSel[0] | ((uint32)p.exportSel[1] << 3) | ((uint32)p.exportSel[2] << 6) | ((uint32)p.exportSel[3] << 9) |
		(1u << 21) | ((uint32)GPU7_CF_INST_EXPORT_DONE << 23) | (1u << 31));
	std::vector<uint32> out(aluBase * 2, 0);
	std::memcpy(out.data(), cf.data(), cf.size() * 4);
	out.insert(out.end(), alu.begin(), alu.end());
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
// what the emitter declines, now and then
static const uint32 kDeclined[] = { ALU_OP2_INST_MOVA_INT, ALU_OP2_INST_MOVA_FLOOR, ALU_OP2_INST_PRED_SETE };

static const int kGPRs = 8;
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
	uint32 R(uint32 n) { return rng() % n; }
	bool Chance(uint32 percent) { return R(100) < percent; }

	// a source: what was written in the previous group (PV/PS), a GPR, a
	// constant, a literal or a uniform
	void Source(AluOp& op, int s, const bool pvWritten[5], bool firstGroup)
	{
		uint32 kind = R(100);
		op.chan[s] = (uint8)R(4);
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
			op.src[s] = 256 + R(kUniforms);
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
		// a DOT4 now and then: four ops, x to w
		if (Chance(8))
		{
			uint32 opc = Chance(50) ? ALU_OP2_INST_DOT4 : ALU_OP2_INST_DOT4_IEEE;
			uint8 dst = (uint8)R(kGPRs);
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
			if (allowDeclined && kind < 2)
				op.opcode = kDeclined[R(3)];
			else if (kind < 25)
			{
				op.op3 = true;
				op.opcode = kOp3s[R(sizeof(kOp3s) / 4)];
			}
			else
				op.opcode = kVectorOps[R(sizeof(kVectorOps) / 4)];
			for (int s = 0; s < (op.op3 ? 3 : 2); s++)
				op.neg[s] = Chance(12);
			if (!op.op3)
			{
				op.abs[0] = Chance(10);
				op.abs[1] = Chance(10);
			}
			for (int s = 0; s < (op.op3 ? 3 : 2); s++)
				Source(op, s, pvWritten, firstGroup);
			op.destGpr = (uint8)R(kGPRs);
			op.destElem = e;
			op.writeMask = op.op3 ? 1 : (Chance(75) ? 1 : 0);
			op.clamp = Chance(10);
			op.omod = (!op.op3 && Chance(10)) ? (uint8)(1 + R(3)) : 0;
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
			op.destGpr = (uint8)R(kGPRs);
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

	Program Make(bool allowDeclined)
	{
		Program p{};
		uint32 clauses = 1 + R(3);
		for (uint32 c = 0; c < clauses; c++)
		{
			bool pv[5] = {};
			std::vector<AluGroup> groups;
			uint32 n = 1 + R(8);
			for (uint32 i = 0; i < n; i++)
				groups.push_back(Group(i == 0, pv, allowDeclined));
			p.clauses.push_back(std::move(groups));
		}
		p.exportGpr = R(kGPRs - 1);
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
	std::string glsl;
};

static bool CompileGlsl(const std::string& source, std::vector<uint32>& out, bool rte)
{
	const std::string glsl = RslcGlsl(source, rte);
	char err[512] = {};
	rslc_opts opts{};
	opts.dst = RSLC_DST_SPIRV;
	opts.err = err;
	opts.err_cap = sizeof(err);
	rslc_result res{};
	if (!rslc_compile(glsl.data(), glsl.size(), RSLC_STAGE_FRAGMENT, &opts, &res))
		return false;
	out.assign(res.spirv, res.spirv + res.spirv_len);
	rslc_result_free(&res, &opts);
	return true;
}

static Decompiled Decompile(const std::vector<uint32>& program, bool direct, bool strictMul, bool rte)
{
	Decompiled d;
	LatteDecompilerOptions options;
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
	VkImageView colorView = VK_NULL_HANDLE;
	VkFramebuffer framebuffer = VK_NULL_HANDLE;
	VkBuffer readback = VK_NULL_HANDLE, ubo = VK_NULL_HANDLE;
	uint8* uboData = nullptr;
	VkShaderModule vs = VK_NULL_HANDLE;
	bool rte = false;
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
	gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
	// inputs that vary over the target, and some that stay put
	p0 = pc.v[0] * pos.x + pc.v[1];
	p1 = pc.v[1] * pos.y - pc.v[2];
	p2 = pc.v[2];
	p3 = pc.v[3] * (pos.x - pos.y);
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
	bool hasFloatControls = false;
	for (auto& e : exts)
		hasFloatControls |= !strcmp(e.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
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
	const char* devExts[] = {VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME};
	VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qi;
	dci.enabledExtensionCount = c.rte ? 1 : 0;
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

	VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bi.size = W * H * 16;
	bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &c.readback));
	VkMemoryRequirements breq;
	vkGetBufferMemoryRequirements(c.device, c.readback, &breq);
	c.readMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, c.readback, c.readMem, 0));
	bi.size = 4096;
	bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	VKCHECK(vkCreateBuffer(c.device, &bi, nullptr, &c.ubo));
	vkGetBufferMemoryRequirements(c.device, c.ubo, &breq);
	c.uboMem = Bind(c, breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VKCHECK(vkBindBufferMemory(c.device, c.ubo, c.uboMem, 0));
	void* map;
	VKCHECK(vkMapMemory(c.device, c.uboMem, 0, 4096, 0, &map));
	c.uboData = (uint8*)map;

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
	return true;
}

static void DestroyContext(Context& c)
{
	if (!c.device)
		return;
	vkDeviceWaitIdle(c.device);
	vkDestroyShaderModule(c.device, c.vs, nullptr);
	vkUnmapMemory(c.device, c.uboMem);
	vkDestroyBuffer(c.device, c.ubo, nullptr);
	vkDestroyBuffer(c.device, c.readback, nullptr);
	vkFreeMemory(c.device, c.uboMem, nullptr);
	vkFreeMemory(c.device, c.readMem, nullptr);
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
static std::vector<uint32> Draw(const Context& c, const Decompiled& d, const float inputs[16], const uint32* uniforms, size_t uniformBytes)
{
	// the uniform block at the shader's set and binding (set 0 empty, as the
	// renderer's vertex set)
	VkDescriptorSetLayoutBinding b{};
	b.binding = (uint32)std::max(d.binding, (sint32)0);
	b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	b.descriptorCount = 1;
	b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	VkDescriptorSetLayout empty, set;
	VKCHECK(vkCreateDescriptorSetLayout(c.device, &dli, nullptr, &empty));
	dli.bindingCount = d.binding >= 0 ? 1 : 0;
	dli.pBindings = &b;
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
	VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
	VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	dpi.maxSets = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes = &ps;
	VkDescriptorPool dpool;
	VKCHECK(vkCreateDescriptorPool(c.device, &dpi, nullptr, &dpool));
	VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	dai.descriptorPool = dpool;
	dai.descriptorSetCount = 1;
	dai.pSetLayouts = &set;
	VkDescriptorSet ds;
	VKCHECK(vkAllocateDescriptorSets(c.device, &dai, &ds));
	std::memset(c.uboData, 0, 4096);
	std::memcpy(c.uboData, uniforms, uniformBytes);
	if (d.binding >= 0)
	{
		VkDescriptorBufferInfo dbi{c.ubo, 0, 4096};
		VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		wds.dstSet = ds;
		wds.dstBinding = (uint32)d.binding;
		wds.descriptorCount = 1;
		wds.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		wds.pBufferInfo = &dbi;
		vkUpdateDescriptorSets(c.device, 1, &wds, 0, nullptr);
	}

	VkShaderModule fs = Module(c, d.spirv);
	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = c.vs;
	stages[0].pName = "main";
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fs;
	stages[1].pName = "main";
	VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkViewport vp{0, 0, (float)W, (float)H, 0, 1};
	VkRect2D sc{{0, 0}, {W, H}};
	VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	vps.viewportCount = 1;
	vps.pViewports = &vp;
	vps.scissorCount = 1;
	vps.pScissors = &sc;
	VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rs.cullMode = VK_CULL_MODE_NONE;
	rs.lineWidth = 1.0f;
	VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineColorBlendAttachmentState cba{};
	cba.colorWriteMask = 0xF;
	VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	cb.attachmentCount = 1;
	cb.pAttachments = &cba;
	VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	gpi.stageCount = 2;
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
	clear.color = {{-7.0f, 13.0f, -21.0f, 42.0f}}; // a discarded pixel keeps it
	VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
	rbi.renderPass = c.renderPass;
	rbi.framebuffer = c.framebuffer;
	rbi.renderArea = {{0, 0}, {W, H}};
	rbi.clearValueCount = 1;
	rbi.pClearValues = &clear;
	vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	if (d.binding >= 0)
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &ds, 0, nullptr);
	vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, inputs);
	vkCmdDraw(cmd, 3, 1, 0, 0);
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
	VKCHECK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
	vkDestroyFence(c.device, fence, nullptr);
	std::vector<uint32> pixels(W * H * 4);
	void* map;
	VKCHECK(vkMapMemory(c.device, c.readMem, 0, pixels.size() * 4, 0, &map));
	std::memcpy(pixels.data(), map, pixels.size() * 4);
	vkUnmapMemory(c.device, c.readMem);
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
		a.offset_windowSpaceToClipSpaceTransform != b.offset_windowSpaceToClipSpaceTransform || a.offset_endOfBlock != b.offset_endOfBlock)
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
	bool strictMul, rte;
	float inputs[16];
	uint32 cfile[256 * 4];
	float alphaRef;
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
	Decompiled viaGlsl = Decompile(words, false, k.strictMul, k.rte);
	Decompiled viaSpirv = Decompile(words, true, k.strictMul, k.rte);
	if (viaSpirv.declined || !viaGlsl.ok || !viaSpirv.ok)
		return -1;
	uint32 uniforms[1024] = {};
	if (viaGlsl.uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_REMAPPED)
	{
		for (auto& e : viaGlsl.remapped)
			std::memcpy(uniforms + viaGlsl.offsets.offset_remapped / 4 + e.mappedIndex * 4, k.cfile + e.index * 4, 16);
	}
	else if (viaGlsl.uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CFILE)
		std::memcpy(uniforms + viaGlsl.offsets.offset_uniformRegister / 4, k.cfile, viaGlsl.offsets.count_uniformRegister * 16);
	if (viaGlsl.offsets.offset_alphaTestRef >= 0)
		std::memcpy(uniforms + viaGlsl.offsets.offset_alphaTestRef / 4, &k.alphaRef, 4);
	std::vector<uint32> a = Draw(c, viaGlsl, k.inputs, uniforms, sizeof(uniforms));
	std::vector<uint32> b = Draw(c, viaSpirv, k.inputs, uniforms, sizeof(uniforms));
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
					t.p.clauses.erase(t.p.clauses.begin() + cl);
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
	return out;
}

int main(int argc, char** argv)
{
	const char* outDir = argc > 1 ? argv[1] : nullptr;
	const int count = argc > 2 ? std::atoi(argv[2]) : 400;
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
	// four inputs from the vertex shader, R0 to R3
	g_testPSInputTable.count = 4;
	for (int i = 0; i < 4; i++)
	{
		g_testPSInputTable.import[i].semanticId = (uint32)i;
		g_testPSInputTable.import[i].isFlat = false;
		g_testPSInputTable.import[i].isNoPerspective = i == 3;
	}
	s_regs[mmCB_SHADER_MASK] = 0xF;
	Gen gen;
	gen.rng.seed(argc > 3 ? (uint32)std::strtoul(argv[3], nullptr, 10) : 20261116u);
	int direct = 0, declined = 0, drawn = 0, differing = 0, glslFailed = 0, offsetsDiffer = 0, shapeDiffers = 0;
	std::map<std::string, int> reasons;
	const std::vector<Program> directed = DirectedPrograms();
	CHECK(directed.size() == 3);
	for (int i = 0; i < count + (int)directed.size(); i++)
	{
		const bool allowDeclined = (i % 5) == 0;
		Case k{};
		k.p = i < (int)directed.size() ? directed[i] : gen.Make(allowDeclined);
		Program& p = k.p;
		std::vector<uint32> words = Assemble(p);
		// alpha test now and then
		Latte::LATTE_SX_ALPHA_TEST_CONTROL alpha{};
		if (gen.Chance(30))
		{
			alpha.set_ALPHA_TEST_ENABLE(true);
			alpha.set_ALPHA_FUNC((Latte::E_COMPAREFUNC)gen.R(8));
		}
		k.alphaRaw = alpha.getRawValue();
		s_regs[Latte::REGADDR::SX_ALPHA_TEST_CONTROL] = k.alphaRaw;
		const bool strictMul = k.strictMul = gen.Chance(50);
		const bool rte = k.rte = c.rte && gen.Chance(50);
		Decompiled viaGlsl = Decompile(words, false, strictMul, rte);
		Decompiled viaSpirv = Decompile(words, true, strictMul, rte);
		if (viaSpirv.declined)
		{
			declined++;
			reasons[viaSpirv.declined]++;
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
		if (!SameOffsets(viaGlsl.offsets, viaSpirv.offsets) || viaGlsl.binding != viaSpirv.binding || viaGlsl.setIndex != viaSpirv.setIndex)
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
		for (auto& f : k.inputs)
		{
			uint32 b = gen.Literal();
			std::memcpy(&f, &b, 4);
			if (!std::isfinite(f) && gen.Chance(70))
				f = (float)((int)gen.R(2000) - 1000) / 64.0f;
		}
		drawn++;
		if (Differs(c, k) == 1)
		{
			differing++;
			std::printf("FAIL: program %d draws differently\n", i);
			if (differing == 1 && getenv("DECOMPILER_MINIMIZE"))
				Minimize(c, k, outDir);
		}
	}
	std::printf("decompiler: %d random pixel shaders, %d written as SPIR-V directly (%d drawn both ways, %d differing), %d declined\n",
		count, direct, drawn, differing, declined);
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
