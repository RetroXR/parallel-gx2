// The decompiler's Vulkan output written as SPIR-V directly, where the GLSL
// emitter (LatteDecompilerEmitGLSL.cpp) writes text that rslc then parses
// back and compiles. It follows the GLSL emitter statement for statement -
// the same variables, the same conversions, the same operations in the same
// order, each the instruction rslc makes of the GLSL - so a shader drawn
// from either is the same to the bit (test/decompiler draws both on
// lavapipe and compares). rslc folds an expression of constants as glslang
// does, in double, and rounds once at the end; the same expressions fold
// here, by the same rules, where the GLSL emitter writes them inline (a
// value stored in a variable is no constant to rslc, nor here). Control
// flow is the GLSL emitter's too: the active mask stacks as arrays of
// bool, each clause in an if on them, loops as rslc lowers a while (a
// header, a check, the body, a continue block), its short-circuited &&
// where the right side is no bare variable or constant. Values move
// through memory as rslc moves them too - a register's or a uniform's
// vec4 loaded and the channel extracted, a channel stored through its
// access chain, the registers zeroed by stores at the start - as lavapipe's
// optimizer folds otherwise in places.
// A vertex shader's attributes are decoded as
// LatteDecompiler_emitAttributeDecodeGLSL writes it, case for case.
// What it does not cover yet it declines
// (LatteDecompiler_emitSPIRVShader returns false) and the shader takes the
// GLSL path as before; LatteDecompiler_SPIRVDeclineReason says why.
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/Core/LatteShaderAssembly.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "Cafe/HW/Latte/Core/Latte.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/Core/FetchShader.h"
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompilerInternal.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompilerInstructions.h"
#include "config/ActiveSettings.h"
#include "util/spirv/spirv_writer.h"
#include "util/spirv/spirv_ops.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace
{
	constexpr sint32 DT_F = LATTE_DECOMPILER_DTYPE_FLOAT;
	constexpr sint32 DT_I = LATTE_DECOMPILER_DTYPE_SIGNED_INT;
	constexpr sint32 DT_U = LATTE_DECOMPILER_DTYPE_UNSIGNED_INT;
	constexpr sint32 DT_B = 100; // bool, the emitter's own

	// a value: its id, its type, and - when rslc would hold it as a
	// constant expression - its value as rslc folds it (a float's double)
	struct Val
	{
		uint32 id;
		sint32 type;
		bool k{};
		double d{};
		uint32 bits{};
		bool leaf{}; // a bare variable in the GLSL (PV, PS, a backup)
	};

	// rslc_front.c's folding rules
	static uint32 FloatBits(double d)
	{
		float f = (float)d;
		uint32 b;
		std::memcpy(&b, &f, 4);
		return b;
	}
	static double FoldDiv(double x, double y)
	{
		double inf = HUGE_VAL;
		if (y != 0.0)
			return x / y;
		if (x != x || x == 0.0)
			return inf - inf;
		uint64 xb, yb;
		std::memcpy(&xb, &x, 8);
		std::memcpy(&yb, &y, 8);
		return ((xb ^ yb) >> 63) ? -inf : inf;
	}
	static double FoldRoundEven(double x)
	{
		double f = std::floor(x);
		double r = x - f;
		if (r < 0.5)
			return f;
		if (r > 0.5)
			return f + 1.0;
		return std::fmod(f, 2.0) == 0.0 ? f : f + 1.0;
	}
	static double FoldTrunc(double x) { return x < 0.0 ? std::ceil(x) : std::floor(x); }
	static double FoldMin(double a, double b) { return b < a ? b : a; }
	static double FoldMax(double a, double b) { return a < b ? b : a; }

	// as ALUClauseTemporariesState in the GLSL emitter
	struct PVPSAlias
	{
		enum : uint8 { NONE, GPR, PVPS } location{ NONE };
		uint8 index{}; // GPR, or the group's parity
		uint8 unit{};  // channel, or the ALU unit (4 for PS)
	};

	struct GPRTemporary
	{
		uint8 gpr;
		uint8 chan;
		uint8 var;
	};

	struct SpvShader
	{
		spv_writer_t* w;
		LatteDecompilerShaderContext* ctx;
		LatteDecompilerShader* shader;
		const char* decline{};

		sint32 dt; // the registers' type
		uint32 tVoid, tBool, tF, tI, tU, tV4F, tV4I, tV4U, glsl;
		uint32 gprVar[LATTE_NUM_GPR]{};
		// ivec4 Ri[128] (vec4 Rf[128]): the registers as one array, where a
		// register is written relative to AR
		uint32 gprArrayVar{};
		uint32 pvVar[2][4]{}, psVar[2]{}, backupVar[5]{}, tempVar{}, tempResultF{}, tempResultI{}, ariVar{}, predVar{};
		uint32 ufBlockVar{};
		// uniform buffer mode's: uniform UniformBlock<stage><n> { vec4
		// uf_block<stage><n>[size]; } for each buffer read
		uint32 ubVar[LATTE_NUM_MAX_UNIFORM_BUFFERS]{};
		// the active mask stacks (activeMaskStack, activeMaskStackC) and
		// their lengths, when the shader has control flow
		bool hasActiveMask{};
		uint32 maskVar{}, maskCVar{};
		sint32 maskLen{}, maskCLen{};
		uint32 aluBreakVar{}; // the ALU_BREAK clause's aluBreak
		// the texture units' sampler variables and the sampled image types
		uint32 texVar[LATTE_NUM_MAX_TEX_UNITS]{}, texType[LATTE_NUM_MAX_TEX_UNITS]{}, texImage[LATTE_NUM_MAX_TEX_UNITS]{};
		// uf_texScale's member in the uniform block, -1 for none; whether
		// the image queries' capability is taken
		sint32 texScaleMember[LATTE_NUM_MAX_TEX_UNITS];
		bool imageQuery{}, sampled1D{};
		// gradH and gradV: SET_GRADIENTS_H/V's, SAMPLE_G's
		uint32 gradVar[2]{};
		// the loops open: rslc's blocks for each, its loopActive and counter
		struct Loop
		{
			uint32 header, check, body, cont, merge, active, counter;
		};
		Loop loops[16];
		uint32 loopDepth{};
		uint32 fnMulNonIEEE{}, fnClampFI32{}, fnRedcCUBE{}, fnCubeReverse{};
		// cubeMapSTM, cubeMapFaceId and cubeMapArrayIndex<n>
		uint32 cubeSTMVar{}, cubeFaceVar{}, cubeIndexVar[LATTE_NUM_MAX_TEX_UNITS]{};
		bool sampledCubeArray{};
		uint32 tempResultFVar{};
		uint32 ufMember[16 + LATTE_NUM_MAX_TEX_UNITS]{};
		uint32 inputVar[GPU7_PS_MAX_INPUTS]{};
		uint32 colorOut[LATTE_NUM_COLOR_TARGET]{};
		// a vertex shader's: the attributes, attrDecoder, the per-vertex
		// block, the parameters by semantic, the vertex and instance index
		bool isVertex{};
		uint32 attrVar[LATTE_NUM_MAX_ATTRIBUTE_LOCATIONS]{};
		uint32 decVar{};
		uint32 perVertexVar{};
		// uf_windowSpaceToClipSpaceTransform's and uf_pointSize's members of
		// the uniform block, -1 where it has none
		sint32 windowMember{-1}, pointSizeMember{-1}, alphaRefMember{-1};
		// streamout through the storage buffer: uf_verticesPerInstance's and
		// uf_streamoutBufferBase<n>'s members, int sb_buffer[], sbBase<n>
		sint32 verticesPerInstanceMember{-1};
		sint32 sbBaseMember[LATTE_NUM_STREAMOUT_BUFFER]{ -1, -1, -1, -1 };
		uint32 sbVar{}, sbBaseVar[LATTE_NUM_STREAMOUT_BUFFER]{}, primitiveIdVar{};
		uint32 paramVar[LATTE_ANALYZER_IMPORT_INDEX_PARAM_MAX + 1]{};
		// a geometry shader's draw: the vertex shader's out Vertex { ivec4
		// passV2GParameter<n>; } v2g, the ring it writes, and the pixel
		// shader's inputs at their semantics' locations
		bool usesGS{};
		uint32 v2gVar{}, v2gCount{};
		// a geometry shader's: in Vertex { ... } v2g[], the vertices it takes
		// (1, 3 or 4), its passG2PParameter<n> outputs by n, gl_PerVertex
		bool isGeometry{};
		uint32 gsInputVertices{};
		uint32 g2pVar[256]{};
		// a stage other than the pixel shader: no discard, no implicit level
		bool NotPixel() const { return isVertex || isGeometry; }
		uint32 vertexIndexVar{}, instanceIndexVar{};
		uint32 interfaceIds[LATTE_NUM_MAX_ATTRIBUTE_LOCATIONS + 32 + 8];
		uint32 interfaceCount{};
		// the ALU clause being written
		PVPSAlias pvps[5];
		GPRTemporary temporaries[8];
		uint32 temporaryCount{};

		bool Decline(const char* why)
		{
			if (!decline)
				decline = why;
			return false;
		}
		bool Ok() const { return decline == nullptr && !w->failed; }

		uint32 TypeOf(sint32 t) const
		{
			switch (t)
			{
			case DT_F: return tF;
			case DT_I: return tI;
			case DT_U: return tU;
			default: return tBool;
			}
		}
		uint32 Vec4Of(sint32 t) const
		{
			return t == DT_F ? tV4F : (t == DT_I ? tV4I : tV4U);
		}

		// an instruction in the body with a result
		uint32 Op(uint32 opcode, uint32 resultType, std::initializer_list<uint32> args)
		{
			EnsureBlock();
			uint32 words[16];
			uint32 id = spv_id(w);
			size_t n = 0;
			words[n++] = resultType;
			words[n++] = id;
			for (uint32 a : args)
				words[n++] = a;
			spv_inst(w, SPV_SEC_BODY, opcode, words, n);
			return id;
		}
		void OpVoid(uint32 opcode, std::initializer_list<uint32> args)
		{
			if (opcode != SpvOpLabel)
				EnsureBlock();
			uint32 words[16];
			size_t n = 0;
			for (uint32 a : args)
				words[n++] = a;
			spv_inst(w, SPV_SEC_BODY, opcode, words, n);
		}
		uint32 Ext(uint32 inst, uint32 resultType, std::initializer_list<uint32> args)
		{
			EnsureBlock();
			uint32 words[16];
			uint32 id = spv_id(w);
			size_t n = 0;
			words[n++] = resultType;
			words[n++] = id;
			words[n++] = glsl;
			words[n++] = inst;
			for (uint32 a : args)
				words[n++] = a;
			spv_inst(w, SPV_SEC_BODY, SpvOpExtInst, words, n);
			return id;
		}
		void Decorate(uint32 id, uint32 decoration, std::initializer_list<uint32> args = {})
		{
			uint32 words[8];
			size_t n = 0;
			words[n++] = id;
			words[n++] = decoration;
			for (uint32 a : args)
				words[n++] = a;
			spv_inst(w, SPV_SEC_DECORATION, SpvOpDecorate, words, n);
		}
		uint32 ConstF(float f)
		{
			uint32 bits;
			std::memcpy(&bits, &f, 4);
			return spv_const_bits(w, tF, bits);
		}
		uint32 ConstBitsF(uint32 bits) { return spv_const_bits(w, tF, bits); }
		uint32 ConstI(sint32 v) { return spv_const_bits(w, tI, (uint32)v); }
		uint32 ConstU(uint32 v) { return spv_const_bits(w, tU, v); }

		// constants as the GLSL emitter writes them
		Val KF(double d)
		{
			Val v{ ConstBitsF(FloatBits(d)), DT_F, true, d, FloatBits(d) };
			return v;
		}
		Val KFText(const char* text) { return KF(std::strtod(text, nullptr)); }
		Val KI(sint32 i) { return { ConstI(i), DT_I, true, 0.0, (uint32)i }; }
		Val KU(uint32 u) { return { ConstU(u), DT_U, true, 0.0, u }; }
		Val KB(bool b) { return { spv_const_bool(w, b ? 1 : 0), DT_B, true, 0.0, b ? 1u : 0u }; }
		Val Run(uint32 id, sint32 type) { return { id, type }; }
		Val Leaf(uint32 id)
		{
			Val v{ id, dt };
			v.leaf = true;
			return v;
		}
		Val K(sint32 type, uint32 bits)
		{
			if (type == DT_I)
				return KI((sint32)bits);
			if (type == DT_U)
				return KU(bits);
			return KB(bits != 0);
		}

		// arithmetic, folded as rslc folds it when every operand is constant
		Val FBin(uint32 opcode, Val a, Val b)
		{
			if (a.k && b.k)
			{
				switch (opcode)
				{
				case SpvOpFAdd: return KF(a.d + b.d);
				case SpvOpFSub: return KF(a.d - b.d);
				case SpvOpFMul: return KF(a.d * b.d);
				default: return KF(FoldDiv(a.d, b.d));
				}
			}
			return Run(Op(opcode, tF, { a.id, b.id }), DT_F);
		}
		Val FNeg(Val a) { return a.k ? KF(-a.d) : Run(Op(SpvOpFNegate, tF, { a.id }), DT_F); }
		Val FUnary(uint32 inst, Val a)
		{
			if (a.k)
			{
				switch (inst)
				{
				case GLSLstd450FAbs: return KF(std::fabs(a.d));
				case GLSLstd450Floor: return KF(std::floor(a.d));
				case GLSLstd450Fract: return KF(a.d - std::floor(a.d));
				case GLSLstd450Trunc: return KF(FoldTrunc(a.d));
				case GLSLstd450RoundEven: return KF(FoldRoundEven(a.d));
				case GLSLstd450Exp2: return KF(std::pow(2.0, a.d));
				case GLSLstd450Log2: return KF(std::log(a.d) / std::log(2.0));
				case GLSLstd450Sqrt: return KF(std::sqrt(a.d));
				case GLSLstd450Sin: return KF(std::sin(a.d));
				case GLSLstd450Cos: return KF(std::cos(a.d));
				}
			}
			return Run(Ext(inst, tF, { a.id }), DT_F);
		}
		Val FMinMax(bool isMax, Val a, Val b)
		{
			if (a.k && b.k)
				return KF(isMax ? FoldMax(a.d, b.d) : FoldMin(a.d, b.d));
			return Run(Ext(isMax ? GLSLstd450FMax : GLSLstd450FMin, tF, { a.id, b.id }), DT_F);
		}
		Val FCmp(uint32 opcode, Val a, Val b)
		{
			if (a.k && b.k)
			{
				switch (opcode)
				{
				case SpvOpFOrdGreaterThan: return KB(a.d > b.d);
				case SpvOpFOrdGreaterThanEqual: return KB(a.d >= b.d);
				case SpvOpFOrdLessThan: return KB(a.d < b.d);
				case SpvOpFOrdLessThanEqual: return KB(a.d <= b.d);
				case SpvOpFOrdEqual: return KB(a.d == b.d);
				default: return KB(a.d != b.d);
				}
			}
			return Run(Op(opcode, tBool, { a.id, b.id }), DT_B);
		}
		Val ICmp(uint32 opcode, Val a, Val b)
		{
			if (a.k && b.k)
			{
				sint32 x = (sint32)a.bits, y = (sint32)b.bits;
				switch (opcode)
				{
				case SpvOpIEqual: return KB(a.bits == b.bits);
				case SpvOpINotEqual: return KB(a.bits != b.bits);
				case SpvOpSGreaterThan: return KB(x > y);
				case SpvOpSGreaterThanEqual: return KB(x >= y);
				case SpvOpUGreaterThan: return KB(a.bits > b.bits);
				default: return KB(a.bits >= b.bits);
				}
			}
			return Run(Op(opcode, tBool, { a.id, b.id }), DT_B);
		}
		// (cond)?a:b with constant sides: an OpSelect, even on a constant
		// condition - rslc folds no ?: the GLSL emitter writes
		Val Select(Val c, Val a, Val b, sint32 type)
		{
			return Run(Op(SpvOpSelect, TypeOf(type), { c.id, a.id, b.id }), type);
		}
		Val IBin(uint32 opcode, sint32 type, Val a, Val b)
		{
			if (a.k && b.k)
			{
				uint32 x = a.bits, y = b.bits;
				uint32 r;
				switch (opcode)
				{
				case SpvOpIAdd: r = x + y; break;
				case SpvOpISub: r = x - y; break;
				case SpvOpIMul: r = x * y; break;
				case SpvOpBitwiseAnd: r = x & y; break;
				case SpvOpBitwiseOr: r = x | y; break;
				case SpvOpBitwiseXor: r = x ^ y; break;
				case SpvOpShiftLeftLogical: r = x << (y & 31); break;
				case SpvOpShiftRightLogical: r = x >> (y & 31); break;
				default: r = (uint32)((sint32)x >> (y & 31)); break;
				}
				return K(type, r);
			}
			return Run(Op(opcode, TypeOf(type), { a.id, b.id }), type);
		}
		Val IMinMax(uint32 inst, sint32 type, Val a, Val b)
		{
			if (a.k && b.k)
			{
				uint32 r;
				sint32 sx = (sint32)a.bits, sy = (sint32)b.bits;
				switch (inst)
				{
				case GLSLstd450SMax: r = (uint32)(sx < sy ? sy : sx); break;
				case GLSLstd450SMin: r = (uint32)(sy < sx ? sy : sx); break;
				case GLSLstd450UMax: r = a.bits < b.bits ? b.bits : a.bits; break;
				default: r = b.bits < a.bits ? b.bits : a.bits; break;
				}
				return K(type, r);
			}
			return Run(Ext(inst, TypeOf(type), { a.id, b.id }), type);
		}
		// int(float), uint(float), float(int), float(uint): constructors
		Val Construct(Val a, sint32 to)
		{
			if (a.k)
			{
				if (to == DT_F)
					return KF(a.type == DT_I ? (double)(sint32)a.bits : (double)a.bits);
				if (a.type == DT_F)
					return K(to, to == DT_I ? (uint32)(sint32)a.d : (uint32)a.d);
				return K(to, a.bits);
			}
			uint32 opcode;
			if (to == DT_F)
				opcode = a.type == DT_I ? SpvOpConvertSToF : SpvOpConvertUToF;
			else if (a.type == DT_F)
				opcode = to == DT_I ? SpvOpConvertFToS : SpvOpConvertFToU;
			else
				opcode = SpvOpBitcast;
			return Run(Op(opcode, TypeOf(to), { a.id }), to);
		}
		// an instruction in a helper function
		uint32 HOp(uint32 opcode, uint32 resultType, std::initializer_list<uint32> args)
		{
			uint32 words[8];
			uint32 id = spv_id(w);
			size_t n = 0;
			words[n++] = resultType;
			words[n++] = id;
			for (uint32 a : args)
				words[n++] = a;
			spv_inst(w, SPV_SEC_HELPERS, opcode, words, n);
			return id;
		}
		void HOpVoid(uint32 opcode, std::initializer_list<uint32> args)
		{
			uint32 words[8];
			size_t n = 0;
			for (uint32 a : args)
				words[n++] = a;
			spv_inst(w, SPV_SEC_HELPERS, opcode, words, n);
		}
		uint32 HelperBegin(uint32 ret, uint32 paramPtr, uint32 paramCount, uint32* params)
		{
			uint32 types[2] = { paramPtr, paramPtr };
			uint32 fnType = spv_type_function(w, ret, types, paramCount);
			uint32 fn = spv_id(w);
			HOpVoid(SpvOpFunction, { ret, fn, SpvFunctionControlMaskNone, fnType });
			for (uint32 i = 0; i < paramCount; i++)
			{
				params[i] = spv_id(w);
				HOpVoid(SpvOpFunctionParameter, { paramPtr, params[i] });
			}
			HOpVoid(SpvOpLabel, { spv_id(w) });
			return fn;
		}
		// float mul_nonIEEE(float a, float b){ if( a == 0.0 || b == 0.0 ) return 0.0; return a*b; }
		uint32 MulNonIEEEFunction()
		{
			if (fnMulNonIEEE)
				return fnMulNonIEEE;
			uint32 ptr = spv_type_pointer(w, SpvStorageClassFunction, tF);
			uint32 p[2];
			fnMulNonIEEE = HelperBegin(tF, ptr, 2, p);
			uint32 a = HOp(SpvOpLoad, tF, { p[0] });
			uint32 za = HOp(SpvOpFOrdEqual, tBool, { a, ConstF(0.0f) });
			uint32 b = HOp(SpvOpLoad, tF, { p[1] });
			uint32 zb = HOp(SpvOpFOrdEqual, tBool, { b, ConstF(0.0f) });
			uint32 z = HOp(SpvOpLogicalOr, tBool, { za, zb });
			uint32 thenLabel = spv_id(w), mergeLabel = spv_id(w);
			HOpVoid(SpvOpSelectionMerge, { mergeLabel, SpvSelectionControlMaskNone });
			HOpVoid(SpvOpBranchConditional, { z, thenLabel, mergeLabel });
			HOpVoid(SpvOpLabel, { thenLabel });
			HOpVoid(SpvOpReturnValue, { ConstF(0.0f) });
			HOpVoid(SpvOpLabel, { mergeLabel });
			uint32 a2 = HOp(SpvOpLoad, tF, { p[0] });
			uint32 b2 = HOp(SpvOpLoad, tF, { p[1] });
			uint32 m = HOp(SpvOpFMul, tF, { a2, b2 });
			HOpVoid(SpvOpReturnValue, { m });
			HOpVoid(SpvOpFunctionEnd, {});
			return fnMulNonIEEE;
		}
		// int clampFI32(int v), as the GLSL emitter writes it
		uint32 ClampFI32Function()
		{
			if (fnClampFI32)
				return fnClampFI32;
			uint32 ptr = spv_type_pointer(w, SpvStorageClassFunction, tI);
			uint32 p[1];
			fnClampFI32 = HelperBegin(tI, ptr, 1, p);
			uint32 v = HOp(SpvOpLoad, tI, { p[0] });
			uint32 isMax = HOp(SpvOpIEqual, tBool, { v, ConstI(0x7FFFFFFF) });
			uint32 l1 = spv_id(w), l2 = spv_id(w), merge1 = spv_id(w);
			HOpVoid(SpvOpSelectionMerge, { merge1, SpvSelectionControlMaskNone });
			HOpVoid(SpvOpBranchConditional, { isMax, l1, l2 });
			HOpVoid(SpvOpLabel, { l1 });
			HOpVoid(SpvOpReturnValue, { HOp(SpvOpBitcast, tI, { ConstF(1.0f) }) });
			HOpVoid(SpvOpLabel, { l2 });
			uint32 v2 = HOp(SpvOpLoad, tI, { p[0] });
			uint32 isMinus1 = HOp(SpvOpIEqual, tBool, { v2, ConstI(-1) });
			uint32 l3 = spv_id(w), merge2 = spv_id(w);
			HOpVoid(SpvOpSelectionMerge, { merge2, SpvSelectionControlMaskNone });
			HOpVoid(SpvOpBranchConditional, { isMinus1, l3, merge2 });
			HOpVoid(SpvOpLabel, { l3 });
			HOpVoid(SpvOpReturnValue, { HOp(SpvOpBitcast, tI, { ConstF(0.0f) }) });
			HOpVoid(SpvOpLabel, { merge2 });
			HOpVoid(SpvOpBranch, { merge1 });
			HOpVoid(SpvOpLabel, { merge1 });
			uint32 v3 = HOp(SpvOpLoad, tI, { p[0] });
			uint32 f = HOp(SpvOpBitcast, tF, { v3 });
			uint32 c = HOp(SpvOpExtInst, tF, { glsl, GLSLstd450FClamp, f, ConstF(0.0f), ConstF(1.0f) });
			HOpVoid(SpvOpReturnValue, { HOp(SpvOpBitcast, tI, { c }) });
			HOpVoid(SpvOpFunctionEnd, {});
			return fnClampFI32;
		}
		// a helper of parameters of their own types, all pointers
		uint32 HelperBeginTyped(uint32 ret, const uint32* paramPtrs, uint32 paramCount, uint32* params)
		{
			uint32 fnType = spv_type_function(w, ret, paramPtrs, paramCount);
			uint32 fn = spv_id(w);
			HOpVoid(SpvOpFunction, { ret, fn, SpvFunctionControlMaskNone, fnType });
			for (uint32 i = 0; i < paramCount; i++)
			{
				params[i] = spv_id(w);
				HOpVoid(SpvOpFunctionParameter, { paramPtrs[i], params[i] });
			}
			HOpVoid(SpvOpLabel, { spv_id(w) });
			return fn;
		}
		uint32 HConstruct(uint32 type, std::initializer_list<uint32> parts)
		{
			uint32 words[8];
			size_t n = 0;
			uint32 id = spv_id(w);
			words[n++] = type;
			words[n++] = id;
			for (uint32 a : parts)
				words[n++] = a;
			spv_inst(w, SPV_SEC_HELPERS, SpvOpCompositeConstruct, words, n);
			return id;
		}
		// c ? a : b of a side not a leaf: a branch each way, a phi; the
		// sides' values made by make(true/false) in their blocks
		template<typename F>
		uint32 HTernary(uint32 type, uint32 cond, F make)
		{
			uint32 thenL = spv_id(w), elseL = spv_id(w), merge = spv_id(w);
			HOpVoid(SpvOpSelectionMerge, { merge, SpvSelectionControlMaskNone });
			HOpVoid(SpvOpBranchConditional, { cond, thenL, elseL });
			HOpVoid(SpvOpLabel, { thenL });
			uint32 a = make(true);
			HOpVoid(SpvOpBranch, { merge });
			HOpVoid(SpvOpLabel, { elseL });
			uint32 b = make(false);
			HOpVoid(SpvOpBranch, { merge });
			HOpVoid(SpvOpLabel, { merge });
			return HOp(SpvOpPhi, type, { a, thenL, b, elseL });
		}
		// void redcCUBE(vec4 src0, vec4 src1, out vec3 stm, out int faceId),
		// as the GLSL emitter writes it and rslc lowers it
		uint32 RedcCUBEFunction()
		{
			if (fnRedcCUBE)
				return fnRedcCUBE;
			const uint32 tV3F = spv_type_vector(w, tF, 3);
			const uint32 ptrs[4] = { spv_type_pointer(w, SpvStorageClassFunction, tV4F), spv_type_pointer(w, SpvStorageClassFunction, tV4F),
				spv_type_pointer(w, SpvStorageClassFunction, tV3F), spv_type_pointer(w, SpvStorageClassFunction, tI) };
			uint32 p[4];
			fnRedcCUBE = HelperBeginTyped(tVoid, ptrs, 4, p);
			// float x = src1.y, y = src1.x, z = src0.x;
			uint32 x = HOp(SpvOpCompositeExtract, tF, { HOp(SpvOpLoad, tV4F, { p[1] }), 1 });
			uint32 y = HOp(SpvOpCompositeExtract, tF, { HOp(SpvOpLoad, tV4F, { p[1] }), 0 });
			uint32 z = HOp(SpvOpCompositeExtract, tF, { HOp(SpvOpLoad, tV4F, { p[0] }), 0 });
			uint32 ax = HOp(SpvOpExtInst, tF, { glsl, GLSLstd450FAbs, x });
			uint32 ay = HOp(SpvOpExtInst, tF, { glsl, GLSLstd450FAbs, y });
			uint32 az = HOp(SpvOpExtInst, tF, { glsl, GLSLstd450FAbs, z });
			auto neg = [&](uint32 v) { return HOp(SpvOpFNegate, tF, { v }); };
			auto below0 = [&](uint32 v) { return HOp(SpvOpFOrdLessThan, tBool, { v, ConstF(0.0f) }); };
			auto twice = [&](uint32 v) { return HOp(SpvOpFMul, tF, { v, ConstF(2.0f) }); };
			auto face = [&](uint32 v, sint32 a, sint32 b) {
				HOpVoid(SpvOpStore, { p[3], HOp(SpvOpSelect, tI, { below0(v), ConstI(a), ConstI(b) }) });
			};
			// if (az >= ax && az >= ay)
			uint32 c = HOp(SpvOpLogicalAnd, tBool, { HOp(SpvOpFOrdGreaterThanEqual, tBool, { az, ax }),
				HOp(SpvOpFOrdGreaterThanEqual, tBool, { az, ay }) });
			uint32 t1 = spv_id(w), e1 = spv_id(w), m1 = spv_id(w);
			HOpVoid(SpvOpSelectionMerge, { m1, SpvSelectionControlMaskNone });
			HOpVoid(SpvOpBranchConditional, { c, t1, e1 });
			HOpVoid(SpvOpLabel, { t1 });
			{
				// stm = vec3(-y, z < 0.0 ? -x : x, z * 2.0); faceId = z < 0.0 ? 5 : 4;
				uint32 a = neg(y);
				uint32 b = HTernary(tF, below0(z), [&](bool t) { return t ? neg(x) : x; });
				HOpVoid(SpvOpStore, { p[2], HConstruct(tV3F, { a, b, twice(z) }) });
				face(z, 5, 4);
			}
			HOpVoid(SpvOpBranch, { m1 });
			HOpVoid(SpvOpLabel, { e1 });
			// else if (ay >= ax)
			uint32 t2 = spv_id(w), e2 = spv_id(w), m2 = spv_id(w);
			uint32 c2 = HOp(SpvOpFOrdGreaterThanEqual, tBool, { ay, ax });
			HOpVoid(SpvOpSelectionMerge, { m2, SpvSelectionControlMaskNone });
			HOpVoid(SpvOpBranchConditional, { c2, t2, e2 });
			HOpVoid(SpvOpLabel, { t2 });
			{
				// stm = vec3(y < 0.0 ? -z : z, x, y * 2.0); faceId = y < 0.0 ? 3 : 2;
				uint32 a = HTernary(tF, below0(y), [&](bool t) { return t ? neg(z) : z; });
				HOpVoid(SpvOpStore, { p[2], HConstruct(tV3F, { a, x, twice(y) }) });
				face(y, 3, 2);
			}
			HOpVoid(SpvOpBranch, { m2 });
			HOpVoid(SpvOpLabel, { e2 });
			{
				// stm = vec3(-y, x < 0.0 ? z : -z, x * 2.0); faceId = x < 0.0 ? 1 : 0;
				uint32 a = neg(y);
				uint32 b = HTernary(tF, below0(x), [&](bool t) { return t ? z : neg(z); });
				HOpVoid(SpvOpStore, { p[2], HConstruct(tV3F, { a, b, twice(x) }) });
				face(x, 1, 0);
			}
			HOpVoid(SpvOpBranch, { m2 });
			HOpVoid(SpvOpLabel, { m2 });
			HOpVoid(SpvOpBranch, { m1 });
			HOpVoid(SpvOpLabel, { m1 });
			HOpVoid(SpvOpReturn, {});
			HOpVoid(SpvOpFunctionEnd, {});
			return fnRedcCUBE;
		}
		// vec3 redcCUBEReverse(vec2 st, int faceId)
		uint32 CubeReverseFunction()
		{
			if (fnCubeReverse)
				return fnCubeReverse;
			const uint32 tV2F = spv_type_vector(w, tF, 2), tV3F = spv_type_vector(w, tF, 3);
			const uint32 ptrs[2] = { spv_type_pointer(w, SpvStorageClassFunction, tV2F), spv_type_pointer(w, SpvStorageClassFunction, tI) };
			uint32 p[2];
			fnCubeReverse = HelperBeginTyped(tV3F, ptrs, 2, p);
			// vec2 tc = (st - vec2(1.5)) * 2.0;
			uint32 k15[2] = { ConstF(1.5f), ConstF(1.5f) };
			uint32 d = HOp(SpvOpFSub, tV2F, { HOp(SpvOpLoad, tV2F, { p[0] }), spv_const_composite(w, tV2F, k15, 2) });
			uint32 tc = HOp(SpvOpVectorTimesScalar, tV2F, { d, ConstF(2.0f) });
			auto tx = [&](bool negate) { uint32 v = HOp(SpvOpCompositeExtract, tF, { tc, 0 }); return negate ? HOp(SpvOpFNegate, tF, { v }) : v; };
			auto ty = [&](bool negate) { uint32 v = HOp(SpvOpCompositeExtract, tF, { tc, 1 }); return negate ? HOp(SpvOpFNegate, tF, { v }) : v; };
			for (sint32 f = 0; f < 5; f++)
			{
				uint32 c = HOp(SpvOpIEqual, tBool, { HOp(SpvOpLoad, tI, { p[1] }), ConstI(f) });
				uint32 t = spv_id(w), m = spv_id(w);
				HOpVoid(SpvOpSelectionMerge, { m, SpvSelectionControlMaskNone });
				HOpVoid(SpvOpBranchConditional, { c, t, m });
				HOpVoid(SpvOpLabel, { t });
				uint32 r;
				switch (f)
				{
				case 0: { uint32 a = ty(true); r = HConstruct(tV3F, { ConstF(1.0f), a, tx(true) }); break; }
				case 1: { uint32 a = ty(true); r = HConstruct(tV3F, { ConstF(-1.0f), a, tx(false) }); break; }
				case 2: { uint32 a = tx(false); r = HConstruct(tV3F, { a, ConstF(1.0f), ty(false) }); break; }
				case 3: { uint32 a = tx(false); r = HConstruct(tV3F, { a, ConstF(-1.0f), ty(true) }); break; }
				default: { uint32 a = tx(false); r = HConstruct(tV3F, { a, ty(true), ConstF(1.0f) }); break; }
				}
				HOpVoid(SpvOpReturnValue, { r });
				HOpVoid(SpvOpLabel, { m });
			}
			uint32 a = tx(true);
			uint32 b = ty(true);
			HOpVoid(SpvOpReturnValue, { HConstruct(tV3F, { a, b, ConstF(-1.0f) }) });
			HOpVoid(SpvOpFunctionEnd, {});
			return fnCubeReverse;
		}
		// a call with arguments of their own types: each in a variable of
		// its own, an out argument's left unset
		uint32 CallTyped(uint32 fn, uint32 ret, const uint32* argTypes, const uint32* args, uint32 n, uint32* temps)
		{
			uint32 words[3 + 4];
			words[0] = ret;
			words[1] = spv_id(w);
			words[2] = fn;
			for (uint32 i = 0; i < n; i++)
			{
				temps[i] = Local(argTypes[i], 0);
				if (args[i])
					Store(temps[i], args[i]);
				words[3 + i] = temps[i];
			}
			EnsureBlock();
			spv_inst(w, SPV_SEC_BODY, SpvOpFunctionCall, words, 3 + n);
			return words[1];
		}
		// a call as rslc makes one: each argument into a variable of its own
		Val Call(uint32 fn, uint32 ret, sint32 retType, uint32 argType, std::initializer_list<uint32> args)
		{
			uint32 ptrs[2];
			uint32 n = 0;
			for (uint32 a : args)
			{
				ptrs[n] = Local(argType, 0);
				Store(ptrs[n], a);
				n++;
			}
			if (n == 1)
				return Run(Op(SpvOpFunctionCall, ret, { fn, ptrs[0] }), retType);
			return Run(Op(SpvOpFunctionCall, ret, { fn, ptrs[0], ptrs[1] }), retType);
		}
		Val MulNonIEEE(Val a, Val b)
		{
			return Call(MulNonIEEEFunction(), tF, DT_F, tF, { a.id, b.id });
		}

		// a function variable; initialized as rslc initializes a declaration:
		// a store where it is declared, no initializer of the variable's own
		uint32 Local(uint32 type, uint32 init)
		{
			uint32 ptr = spv_type_pointer(w, SpvStorageClassFunction, type);
			uint32 words[3] = { ptr, spv_id(w), SpvStorageClassFunction };
			spv_inst(w, SPV_SEC_LOCALS, SpvOpVariable, words, 3);
			if (init)
				Store(words[1], init);
			return words[1];
		}
		uint32 Load(uint32 type, uint32 ptr) { return Op(SpvOpLoad, type, { ptr }); }
		void Store(uint32 ptr, uint32 value) { OpVoid(SpvOpStore, { ptr, value }); }
		// a channel of a vec4 variable
		uint32 Channel(uint32 vecPtr, sint32 elemType, uint32 chan, uint32 storage = SpvStorageClassFunction)
		{
			uint32 ptr = spv_type_pointer(w, storage, TypeOf(elemType));
			return Op(SpvOpAccessChain, ptr, { vecPtr, ConstI((sint32)chan) });
		}

		// a channel of a vec4 variable read as rslc reads R1i.x: the whole
		// vector loaded, the channel extracted
		uint32 ReadChannel(uint32 vecVar, uint32 chan)
		{
			uint32 v = Load(Vec4Of(dt), vecVar);
			return Op(SpvOpCompositeExtract, TypeOf(dt), { v, chan });
		}

		// the registers are declared at the start, as the GLSL emitter's
		uint32 GPR(uint32 index)
		{
			if (gprArrayVar)
				return Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassFunction, Vec4Of(dt)), { gprArrayVar, ConstI((sint32)index) });
			if (!gprVar[index])
			{
				Decline("register not declared");
				gprVar[index] = Local(Vec4Of(dt), 0);
			}
			return gprVar[index];
		}

		// _emitTypeConversionPrefix/Suffix: bit casts, as the GLSL functions
		// and constructors used there compile
		Val Convert(Val v, sint32 to)
		{
			if (v.type == to)
				return v;
			// int() and uint() between the integers: constructors, which fold
			if ((v.type == DT_I && to == DT_U) || (v.type == DT_U && to == DT_I))
			{
				if (v.k)
					return K(to, v.bits);
				return Run(Op(SpvOpBitcast, TypeOf(to), { v.id }), to);
			}
			// the bit casts: builtins rslc does not fold
			if ((v.type == DT_F && (to == DT_I || to == DT_U)) || (v.type == DT_I && to == DT_F))
				return Run(Op(SpvOpBitcast, TypeOf(to), { v.id }), to);
			// the GLSL emitter writes nothing there: a type error
			Decline("type conversion the GLSL emitter does not write");
			return { ConstI(0), to };
		}

		// the block the body is in, for an OpPhi's parents; none after a
		// branch or a kill, until something follows - in a block of its
		// own then, unreached, as rslc opens one
		uint32 curBlock{};
		void Label(uint32 id)
		{
			OpVoid(SpvOpLabel, { id });
			curBlock = id;
		}
		void EnsureBlock()
		{
			if (!curBlock)
				Label(spv_id(w));
		}
		void Branch(uint32 target)
		{
			OpVoid(SpvOpBranch, { target });
			curBlock = 0;
		}
		void Kill()
		{
			OpVoid(SpvOpKill, {});
			curBlock = 0;
		}

		// if (cond) discard;
		void KillIf(uint32 cond)
		{
			if (NotPixel())
			{
				Decline("kill in a vertex shader"); // no discard there: no GLSL either
				return;
			}
			uint32 killLabel = spv_id(w), mergeLabel = spv_id(w);
			OpVoid(SpvOpSelectionMerge, { mergeLabel, SpvSelectionControlMaskNone });
			OpVoid(SpvOpBranchConditional, { cond, killLabel, mergeLabel });
			Label(killLabel);
			Kill();
			Label(mergeLabel);
		}

		// if (cond) { then } [else { otherwise }], as rslc lowers it
		template<typename TThen, typename TElse>
		void IfElse(uint32 cond, TThen then, TElse otherwise, bool hasElse)
		{
			uint32 thenLabel = spv_id(w), mergeLabel = spv_id(w);
			uint32 elseLabel = hasElse ? spv_id(w) : mergeLabel;
			OpVoid(SpvOpSelectionMerge, { mergeLabel, SpvSelectionControlMaskNone });
			OpVoid(SpvOpBranchConditional, { cond, thenLabel, elseLabel });
			Label(thenLabel);
			then();
			if (curBlock)
				Branch(mergeLabel);
			if (hasElse)
			{
				Label(elseLabel);
				otherwise();
				if (curBlock)
					Branch(mergeLabel);
			}
			Label(mergeLabel);
		}
		template<typename TThen>
		void If(uint32 cond, TThen then)
		{
			IfElse(cond, then, [] {}, false);
		}

		// a && b where b is no leaf: short-circuited, as rslc does
		template<typename TRight>
		uint32 AndShortCircuit(uint32 a, TRight right)
		{
			uint32 from = curBlock;
			uint32 rightLabel = spv_id(w), mergeLabel = spv_id(w);
			OpVoid(SpvOpSelectionMerge, { mergeLabel, SpvSelectionControlMaskNone });
			OpVoid(SpvOpBranchConditional, { a, rightLabel, mergeLabel });
			Label(rightLabel);
			uint32 b = right();
			uint32 rightEnd = curBlock;
			OpVoid(SpvOpBranch, { mergeLabel });
			Label(mergeLabel);
			return Op(SpvOpPhi, tBool, { spv_const_bool(w, 0), from, b, rightEnd });
		}

		// (cond)?(a):(b) as rslc lowers it: an OpSelect when each side is a
		// constant expression (the GLSL emitter's sides are never a bare
		// variable), control flow otherwise - on a constant condition too,
		// rslc folds no ?: the GLSL emitter writes. The sides' code, written
		// after the condition's from 'sidesFrom', moves into the branches.
		Val Ternary(Val cond, Val a, size_t aEnd, Val b, size_t sidesFrom, sint32 type)
		{
			if ((a.k || a.leaf) && (b.k || b.leaf))
				return Run(Op(SpvOpSelect, TypeOf(type), { cond.id, a.id, b.id }), type);
			spv_words_t& body = w->sec[SPV_SEC_BODY];
			const size_t end = body.n;
			thread_local std::vector<uint32> s_sides;
			s_sides.assign(body.w + sidesFrom, body.w + end);
			body.n = sidesFrom;
			const size_t aWords = aEnd - sidesFrom;
			uint32 aLabel = spv_id(w), bLabel = spv_id(w), mergeLabel = spv_id(w);
			OpVoid(SpvOpSelectionMerge, { mergeLabel, SpvSelectionControlMaskNone });
			OpVoid(SpvOpBranchConditional, { cond.id, aLabel, bLabel });
			// a side's own blocks (a relative read's ternaries): the phi's
			// predecessor is the side's last block
			uint32 aLast = aLabel, bLast = bLabel;
			Label(aLabel);
			for (size_t i = 0; i < aWords;)
			{
				uint32 n = s_sides[i] >> 16;
				if ((s_sides[i] & 0xFFFF) == SpvOpLabel)
					aLast = s_sides[i + 1];
				spv_inst(w, SPV_SEC_BODY, s_sides[i] & 0xFFFF, s_sides.data() + i + 1, n - 1);
				i += n;
			}
			OpVoid(SpvOpBranch, { mergeLabel });
			Label(bLabel);
			for (size_t i = aWords; i < s_sides.size();)
			{
				uint32 n = s_sides[i] >> 16;
				if ((s_sides[i] & 0xFFFF) == SpvOpLabel)
					bLast = s_sides[i + 1];
				spv_inst(w, SPV_SEC_BODY, s_sides[i] & 0xFFFF, s_sides.data() + i + 1, n - 1);
				i += n;
			}
			OpVoid(SpvOpBranch, { mergeLabel });
			Label(mergeLabel);
			return Run(Op(SpvOpPhi, TypeOf(type), { a.id, aLast, b.id, bLast }), type);
		}
		size_t BodyAt() const { return w->sec[SPV_SEC_BODY].n; }

		// activeMaskStack[i] (c false) or activeMaskStackC[i] (c true); an
		// index past either the GLSL emitter writes is no shader to rslc
		uint32 MaskPtr(bool c, sint32 i)
		{
			if (i < 0 || i >= (c ? maskCLen : maskLen))
			{
				Decline("active mask index out of range");
				i = 0;
			}
			uint32 ptr = spv_type_pointer(w, SpvStorageClassFunction, tBool);
			return Op(SpvOpAccessChain, ptr, { c ? maskCVar : maskVar, ConstI(i) });
		}
		uint32 IsTrue(uint32 v) { return Op(SpvOpLogicalEqual, tBool, { v, spv_const_bool(w, 1) }); }
		uint32 MaskIsTrue(bool c, sint32 i) { return IsTrue(Load(tBool, MaskPtr(c, i))); }
		void MaskStore(bool c, sint32 i, uint32 v) { Store(MaskPtr(c, i), v); }
		// a == true && activeMaskStackC[i] == true
		uint32 AndMaskC(uint32 aIsTrue, sint32 i)
		{
			return AndShortCircuit(aIsTrue, [&] { return MaskIsTrue(true, i); });
		}
		// activeMaskStackC[d+1-n] = activeMaskStack[d-n] == true && activeMaskStackC[d-n] == true;
		void Pop(sint32 d, sint32 n)
		{
			uint32 v = AndMaskC(MaskIsTrue(false, d - n), d - n);
			MaskStore(true, d + 1 - n, v);
		}
		// activeMaskStack[d] = activeMaskStack[d] == false; and the C below
		void Else(sint32 d)
		{
			uint32 m = Load(tBool, MaskPtr(false, d));
			MaskStore(false, d, Op(SpvOpLogicalEqual, tBool, { m, spv_const_bool(w, 0) }));
			uint32 v = AndMaskC(MaskIsTrue(false, d), d);
			MaskStore(true, d + 1, v);
		}
		// break; out of the innermost loop
		void Break()
		{
			if (loopDepth == 0)
			{
				Decline("break outside a loop");
				return;
			}
			Branch(loops[loopDepth - 1].merge);
		}
		bool LoopStart(sint32 d);
		bool LoopEnd(sint32 d);
		bool EmitControlledALUClause(LatteDecompilerCFInstruction& cf);
		bool DeclareVertexIO();
		bool DeclareGeometryIO();
		bool EmitRingRead(LatteDecompilerTEXInstruction& t);
		bool EmitVertexInputs();
		bool EmitAttributeImport(LatteParsedFetchShaderAttribute& attrib);
		bool EmitAttributeDecode(LatteParsedFetchShaderAttribute& attrib);
		bool EmitVertexExport(LatteDecompilerCFInstruction& cf);
		bool EmitRingWrite(LatteDecompilerCFInstruction& cf);
		bool EmitGeometryRingWrite(LatteDecompilerCFInstruction& cf);
		bool EmitEmitVertex(LatteDecompilerCFInstruction& cf);
		bool DeclareStreamout();
		bool DeclareUniformBuffers();
		bool EmitUniformFetch(LatteDecompilerTEXInstruction& t);
		bool EmitStreamoutBases();
		void StreamStore(uint32 buffer, uint32 dword, uint32 value);
		bool EmitStreamWrite(LatteDecompilerCFInstruction& cf);
		void SetPosition(uint32 pos);
		uint32 BuiltinInput(uint32 builtin);
		uint32 VertexOutSemantic(uint32 index);
		// attrDecoder's statements, as rslc makes them
		uint32 tV2U{}, tV3U{}, tV2F{};
		uint32 VecU(uint32 n) { return n == 1 ? tU : spv_type_vector(w, tU, n); }
		uint32 VecF(uint32 n) { return n == 1 ? tF : spv_type_vector(w, tF, n); }
		uint32 SplatU(uint32 v, uint32 n)
		{
			if (n == 1)
				return ConstU(v);
			uint32 parts[4] = { ConstU(v), ConstU(v), ConstU(v), ConstU(v) };
			return spv_const_composite(w, VecU(n), parts, n);
		}
		uint32 SplatF(float f, uint32 n)
		{
			if (n == 1)
				return ConstF(f);
			uint32 parts[4] = { ConstF(f), ConstF(f), ConstF(f), ConstF(f) };
			return spv_const_composite(w, VecF(n), parts, n);
		}
		// v.<sel> of a vector value: an extract for one, a shuffle for more
		uint32 Swizzle(uint32 v, uint32 elemType, const uint8* sel, uint32 n)
		{
			if (n == 1)
				return Op(SpvOpCompositeExtract, elemType, { v, sel[0] });
			uint32 words[2 + 2 + 4];
			size_t m = 0;
			words[m++] = spv_type_vector(w, elemType, n);
			words[m++] = spv_id(w);
			words[m++] = v;
			words[m++] = v;
			for (uint32 i = 0; i < n; i++)
				words[m++] = sel[i];
			spv_inst(w, SPV_SEC_BODY, SpvOpVectorShuffle, words, m);
			return words[1];
		}
		uint32 Construct(uint32 type, const uint32* parts, uint32 n)
		{
			uint32 words[2 + 4];
			size_t m = 0;
			words[m++] = type;
			words[m++] = spv_id(w);
			for (uint32 i = 0; i < n; i++)
				words[m++] = parts[i];
			EnsureBlock();
			spv_inst(w, SPV_SEC_BODY, SpvOpCompositeConstruct, words, m);
			return words[1];
		}
		// attrDecoder.<chans> read; written, one channel through its access
		// chain, more as a swizzle (loaded, inserted, stored), all whole
		uint32 DecRead(const char* chans)
		{
			uint8 sel[4];
			uint32 n = 0;
			for (; chans[n]; n++)
				sel[n] = (uint8)(chans[n] == 'w' ? 3 : chans[n] - 'x');
			uint32 v = Load(tV4U, decVar);
			if (n == 4 && sel[0] == 0 && sel[1] == 1 && sel[2] == 2 && sel[3] == 3)
				return v;
			return Swizzle(v, tU, sel, n);
		}
		void DecWrite(const char* chans, uint32 value, bool whole = false)
		{
			uint8 sel[4];
			uint32 n = 0;
			for (; chans[n]; n++)
				sel[n] = (uint8)(chans[n] == 'w' ? 3 : chans[n] - 'x');
			if (whole)
			{
				Store(decVar, value);
				return;
			}
			if (n == 1)
			{
				uint32 ptr = spv_type_pointer(w, SpvStorageClassFunction, tU);
				Store(Op(SpvOpAccessChain, ptr, { decVar, ConstI(sel[0]) }), value);
				return;
			}
			uint32 reg = Load(tV4U, decVar);
			for (uint32 i = 0; i < n; i++)
				reg = Op(SpvOpCompositeInsert, tV4U, { Op(SpvOpCompositeExtract, tU, { value, i }), reg, sel[i] });
			Store(decVar, reg);
		}
		// attrDataSem<n>.<chans>
		uint32 AttrRead(uint32 semantic, const char* chans)
		{
			uint8 sel[4];
			uint32 n = 0;
			for (; chans[n]; n++)
				sel[n] = (uint8)(chans[n] == 'w' ? 3 : chans[n] - 'x');
			uint32 v = Load(tV4U, attrVar[semantic]);
			if (n == 4 && sel[0] == 0 && sel[1] == 1 && sel[2] == 2 && sel[3] == 3)
				return v;
			return Swizzle(v, tU, sel, n);
		}
		// uvec<n>(v, 0, ...): the components and zeros
		uint32 PadU(uint32 v, uint32 have, uint32 n)
		{
			uint32 parts[4];
			for (uint32 i = 0; i < have; i++)
				parts[i] = have == 1 ? v : Op(SpvOpCompositeExtract, tU, { v, i });
			for (uint32 i = have; i < n; i++)
				parts[i] = ConstU(0);
			return Construct(VecU(n), parts, n);
		}
		// vecN(uvecN) / vecN(ivecN): a constructor, each component converted
		uint32 ToFloat(uint32 v, uint32 n, bool fromSigned)
		{
			if (n == 1)
				return Op(fromSigned ? SpvOpConvertSToF : SpvOpConvertUToF, tF, { v });
			uint32 parts[4];
			for (uint32 i = 0; i < n; i++)
				parts[i] = Op(fromSigned ? SpvOpConvertSToF : SpvOpConvertUToF, tF, { Op(SpvOpCompositeExtract, fromSigned ? tI : tU, { v, i }) });
			return Construct(VecF(n), parts, n);
		}
		uint32 BitcastU(uint32 v, uint32 n) { return Op(SpvOpBitcast, VecU(n), { v }); }
		// X = (X>>24)|((X>>8)&0xFF00)|((X<<8)&0xFF0000)|((X<<24)) on attrDecoder.<chans>
		void DecSwap32(const char* chans, bool whole)
		{
			uint32 n = (uint32)std::strlen(chans);
			uint32 t = VecU(n);
			uint32 a = Op(SpvOpShiftRightLogical, t, { DecRead(chans), SplatU(24, n) });
			uint32 b = Op(SpvOpBitwiseAnd, t, { Op(SpvOpShiftRightLogical, t, { DecRead(chans), SplatU(8, n) }), SplatU(0xFF00, n) });
			uint32 ab = Op(SpvOpBitwiseOr, t, { a, b });
			uint32 c = Op(SpvOpBitwiseAnd, t, { Op(SpvOpShiftLeftLogical, t, { DecRead(chans), SplatU(8, n) }), SplatU(0xFF0000, n) });
			uint32 abc = Op(SpvOpBitwiseOr, t, { ab, c });
			uint32 d = Op(SpvOpShiftLeftLogical, t, { DecRead(chans), SplatU(24, n) });
			DecWrite(chans, Op(SpvOpBitwiseOr, t, { abc, d }), whole);
		}
		// X = ((X>>8)&0xFF)|((X<<8)&0xFF00)
		void DecSwap16(const char* chans, bool whole)
		{
			uint32 n = (uint32)std::strlen(chans);
			uint32 t = VecU(n);
			uint32 a = Op(SpvOpBitwiseAnd, t, { Op(SpvOpShiftRightLogical, t, { DecRead(chans), SplatU(8, n) }), SplatU(0xFF, n) });
			uint32 b = Op(SpvOpBitwiseAnd, t, { Op(SpvOpShiftLeftLogical, t, { DecRead(chans), SplatU(8, n) }), SplatU(0xFF00, n) });
			DecWrite(chans, Op(SpvOpBitwiseOr, t, { a, b }), whole);
		}
		// if( (attrDecoder.c&bit) != 0 ) attrDecoder.c |= ext;
		void DecSignExtend(const char* c, uint32 bit, uint32 ext)
		{
			uint32 masked = Op(SpvOpBitwiseAnd, tU, { DecRead(c), ConstU(bit) });
			uint32 set = Op(SpvOpINotEqual, tBool, { masked, ConstU(0) });
			If(set, [&] { DecWrite(c, Op(SpvOpBitwiseOr, tU, { DecRead(c), ConstU(ext) })); });
		}
		// attrDecoder.c = floatBitsToUint(max(float(int(attrDecoder.c))/d,-1.0));
		void DecSNorm(const char* c, const char* d)
		{
			uint32 f = Op(SpvOpConvertSToF, tF, { Op(SpvOpBitcast, tI, { DecRead(c) }) });
			uint32 q = Op(SpvOpFDiv, tF, { f, KFText(d).id });
			uint32 m = Ext(GLSLstd450FMax, tF, { q, KF(-1.0).id });
			DecWrite(c, Op(SpvOpBitcast, tU, { m }));
		}
		// attrDecoder.c = floatBitsToUint(float(int(attrDecoder.c))[/d]);
		void DecIntToFloat(const char* c, const char* d)
		{
			uint32 f = Op(SpvOpConvertSToF, tF, { Op(SpvOpBitcast, tI, { DecRead(c) }) });
			if (d)
				f = Op(SpvOpFDiv, tF, { f, KFText(d).id });
			DecWrite(c, Op(SpvOpBitcast, tU, { f }));
		}
		// the 2_10_10_10 split, then x, y, z normalized by d and w by 3
		void Dec1010102(const char* d, bool isSigned)
		{
			static const uint32 shifts[4] = { 0, 10, 20, 30 };
			uint32 parts[4];
			for (uint32 i = 0; i < 4; i++)
				parts[i] = Op(SpvOpBitwiseAnd, tU, { Op(SpvOpShiftRightLogical, tU, { DecRead("x"), ConstU(shifts[i]) }), ConstU(i == 3 ? 0x3 : 0x3FF) });
			DecWrite("xyzw", Construct(tV4U, parts, 4), true);
			if (isSigned)
			{
				DecSignExtend("x", 0x200, 0xFFFFFC00);
				DecSignExtend("y", 0x200, 0xFFFFFC00);
				DecSignExtend("z", 0x200, 0xFFFFFC00);
			}
			DecSNorm("x", d);
			DecSNorm("y", d);
			DecSNorm("z", d);
			// floatBitsToUint(float(attrDecoder.w) * (1.0 / 3.0))
			uint32 f = Op(SpvOpConvertUToF, tF, { DecRead("w") });
			DecWrite("w", Op(SpvOpBitcast, tU, { Op(SpvOpFMul, tF, { f, KF(1.0 / 3.0).id }) }));
		}
		// unpackHalf2x16(attrDecoder.a|(attrDecoder.b<<16))
		uint32 DecUnpackHalf(const char* a, const char* b)
		{
			uint32 lo = DecRead(a);
			uint32 hi = Op(SpvOpShiftLeftLogical, tU, { DecRead(b), ConstU(16) });
			return Ext(GLSLstd450UnpackHalf2x16, tV2F, { Op(SpvOpBitwiseOr, tU, { lo, hi }) });
		}
		bool DeclareTextures();
		bool EmitTEXClause(LatteDecompilerCFInstruction& cf);
		bool EmitSample(LatteDecompilerTEXInstruction& t);
		bool EmitShadowSample(LatteDecompilerTEXInstruction& t);
		bool EmitGather(LatteDecompilerTEXInstruction& t);
		bool EmitTexelFetch(LatteDecompilerTEXInstruction& t);
		bool EmitResInfo(LatteDecompilerTEXInstruction& t);
		bool EmitCompTexLod(LatteDecompilerTEXInstruction& t);
		bool EmitGradSample(LatteDecompilerTEXInstruction& t);
		bool EmitGetGradients(LatteDecompilerTEXInstruction& t);
		bool EmitSetGradients(LatteDecompilerTEXInstruction& t);
		uint32 TexGPRAccess1(LatteDecompilerTEXInstruction& t, sint32 component);
		Val TexCoord(LatteDecompilerTEXInstruction& t, sint32 component);
		Val TexCoordAs(LatteDecompilerTEXInstruction& t, sint32 component, sint32 type);
		uint32 TexGPRAccess(LatteDecompilerTEXInstruction& t, const uint8* sel, uint32 n);
		uint32 CoordVector(const Val* c, uint32 n);
		uint32 ImageOp(uint32 opcode, uint32 resultType, uint32 image, uint32 coord, uint32 extra, uint32 mask, const uint32* operands, uint32 nOperands);
		uint32 TexOffset(LatteDecompilerTEXInstruction& t, Latte::E_DIM dim);
		uint32 TexWritten(LatteDecompilerTEXInstruction& t, uint32 written[4]);
		bool StoreTexResult(LatteDecompilerTEXInstruction& t, uint32 v, sint32 sampleType, uint32 count, const uint32* written);
		uint32 TexSwizzle(LatteDecompilerTEXInstruction& t, uint32 v, uint32 elem, uint32 count, const uint32* written, const uint8* order);
		void StoreTexChannel(LatteDecompilerTEXInstruction& t, uint32 chan, Val v);

		bool Setup();
		bool DeclareUniforms(LatteDecompilerOutputUniformOffsets& offsets, RendererAPI api, bool declare);
		bool DeclareIO();
		bool EmitInputs();
		bool EmitClause(LatteDecompilerCFInstruction& cf);
		bool EmitALUClause(LatteDecompilerCFInstruction& cf);
		bool EmitExport(LatteDecompilerCFInstruction& cf);
		Val Operand(LatteDecompilerALUInstruction& inst, sint32 index, sint32 requiredType);
		Val UniformOperand(LatteDecompilerALUInstruction& inst, sint32 index, sint32 requiredType);
		bool EmitMova(LatteDecompilerALUInstruction& inst);
		uint32 ARChannel(uint32 indexMode);
		uint32 RelativeGPRChain(uint32 base, uint32 from, uint32 chan, uint32 indexMode);
		uint32 RelativeGPR(uint32 base, uint32 indexMode);
		Val RegisterRead(LatteDecompilerALUInstruction& inst, sint32 index);
		Val PVPSRead(LatteDecompilerALUInstruction& inst, uint8 unit);
		uint32 OutputPtr(LatteDecompilerALUInstruction& inst);
		uint32 PVPSOutputPtr(LatteDecompilerALUInstruction& inst);
		uint32 ReadOutput(LatteDecompilerALUInstruction& inst);
		void Assign(LatteDecompilerALUInstruction& inst, Val v);
		bool EmitOP2(LatteDecompilerCFInstruction& cf, LatteDecompilerALUInstruction& inst);
		bool EmitOP3(LatteDecompilerALUInstruction& inst);
		bool EmitReduction(LatteDecompilerALUInstruction* redc[4]);
		bool EmitCube(LatteDecompilerALUInstruction* redc[4]);
		bool EmitSetCubeIndex(LatteDecompilerTEXInstruction& t);
		uint32 CubeCoord(LatteDecompilerTEXInstruction& t);
		void TrackGroupOutputPVPS(LatteDecompilerALUInstruction* group, size_t count);
		bool CreateGPRTemporaries(LatteDecompilerALUInstruction* group, size_t count);
		sint32 TemporaryFor(uint8 gpr, uint8 chan) const;
		Val ExportComponent(LatteDecompilerCFInstruction& cf, uint32 burst, uint32 i);
	};

	bool SpvShader::Setup()
	{
		tVoid = spv_type_void(w);
		tBool = spv_type_bool(w);
		tI = spv_type_int(w, 1);
		tU = spv_type_int(w, 0);
		tF = spv_type_float(w);
		tV4F = spv_type_vector(w, tF, 4);
		tV4I = spv_type_vector(w, tI, 4);
		tV4U = spv_type_vector(w, tU, 4);
		tV2U = spv_type_vector(w, tU, 2);
		tV3U = spv_type_vector(w, tU, 3);
		tV2F = spv_type_vector(w, tF, 2);
		uint32 cap = SpvCapabilityShader;
		spv_inst(w, SPV_SEC_CAPABILITY, SpvOpCapability, &cap, 1);
		glsl = spv_id(w);
		spv_inst_str(w, SPV_SEC_IMPORT, SpvOpExtInstImport, &glsl, 1, "GLSL.std.450", nullptr, 0);
		uint32 model[2] = { SpvAddressingModelLogical, SpvMemoryModelGLSL450 };
		spv_inst(w, SPV_SEC_MEMORY_MODEL, SpvOpMemoryModel, model, 2);
		return true;
	}

	// _emitUniformVariables: the members and their offsets, the Vulkan
	// block declared when asked
	bool SpvShader::DeclareUniforms(LatteDecompilerOutputUniformOffsets& uniformOffsets, RendererAPI api, bool declare)
	{
		auto* dc = ctx;
		uint32 memberTypes[16 + LATTE_NUM_MAX_TEX_UNITS];
		uint32 memberOffsets[16 + LATTE_NUM_MAX_TEX_UNITS];
		uint32 memberCount = 0;
		uint32 uniformCurrentOffset = 0;
		if (shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_REMAPPED)
		{
			uint32 count = (uint32)shader->list_remappedUniformEntries.size();
			if (count == 0)
				return Decline("empty remapped uniform array");
			uint32 arr = spv_type_array(w, tV4I, ConstI((sint32)count));
			if (declare)
				Decorate(arr, SpvDecorationArrayStride, { 16 });
			memberTypes[memberCount] = arr;
			memberOffsets[memberCount++] = uniformCurrentOffset;
			uniformOffsets.offset_remapped = uniformCurrentOffset;
			uniformCurrentOffset += 16 * count;
		}
		else if (shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CFILE)
		{
			uint32 cfileSize = dc->analyzer.uniformRegisterAccessTracker.DetermineSize(dc->shaderBaseHash, 256);
			uint32 arr = spv_type_array(w, tV4I, ConstI((sint32)cfileSize));
			if (declare)
				Decorate(arr, SpvDecorationArrayStride, { 16 });
			memberTypes[memberCount] = arr;
			memberOffsets[memberCount++] = uniformCurrentOffset;
			uniformOffsets.offset_uniformRegister = uniformCurrentOffset;
			uniformOffsets.count_uniformRegister = cfileSize;
			uniformCurrentOffset += 16 * cfileSize;
		}
		else if (shader->uniformMode != LATTE_DECOMPILER_UNIFORM_MODE_NONE && shader->uniformMode != LATTE_DECOMPILER_UNIFORM_MODE_FULL_CBANK)
			return Decline("uniform mode");
		// special uniforms
		const bool hasAnyViewportScaleDisabled =
			!dc->contextRegistersNew->PA_CL_VTE_CNTL.get_VPORT_X_SCALE_ENA() ||
			!dc->contextRegistersNew->PA_CL_VTE_CNTL.get_VPORT_Y_SCALE_ENA() ||
			!dc->contextRegistersNew->PA_CL_VTE_CNTL.get_VPORT_Z_SCALE_ENA();
		if (isVertex && hasAnyViewportScaleDisabled)
		{
			// uf_windowSpaceToClipSpaceTransform (GX2 special state 0)
			uniformCurrentOffset = (uniformCurrentOffset + 7) & ~7;
			windowMember = (sint32)memberCount;
			memberTypes[memberCount] = spv_type_vector(w, tF, 2);
			memberOffsets[memberCount++] = uniformCurrentOffset;
			uniformOffsets.offset_windowSpaceToClipSpaceTransform = uniformCurrentOffset;
			uniformCurrentOffset += 8;
		}
		bool alphaTestEnable = !NotPixel() && dc->contextRegistersNew->SX_ALPHA_TEST_CONTROL.get_ALPHA_TEST_ENABLE();
		if (alphaTestEnable)
		{
			uniformCurrentOffset = (uniformCurrentOffset + 3) & ~3;
			alphaRefMember = (sint32)memberCount;
			memberTypes[memberCount] = tF;
			memberOffsets[memberCount++] = uniformCurrentOffset;
			uniformOffsets.offset_alphaTestRef = uniformCurrentOffset;
			uniformCurrentOffset += 4;
		}
		if (((isVertex && !usesGS) || isGeometry) && dc->analyzer.outputPointSize && !dc->analyzer.writesPointSize)
		{
			// uf_pointSize, the render state's, where the shader writes none
			uniformCurrentOffset = (uniformCurrentOffset + 3) & ~3;
			pointSizeMember = (sint32)memberCount;
			memberTypes[memberCount] = tF;
			memberOffsets[memberCount++] = uniformCurrentOffset;
			uniformOffsets.offset_pointSize = uniformCurrentOffset;
			uniformCurrentOffset += 4;
		}
		if (!NotPixel() && dc->analyzer.hasFragCoordAccess)
			return Decline("gl_FragCoord");
		// uf_tex<n>Scale for each unit fetched from by texel, the first
		// after the padding where uf_fragCoordScale was
		bool fragCoordPadding = !NotPixel();
		for (sint32 t = 0; t < LATTE_NUM_MAX_TEX_UNITS; t++)
		{
			texScaleMember[t] = -1;
			if (!dc->analyzer.texUnitUsesTexelCoordinates.test(t))
				continue;
			if (NotPixel())
				return Decline("texel coordinates in a vertex shader");
			if (fragCoordPadding)
			{
				const bool gl = api == RendererAPI::OpenGL;
				uniformCurrentOffset = gl ? (uniformCurrentOffset + 7) & ~7 : (uniformCurrentOffset + 15) & ~15;
				memberTypes[memberCount] = gl ? tV2F : tV4F;
				memberOffsets[memberCount++] = uniformCurrentOffset;
				uniformCurrentOffset += gl ? 8 : 16;
				fragCoordPadding = false;
			}
			uniformCurrentOffset = (uniformCurrentOffset + 7) & ~7;
			texScaleMember[t] = (sint32)memberCount;
			memberTypes[memberCount] = tV2F;
			memberOffsets[memberCount++] = uniformCurrentOffset;
			uniformOffsets.offset_texScale[t] = uniformCurrentOffset;
			uniformCurrentOffset += 8;
		}
		// uf_verticesPerInstance (and the streamout bases): a geometry
		// shader's always, as the GLSL emitter's condition reads
		if ((dc->analyzer.useSSBOForStreamout && isVertex && !usesGS) || isGeometry)
		{
			verticesPerInstanceMember = (sint32)memberCount;
			memberTypes[memberCount] = tI;
			memberOffsets[memberCount++] = uniformCurrentOffset;
			uniformOffsets.offset_verticesPerInstance = uniformCurrentOffset;
			uniformCurrentOffset += 4;
			for (uint32 i = 0; i < LATTE_NUM_STREAMOUT_BUFFER; i++)
			{
				if (!dc->output->streamoutBufferWriteMask[i])
					continue;
				sbBaseMember[i] = (sint32)memberCount;
				memberTypes[memberCount] = tI;
				memberOffsets[memberCount++] = uniformCurrentOffset;
				uniformOffsets.offset_streamoutBufferBase[i] = uniformCurrentOffset;
				uniformCurrentOffset += 4;
			}
		}
		uniformOffsets.offset_endOfBlock = uniformCurrentOffset;
		if (!declare || api != RendererAPI::Vulkan || !dc->hasUniformVarBlock)
			return true;
		uint32 block = spv_type_struct(w, memberTypes, memberCount);
		Decorate(block, SpvDecorationBlock);
		for (uint32 i = 0; i < memberCount; i++)
		{
			uint32 words[4] = { block, i, SpvDecorationOffset, memberOffsets[i] };
			spv_inst(w, SPV_SEC_DECORATION, SpvOpMemberDecorate, words, 4);
		}
		uint32 ptr = spv_type_pointer(w, SpvStorageClassUniform, block);
		ufBlockVar = spv_global_variable(w, ptr, SpvStorageClassUniform);
		Decorate(ufBlockVar, SpvDecorationDescriptorSet, { (uint32)dc->output->resourceMappingVK.setIndex });
		Decorate(ufBlockVar, SpvDecorationBinding, { (uint32)dc->output->resourceMappingVK.uniformVarsBufferBindingPoint });
		for (uint32 i = 0; i < memberCount; i++)
			ufMember[i] = memberTypes[i];
		return true;
	}

	// _emitPSImports and the pixel outputs
	bool SpvShader::DeclareIO()
	{
		LatteShaderPSInputTable* psInputTable = LatteSHRC_GetPSInputTable();
		uint32 ptrIn = spv_type_pointer(w, SpvStorageClassInput, tV4F);
		for (sint32 i = 0; i < psInputTable->count; i++)
		{
			if (psInputTable->import[i].semanticId > LATTE_ANALYZER_IMPORT_INDEX_PARAM_MAX)
				continue;
			// passG2PParameter<n> at the semantic's location after a geometry
			// shader, passParameterSem<n> at the input's index otherwise
			const uint32 location = usesGS ? (psInputTable->import[i].semanticId & 0x7F) : (uint32)i;
			for (sint32 j = 0; usesGS && j < i; j++)
			{
				if (inputVar[j] && (psInputTable->import[j].semanticId & 0x7F) == location)
					return Decline("parameter declared twice");
			}
			uint32 v = spv_global_variable(w, ptrIn, SpvStorageClassInput);
			Decorate(v, SpvDecorationLocation, { location });
			if (psInputTable->import[i].isFlat)
				Decorate(v, SpvDecorationFlat);
			if (psInputTable->import[i].isNoPerspective)
				Decorate(v, SpvDecorationNoPerspective);
			inputVar[i] = v;
			interfaceIds[interfaceCount++] = v;
		}
		uint32 ptrOut = spv_type_pointer(w, SpvStorageClassOutput, tV4F);
		for (uint32 i = 0; i < LATTE_NUM_COLOR_TARGET; i++)
		{
			if ((shader->pixelColorOutputMask & (1 << i)) == 0)
				continue;
			uint32 v = spv_global_variable(w, ptrOut, SpvStorageClassOutput);
			Decorate(v, SpvDecorationLocation, { i });
			colorOut[i] = v;
			interfaceIds[interfaceCount++] = v;
		}
		return true;
	}

	// the pixel shader's inputs into its registers
	bool SpvShader::EmitInputs()
	{
		LatteShaderPSInputTable* psInputTable = LatteSHRC_GetPSInputTable();
		uint32 psControl1 = ctx->contextRegisters[mmSPI_PS_IN_CONTROL_1];
		uint32 spiInterpControl = ctx->contextRegisters[mmSPI_INTERP_CONTROL_0];
		if ((spiInterpControl >> 1) & 1)
			return Decline("sprites");
		if (psInputTable->paramGen != 0)
			return Decline("param gen");
		if ((psControl1 >> 8) & 1)
			return Decline("front face");
		for (sint32 i = 0; i < psInputTable->count; i++)
		{
			sint32 gprIndex = i;
			if ((ctx->analyzer.gprUseMask[gprIndex / 8] & (1 << (gprIndex % 8))) == 0 && ctx->analyzer.usesRelativeGPRRead == false)
				continue;
			uint32 semanticId = psInputTable->import[i].semanticId;
			if (semanticId == LATTE_ANALYZER_IMPORT_INDEX_SPIPOSITION)
				return Decline("gl_FragCoord");
			if (!inputVar[i])
				return Decline("input without a declaration");
			uint32 v = Load(tV4F, inputVar[i]);
			if (dt == DT_I)
				v = Op(SpvOpBitcast, tV4I, { v });
			Store(GPR(gprIndex), v);
		}
		return true;
	}

	sint32 SpvShader::TemporaryFor(uint8 gpr, uint8 chan) const
	{
		for (uint32 i = 0; i < temporaryCount; i++)
		{
			if (temporaries[i].gpr == gpr && temporaries[i].chan == chan)
				return temporaries[i].var;
		}
		return -1;
	}

	void SpvShader::TrackGroupOutputPVPS(LatteDecompilerALUInstruction* group, size_t count)
	{
		for (auto& it : pvps)
			it.location = PVPSAlias::NONE;
		for (size_t i = 0; i < count; i++)
		{
			LatteDecompilerALUInstruction& inst = group[i];
			if (!inst.isOP3 && inst.opcode == ALU_OP2_INST_NOP)
				continue;
			if (inst.writeMask == 0)
			{
				pvps[inst.aluUnit].location = PVPSAlias::PVPS;
				pvps[inst.aluUnit].index = group->instructionGroupIndex & 1;
				pvps[inst.aluUnit].unit = inst.aluUnit;
			}
			else if (inst.destRel == 0)
			{
				pvps[inst.aluUnit].location = PVPSAlias::GPR;
				pvps[inst.aluUnit].index = (uint8)inst.destGpr;
				pvps[inst.aluUnit].unit = inst.destElem;
			}
		}
	}

	// GPR channels read in a group after an earlier instruction of it wrote
	// them, copied first (the GLSL emitter's checks, as they are)
	bool SpvShader::CreateGPRTemporaries(LatteDecompilerALUInstruction* group, size_t count)
	{
		uint8 writeMask[(LATTE_NUM_GPR * 4 + 7) / 8] = { 0 };
		temporaryCount = 0;
		for (size_t g = 0; g < count; g++)
		{
			LatteDecompilerALUInstruction& inst = group[g];
			if (inst.isOP3 == false && inst.opcode == ALU_OP2_INST_NOP)
				continue;
			for (sint32 f = 0; f < 3; f++)
			{
				uint32 readGPR, readChan;
				if (GPU7_ALU_SRC_IS_GPR(inst.sourceOperand[f].sel))
				{
					readGPR = GPU7_ALU_SRC_GET_GPR_INDEX(inst.sourceOperand[f].sel);
					readChan = inst.sourceOperand[f].chan;
				}
				else if (GPU7_ALU_SRC_IS_PV(inst.sourceOperand[f].sel) || GPU7_ALU_SRC_IS_PS(inst.sourceOperand[f].sel))
				{
					uint8 unit = GPU7_ALU_SRC_IS_PV(inst.sourceOperand[f].sel) ? inst.sourceOperand[f].chan : 4;
					if (pvps[unit].location != PVPSAlias::GPR)
						continue;
					readGPR = pvps[unit].index;
					readChan = pvps[unit].unit;
				}
				else
					continue;
				uint32 bit = readGPR * 4 + inst.sourceOperand[f].chan;
				if ((writeMask[bit / 8] & (1 << (bit % 8))) != 0)
				{
					if (TemporaryFor((uint8)readGPR, (uint8)readChan) < 0)
					{
						if (temporaryCount >= 5)
							return Decline("more than five register backups");
						temporaries[temporaryCount] = { (uint8)readGPR, (uint8)readChan, (uint8)temporaryCount };
						temporaryCount++;
					}
				}
			}
			if (inst.writeMask != 0)
			{
				uint32 bit = inst.destGpr * 4 + inst.destElem;
				writeMask[bit / 8] |= (1 << (bit % 8));
			}
		}
		for (uint32 i = 0; i < temporaryCount; i++)
		{
			uint32 v = ReadChannel(GPR(temporaries[i].gpr), temporaries[i].chan);
			Store(backupVar[temporaries[i].var], v);
		}
		return true;
	}

	Val SpvShader::RegisterRead(LatteDecompilerALUInstruction& inst, sint32 index)
	{
		uint32 gpr = GPU7_ALU_SRC_GET_GPR_INDEX(inst.sourceOperand[index].sel);
		sint32 temp = TemporaryFor((uint8)gpr, inst.sourceOperand[index].chan);
		if (temp >= 0)
			return Leaf(Load(TypeOf(dt), backupVar[temp]));
		return { ReadChannel(GPR(gpr), inst.sourceOperand[index].chan), dt };
	}

	Val SpvShader::PVPSRead(LatteDecompilerALUInstruction& inst, uint8 unit)
	{
		switch (pvps[unit].location)
		{
		case PVPSAlias::GPR:
		{
			sint32 temp = TemporaryFor(pvps[unit].index, pvps[unit].unit);
			if (temp >= 0)
				return Leaf(Load(TypeOf(dt), backupVar[temp]));
			return { ReadChannel(GPR(pvps[unit].index), pvps[unit].unit), dt };
		}
		case PVPSAlias::PVPS:
		{
			uint32 parity = (inst.instructionGroupIndex - 1) & 1;
			uint32 var = pvps[unit].unit == 4 ? psVar[parity] : pvVar[parity][pvps[unit].unit];
			return Leaf(Load(TypeOf(dt), var));
		}
		default:
			Decline("PV/PS read before written");
			return { ConstI(0), DT_I };
		}
	}

	Val SpvShader::UniformOperand(LatteDecompilerALUInstruction& inst, sint32 index, sint32 requiredType)
	{
		auto& op = inst.sourceOperand[index];
		bool isRegister = GPU7_ALU_SRC_IS_CFILE(op.sel);
		if (op.rel != 0 && !(isRegister && shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CFILE) &&
			shader->uniformMode != LATTE_DECOMPILER_UNIFORM_MODE_FULL_CBANK)
		{
			Decline("relative uniform access");
			return { ConstI(0), requiredType };
		}
		sint32 uniformOffset;
		sint32 bufferIndex = 0;
		if (isRegister)
			uniformOffset = GPU7_ALU_SRC_GET_CFILE_INDEX(op.sel);
		else if (GPU7_ALU_SRC_IS_CBANK0(op.sel))
		{
			uniformOffset = GPU7_ALU_SRC_GET_CBANK0_INDEX(op.sel) + inst.cfInstruction->cBank0AddrBase;
			bufferIndex = inst.cfInstruction->cBank0Index;
		}
		else
		{
			uniformOffset = GPU7_ALU_SRC_GET_CBANK1_INDEX(op.sel) + inst.cfInstruction->cBank1AddrBase;
			bufferIndex = inst.cfInstruction->cBank1Index;
		}
		if (shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CBANK)
		{
			// uf_block<stage><bank's buffer>[offset or ARi.<c>+offset].<c>, a
			// float, converted to the type wanted
			if (isRegister)
			{
				Decline("uniform register in buffer mode");
				return { ConstI(0), requiredType };
			}
			if (bufferIndex >= LATTE_NUM_MAX_UNIFORM_BUFFERS || !ubVar[bufferIndex])
			{
				Decline("uniform buffer not declared");
				return { ConstI(0), requiredType };
			}
			uint32 at = op.rel != 0 ? Op(SpvOpIAdd, tI, { ARChannel(inst.indexMode), ConstI(uniformOffset) }) : ConstI(uniformOffset);
			if (!Ok())
				return { ConstI(0), requiredType };
			uint32 vec = Load(tV4F, Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassUniform, tV4F), { ubVar[bufferIndex], ConstI(0), at }));
			return Convert({ Op(SpvOpCompositeExtract, tF, { vec, (uint32)op.chan }), DT_F }, requiredType);
		}
		sint32 arrayIndex;
		if (shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_REMAPPED)
		{
			const LatteDecompilerRemappedUniformEntry_t* entry = nullptr;
			for (auto& it : shader->list_remappedUniformEntries)
			{
				if (it.isRegister && isRegister)
				{
					if (it.index == (uint32)uniformOffset)
					{
						entry = &it;
						break;
					}
				}
				else if (!it.isRegister && !isRegister && it.kcacheBankId == (uint32)bufferIndex && it.index == (uint32)uniformOffset)
				{
					entry = &it;
					break;
				}
			}
			if (!entry)
			{
				Decline("uniform not remapped");
				return { ConstI(0), requiredType };
			}
			arrayIndex = (sint32)entry->mappedIndex;
		}
		else if (shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CFILE && isRegister)
			arrayIndex = uniformOffset;
		else
		{
			Decline("uniform mode");
			return { ConstI(0), requiredType };
		}
		// as rslc reads uf_remappedPS[i].x: the ivec4 loaded, the channel
		// extracted (a driver's optimizer sees the two differently)
		uint32 ptr = spv_type_pointer(w, SpvStorageClassUniform, tV4I);
		// uf_uniformRegister<stage>[ARi.<c>+offset]: the sum not folded
		uint32 at = op.rel != 0 ? Op(SpvOpIAdd, tI, { ARChannel(inst.indexMode), ConstI(arrayIndex) }) : ConstI(arrayIndex);
		if (!Ok())
			return { ConstI(0), requiredType };
		uint32 p = Op(SpvOpAccessChain, ptr, { ufBlockVar, ConstI(0), at });
		uint32 v = Load(tV4I, p);
		return Convert({ Op(SpvOpCompositeExtract, tI, { v, (uint32)op.chan }), DT_I }, requiredType);
	}

	// ARi.x to ARi.w by the index mode, as the GLSL reads it: ARi loaded,
	// the channel extracted
	uint32 SpvShader::ARChannel(uint32 indexMode)
	{
		if (indexMode > GPU7_INDEX_AR_W)
		{
			Decline("index mode");
			return ConstI(0);
		}
		return Op(SpvOpCompositeExtract, tI, { Load(tV4I, ariVar), indexMode });
	}

	// (ARi.c==i-base)?R<i>i.chan:... for each register from 'from' the
	// analyzer marks used, then 0: rslc's ternaries of non-leaf sides, a
	// branch each way and a phi, nested in the else
	uint32 SpvShader::RelativeGPRChain(uint32 base, uint32 from, uint32 chan, uint32 indexMode)
	{
		uint32 i = from;
		while (i < LATTE_NUM_GPR && (ctx->analyzer.gprUseMask[i / 8] & (1 << (i % 8))) == 0)
			i++;
		if (i >= LATTE_NUM_GPR)
			return ConstI(0);
		uint32 cond = Op(SpvOpIEqual, tBool, { ARChannel(indexMode), ConstI((sint32)(i - base)) });
		uint32 thenL = spv_id(w), elseL = spv_id(w), merge = spv_id(w);
		OpVoid(SpvOpSelectionMerge, { merge, SpvSelectionControlMaskNone });
		OpVoid(SpvOpBranchConditional, { cond, thenL, elseL });
		Label(thenL);
		uint32 a = Op(SpvOpCompositeExtract, tI, { Load(tV4I, GPR(i)), chan });
		uint32 aEnd = curBlock;
		Branch(merge);
		Label(elseL);
		uint32 b = RelativeGPRChain(base, i + 1, chan, indexMode);
		EnsureBlock();
		uint32 bEnd = curBlock;
		Branch(merge);
		Label(merge);
		return Op(SpvOpPhi, tI, { a, aEnd, b, bEnd });
	}

	// R[base+ARi.c]: the register array's element at the sum
	uint32 SpvShader::RelativeGPR(uint32 base, uint32 indexMode)
	{
		if (!gprArrayVar)
		{
			Decline("relative register without the register array");
			return GPR(base);
		}
		uint32 at = Op(SpvOpIAdd, tI, { ConstI((sint32)base), ARChannel(indexMode) });
		return Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassFunction, Vec4Of(dt)), { gprArrayVar, at });
	}

	// MOVA_FLOOR: floor, -256 outside -256..255, into ARi.c as an int, the
	// float's bits out; MOVA_INT: the int the same way
	bool SpvShader::EmitMova(LatteDecompilerALUInstruction& inst)
	{
		if (dt != DT_I)
			return Decline("MOVA of a float shader"); // its int output a GLSL error
		if (inst.destElem > 3)
			return Decline("MOVA channel");
		if (inst.opcode == ALU_OP2_INST_MOVA_FLOOR)
		{
			Store(tempResultFVar, Operand(inst, 0, DT_F).id);
			Store(tempResultFVar, Ext(GLSLstd450Floor, tF, { Load(tF, tempResultFVar) }));
			uint32 lo = Op(SpvOpFOrdGreaterThanEqual, tBool, { Load(tF, tempResultFVar), ConstF(-256.0f) });
			uint32 hi = Op(SpvOpFOrdLessThanEqual, tBool, { Load(tF, tempResultFVar), ConstF(255.0f) });
			uint32 in = Op(SpvOpLogicalAnd, tBool, { lo, hi });
			Store(tempResultFVar, Op(SpvOpSelect, tF, { in, Load(tF, tempResultFVar), ConstF(-256.0f) }));
			Store(Channel(ariVar, DT_I, inst.destElem), Op(SpvOpConvertFToS, tI, { Load(tF, tempResultFVar) }));
			Assign(inst, Run(Op(SpvOpBitcast, tI, { Load(tF, tempResultFVar) }), DT_I));
		}
		else
		{
			Store(tempResultI, Operand(inst, 0, DT_I).id);
			uint32 lo = Op(SpvOpSGreaterThanEqual, tBool, { Load(tI, tempResultI), ConstI(-256) });
			uint32 hi = Op(SpvOpSLessThanEqual, tBool, { Load(tI, tempResultI), ConstI(255) });
			uint32 in = Op(SpvOpLogicalAnd, tBool, { lo, hi });
			Store(tempResultI, Op(SpvOpSelect, tI, { in, Load(tI, tempResultI), ConstI(-256) }));
			Store(Channel(ariVar, DT_I, inst.destElem), Load(tI, tempResultI));
			Assign(inst, Run(Load(tI, tempResultI), DT_I));
		}
		return Ok();
	}

	// a literal as the GLSL emitter writes it: the shortest decimal that
	// reads back as the float, which rslc reads as a double
	static double LiteralAsGLSLReads(float f)
	{
		char text[64];
		size_t len = fmt::format_to_n(text, 63, "{:#}", f).size;
		if (len > 62)
			len = 62;
		if (len > 0 && text[len - 1] == '.')
			text[len++] = '0';
		text[len] = '\0';
		return std::strtod(text, nullptr);
	}

	// _emitOperandInputCode
	Val SpvShader::Operand(LatteDecompilerALUInstruction& inst, sint32 index, sint32 requiredType)
	{
		auto& op = inst.sourceOperand[index];
		sint32 requiredTypeOut = requiredType;
		if (requiredType != DT_F && (op.abs != 0 || op.neg != 0))
			requiredType = DT_F;
		Val v{};
		if (GPU7_ALU_SRC_IS_GPR(op.sel))
		{
			if (op.rel != 0 && gprArrayVar)
			{
				// _emitCodeToReadRelativeGPR of the register array: R[base+ARi.c].chan
				const uint32 base = GPU7_ALU_SRC_GET_GPR_INDEX(op.sel);
				uint32 ptr = RelativeGPR(base, inst.indexMode);
				if (!Ok())
					return KI(0);
				v = Convert(Run(ReadChannel(ptr, op.chan), dt), requiredType);
			}
			else if (op.rel != 0)
			{
				// _emitCodeToReadRelativeGPR: (ARi.c==0)?R<base>i.c:(ARi.c==1)?...:0,
				// of an int shader (a float one mixes float and int sides)
				if (dt != DT_I)
				{
					Decline("relative GPR read of a float shader");
					return KI(0);
				}
				const uint32 base = GPU7_ALU_SRC_GET_GPR_INDEX(op.sel);
				Val chain = Run(RelativeGPRChain(base, base, op.chan, inst.indexMode), DT_I);
				if (!Ok())
					return KI(0);
				v = Convert(chain, requiredType);
			}
			else
			{
				Val r = RegisterRead(inst, index);
				if (requiredType == DT_U && r.type == DT_F)
				{
					Decline("float register read as uint");
					return KU(0);
				}
				v = Convert(r, requiredType);
			}
		}
		else if (GPU7_ALU_SRC_IS_CONST_0F(op.sel))
			v = requiredType == DT_F ? KF(0.0) : K(requiredType, 0);
		else if (GPU7_ALU_SRC_IS_CONST_1F(op.sel))
			v = Convert(KF(1.0), requiredType);
		else if (GPU7_ALU_SRC_IS_CONST_0_5F(op.sel))
			v = Convert(KF(0.5), requiredType);
		else if (GPU7_ALU_SRC_IS_CONST_1I(op.sel))
		{
			if (requiredType == DT_F)
			{
				Decline("integer one read as float");
				return KF(0.0);
			}
			v = K(requiredType, 1);
		}
		else if (GPU7_ALU_SRC_IS_CONST_M1I(op.sel))
		{
			if (requiredType != DT_I)
			{
				Decline("integer minus one read as non-int");
				return KI(0);
			}
			v = KI(-1);
		}
		else if (GPU7_ALU_SRC_IS_LITERAL(op.sel))
		{
			uint32 bits = inst.literalData.w[op.chan];
			if (requiredType != DT_F)
				v = K(requiredType, bits);
			else
			{
				sint32 exponent = (sint32)((bits >> 23) & 0xFF) - 127;
				if ((bits & 0xFF) == 0 && exponent >= -10 && exponent <= 10)
				{
					float f;
					std::memcpy(&f, &bits, 4);
					v = KF(LiteralAsGLSLReads(f));
				}
				else
					v = Run(Op(SpvOpBitcast, tF, { ConstI((sint32)bits) }), DT_F); // intBitsToFloat(0x...)
			}
		}
		else if (GPU7_ALU_SRC_IS_CFILE(op.sel) || GPU7_ALU_SRC_IS_CBANK0(op.sel) || GPU7_ALU_SRC_IS_CBANK1(op.sel))
			v = UniformOperand(inst, index, requiredType);
		else if (GPU7_ALU_SRC_IS_PV(op.sel) || GPU7_ALU_SRC_IS_PS(op.sel))
		{
			if (inst.instructionGroupIndex == 0)
			{
				Decline("PV/PS in the first group");
				return KI(0);
			}
			v = Convert(PVPSRead(inst, GPU7_ALU_SRC_IS_PV(op.sel) ? op.chan : 4), requiredType);
		}
		else
		{
			Decline("operand select");
			return KI(0);
		}
		if (op.abs != 0)
			v = FUnary(GLSLstd450FAbs, v);
		if (op.neg != 0)
			v = FNeg(v);
		if (requiredTypeOut != requiredType)
			v = Convert(v, requiredTypeOut);
		return v;
	}

	uint32 SpvShader::PVPSOutputPtr(LatteDecompilerALUInstruction& inst)
	{
		uint32 parity = inst.instructionGroupIndex & 1;
		return inst.aluUnit == 4 ? psVar[parity] : pvVar[parity][inst.aluUnit];
	}

	static bool IsReduction(const LatteDecompilerALUInstruction& inst)
	{
		return inst.isOP3 == false && (inst.opcode == ALU_OP2_INST_DOT4 || inst.opcode == ALU_OP2_INST_DOT4_IEEE || inst.opcode == ALU_OP2_INST_CUBE);
	}

	// _emitInstructionOutputVariableName
	uint32 SpvShader::OutputPtr(LatteDecompilerALUInstruction& inst)
	{
		if (inst.writeMask == 0)
		{
			if (!IsReduction(inst))
				return PVPSOutputPtr(inst);
			if (inst.aluUnit > 3)
			{
				Decline("reduction on the trans unit");
				return PVPSOutputPtr(inst);
			}
			return Channel(tempVar, dt, inst.aluUnit);
		}
		if (inst.destRel != 0)
			return Channel(RelativeGPR(inst.destGpr, inst.indexMode), dt, inst.destElem);
		return Channel(GPR(inst.destGpr), dt, inst.destElem);
	}

	// what OutputPtr names, read: a PV or PS as itself, a channel of a
	// register or of temp as rslc reads a channel
	uint32 SpvShader::ReadOutput(LatteDecompilerALUInstruction& inst)
	{
		if (inst.writeMask == 0)
		{
			if (!IsReduction(inst) || inst.aluUnit > 3)
				return Load(TypeOf(dt), PVPSOutputPtr(inst));
			return ReadChannel(tempVar, inst.aluUnit);
		}
		if (inst.destRel != 0)
			return ReadChannel(RelativeGPR(inst.destGpr, inst.indexMode), inst.destElem);
		return ReadChannel(GPR(inst.destGpr), inst.destElem);
	}

	void SpvShader::Assign(LatteDecompilerALUInstruction& inst, Val v)
	{
		Store(OutputPtr(inst), Convert(v, dt).id);
	}

	bool SpvShader::EmitOP2(LatteDecompilerCFInstruction& cf, LatteDecompilerALUInstruction& inst)
	{
		const uint32 op = inst.opcode;
		// tempResultf, a variable: no folding through it
		auto loadT = [&]() { return Load(tF, tempResultFVar); };
		// isinf(tempResultf) == true && (floatBitsToInt(tempResultf)&0x80000000) ==/!= 0
		auto infWithSign = [&](bool negative) {
			uint32 inf = Op(SpvOpLogicalEqual, tBool, { Op(SpvOpIsInf, tBool, { loadT() }), spv_const_bool(w, 1) });
			return AndShortCircuit(inf, [&] {
				uint32 bits = Op(SpvOpBitcast, tI, { loadT() });
				uint32 masked = Op(SpvOpBitwiseAnd, tI, { bits, ConstI((sint32)0x80000000) });
				return Op(negative ? SpvOpINotEqual : SpvOpIEqual, tBool, { masked, ConstI(0) });
			});
		};
		auto binF = [&](uint32 opcode) {
			Val a = Operand(inst, 0, DT_F);
			Val b = Operand(inst, 1, DT_F);
			Assign(inst, FBin(opcode, a, b));
		};
		auto binI = [&](uint32 opcode, sint32 type) {
			Val a = Operand(inst, 0, type);
			Val b = Operand(inst, 1, type);
			Assign(inst, IBin(opcode, type, a, b));
		};
		if (op == ALU_OP2_INST_MOV)
		{
			if (inst.sourceOperand[0].abs != 0 || inst.sourceOperand[0].neg != 0)
				Assign(inst, Operand(inst, 0, DT_F));
			else
				Assign(inst, Operand(inst, 0, dt));
		}
		else if (op == ALU_OP2_INST_ADD)
			binF(SpvOpFAdd);
		else if (op == ALU_OP2_INST_MUL)
		{
			auto& s0 = inst.sourceOperand[0];
			auto& s1 = inst.sourceOperand[1];
			if (GPU7_ALU_SRC_IS_CONST_0F(s0.sel) || GPU7_ALU_SRC_IS_CONST_0F(s1.sel))
				Assign(inst, KF(0.0));
			else
			{
				bool useDefaultMul = GPU7_ALU_SRC_IS_LITERAL(s0.sel) || GPU7_ALU_SRC_IS_LITERAL(s1.sel) ||
					GPU7_ALU_SRC_IS_ANY_CONST(s0.sel) || GPU7_ALU_SRC_IS_ANY_CONST(s1.sel);
				Val a = Operand(inst, 0, DT_F);
				Val b = Operand(inst, 1, DT_F);
				if (ctx->options->strictMul && !useDefaultMul)
					Assign(inst, MulNonIEEE(a, b));
				else
					Assign(inst, FBin(SpvOpFMul, a, b));
			}
		}
		else if (op == ALU_OP2_INST_MUL_IEEE)
			binF(SpvOpFMul);
		else if (op == ALU_OP2_INST_RECIP_IEEE)
		{
			Val a = Operand(inst, 0, DT_F);
			Assign(inst, FBin(SpvOpFDiv, KF(1.0), a));
		}
		else if (op == ALU_OP2_INST_RECIP_CLAMPED)
		{
			// tempResultf = 1.0 / (x);
			// if (isinf(tempResultf)) tempResultf = uintBitsToFloat((floatBitsToUint(tempResultf) & 0x80000000u) | 0x7f7fffffu);
			Store(tempResultFVar, FBin(SpvOpFDiv, KF(1.0), Operand(inst, 0, DT_F)).id);
			If(Op(SpvOpIsInf, tBool, { loadT() }), [&] {
				uint32 bitsU = Op(SpvOpBitcast, tU, { loadT() });
				uint32 sign = Op(SpvOpBitwiseAnd, tU, { bitsU, ConstU(0x80000000u) });
				uint32 big = Op(SpvOpBitwiseOr, tU, { sign, ConstU(0x7f7fffffu) });
				Store(tempResultFVar, Op(SpvOpBitcast, tF, { big }));
			});
			Assign(inst, Run(loadT(), DT_F));
		}
		else if (op == ALU_OP2_INST_RECIP_FF)
		{
			// +INF becomes 0.0, -INF becomes -0.0
			Store(tempResultFVar, FBin(SpvOpFDiv, KF(1.0), Operand(inst, 0, DT_F)).id);
			IfElse(infWithSign(false), [&] { Store(tempResultFVar, ConstF(0.0f)); },
				[&] { If(infWithSign(true), [&] { Store(tempResultFVar, KF(-0.0).id); }); }, true);
			Assign(inst, Run(loadT(), DT_F));
		}
		else if (op == ALU_OP2_INST_RECIPSQRT_IEEE || op == ALU_OP2_INST_RECIPSQRT_CLAMPED || op == ALU_OP2_INST_RECIPSQRT_FF)
		{
			Store(tempResultFVar, FBin(SpvOpFDiv, KF(1.0), FUnary(GLSLstd450Sqrt, Operand(inst, 0, DT_F))).id);
			if (op != ALU_OP2_INST_RECIPSQRT_IEEE)
			{
				const bool clamped = op == ALU_OP2_INST_RECIPSQRT_CLAMPED;
				uint32 negV = clamped ? KF(-std::strtod("3.40282347E+38", nullptr)).id : KF(-0.0).id;
				uint32 posV = clamped ? KFText("3.40282347E+38").id : ConstF(0.0f);
				IfElse(infWithSign(true), [&] { Store(tempResultFVar, negV); },
					[&] { If(infWithSign(false), [&] { Store(tempResultFVar, posV); }); }, true);
			}
			Assign(inst, Run(loadT(), DT_F));
		}
		else if (op == ALU_OP2_INST_MAX || op == ALU_OP2_INST_MAX_DX10 || op == ALU_OP2_INST_MIN || op == ALU_OP2_INST_MIN_DX10)
		{
			Val a = Operand(inst, 0, DT_F);
			Val b = Operand(inst, 1, DT_F);
			Assign(inst, FMinMax(op == ALU_OP2_INST_MAX || op == ALU_OP2_INST_MAX_DX10, a, b));
		}
		else if (op == ALU_OP2_INST_FLOOR)
			Assign(inst, FUnary(GLSLstd450Floor, Operand(inst, 0, DT_F)));
		else if (op == ALU_OP2_INST_FRACT)
			Assign(inst, FUnary(GLSLstd450Fract, Operand(inst, 0, DT_F)));
		else if (op == ALU_OP2_INST_TRUNC)
			Assign(inst, FUnary(GLSLstd450Trunc, Operand(inst, 0, DT_F)));
		else if (op == ALU_OP2_INST_RNDNE)
			Assign(inst, FUnary(GLSLstd450RoundEven, Operand(inst, 0, DT_F)));
		else if (op == ALU_OP2_INST_EXP_IEEE)
			Assign(inst, FUnary(GLSLstd450Exp2, Operand(inst, 0, DT_F)));
		else if (op == ALU_OP2_INST_SQRT_IEEE)
			Assign(inst, FUnary(GLSLstd450Sqrt, Operand(inst, 0, DT_F)));
		else if (op == ALU_OP2_INST_LOG_CLAMPED || op == ALU_OP2_INST_LOG_IEEE)
		{
			// tempResultf = max(0.0, x); tempResultf = log2(tempResultf);
			// if( isinf(tempResultf) == true ) tempResultf = -3.40282347E+38F;
			Store(tempResultFVar, FMinMax(true, KF(0.0), Operand(inst, 0, DT_F)).id);
			Store(tempResultFVar, Ext(GLSLstd450Log2, tF, { loadT() }));
			if (op == ALU_OP2_INST_LOG_CLAMPED)
			{
				uint32 inf = Op(SpvOpLogicalEqual, tBool, { Op(SpvOpIsInf, tBool, { loadT() }), spv_const_bool(w, 1) });
				If(inf, [&] { Store(tempResultFVar, KF(-std::strtod("3.40282347E+38", nullptr)).id); });
			}
			Assign(inst, Run(loadT(), DT_F));
		}
		else if (op == ALU_OP2_INST_SIN || op == ALU_OP2_INST_COS)
		{
			Val d = FBin(SpvOpFDiv, Operand(inst, 0, DT_F), KFText("0.1591549367"));
			Assign(inst, FUnary(op == ALU_OP2_INST_SIN ? GLSLstd450Sin : GLSLstd450Cos, d));
		}
		else if (op == ALU_OP2_INST_FLT_TO_INT)
			Assign(inst, Construct(Operand(inst, 0, DT_F), DT_I));
		else if (op == ALU_OP2_INST_FLT_TO_UINT)
		{
			if (dt == DT_F)
				return Decline("uint result into float registers");
			Assign(inst, Construct(Operand(inst, 0, DT_F), DT_U));
		}
		else if (op == ALU_OP2_INST_INT_TO_FLOAT)
			Assign(inst, Construct(Operand(inst, 0, DT_I), DT_F));
		else if (op == ALU_OP2_INST_UINT_TO_FLOAT)
			Assign(inst, Construct(Operand(inst, 0, DT_U), DT_F));
		else if (op == ALU_OP2_INST_AND_INT)
			binI(SpvOpBitwiseAnd, DT_I);
		else if (op == ALU_OP2_INST_OR_INT)
			binI(SpvOpBitwiseOr, DT_I);
		else if (op == ALU_OP2_INST_XOR_INT)
			binI(SpvOpBitwiseXor, DT_I);
		else if (op == ALU_OP2_INST_NOT_INT)
		{
			Val a = Operand(inst, 0, DT_I);
			Assign(inst, a.k ? KI((sint32)~a.bits) : Run(Op(SpvOpNot, tI, { a.id }), DT_I));
		}
		else if (op == ALU_OP2_INST_ADD_INT)
			binI(SpvOpIAdd, DT_I);
		else if (op == ALU_OP2_INST_SUB_INT)
			binI(SpvOpISub, DT_I);
		else if (op == ALU_OP2_INST_MAX_INT || op == ALU_OP2_INST_MIN_INT || op == ALU_OP2_INST_MAX_UINT || op == ALU_OP2_INST_MIN_UINT)
		{
			bool isUnsigned = op == ALU_OP2_INST_MAX_UINT || op == ALU_OP2_INST_MIN_UINT;
			bool isMax = op == ALU_OP2_INST_MAX_INT || op == ALU_OP2_INST_MAX_UINT;
			sint32 type = isUnsigned ? DT_U : DT_I;
			if (isUnsigned && dt == DT_F)
				return Decline("uint result into float registers");
			Val a = Operand(inst, 0, type);
			Val b = Operand(inst, 1, type);
			uint32 ext = isUnsigned ? (isMax ? GLSLstd450UMax : GLSLstd450UMin) : (isMax ? GLSLstd450SMax : GLSLstd450SMin);
			Assign(inst, IMinMax(ext, type, a, b));
		}
		else if (op == ALU_OP2_INST_MULLO_INT)
			binI(SpvOpIMul, DT_I);
		else if (op == ALU_OP2_INST_MULLO_UINT)
		{
			if (dt == DT_F)
				return Decline("uint result into float registers");
			binI(SpvOpIMul, DT_U);
		}
		else if (op == ALU_OP2_INST_LSHL_INT)
			binI(SpvOpShiftLeftLogical, DT_I);
		else if (op == ALU_OP2_INST_LSHR_INT)
		{
			if (dt == DT_F)
				return Decline("uint result into float registers");
			if (GPU7_ALU_SRC_IS_CONST_0F(inst.sourceOperand[0].sel) && inst.sourceOperand[0].neg == 0 && inst.sourceOperand[0].abs == 0)
			{
				// 0 >> uint(x): the literal an int, the shift on it arithmetic
				Val b = Operand(inst, 1, DT_U);
				Assign(inst, IBin(SpvOpShiftRightArithmetic, DT_I, KI(0), b));
			}
			else
				binI(SpvOpShiftRightLogical, DT_U);
		}
		else if (op == ALU_OP2_INST_ASHR_INT)
			binI(SpvOpShiftRightArithmetic, DT_I);
		else if (op == ALU_OP2_INST_SETGT || op == ALU_OP2_INST_SETGE || op == ALU_OP2_INST_SETNE || op == ALU_OP2_INST_SETE)
		{
			Val a = Operand(inst, 0, DT_F);
			Val b = Operand(inst, 1, DT_F);
			uint32 cmp = op == ALU_OP2_INST_SETGT ? SpvOpFOrdGreaterThan : op == ALU_OP2_INST_SETGE ? SpvOpFOrdGreaterThanEqual :
				op == ALU_OP2_INST_SETNE ? SpvOpFOrdNotEqual : SpvOpFOrdEqual;
			Assign(inst, Select(FCmp(cmp, a, b), KF(1.0), KF(0.0), DT_F));
		}
		else if (op == ALU_OP2_INST_SETGT_DX10 || op == ALU_OP2_INST_SETE_DX10 || op == ALU_OP2_INST_SETNE_DX10 || op == ALU_OP2_INST_SETGE_DX10)
		{
			Val a = Operand(inst, 0, DT_F);
			Val b = Operand(inst, 1, DT_F);
			uint32 cmp = op == ALU_OP2_INST_SETGT_DX10 ? SpvOpFOrdGreaterThan : op == ALU_OP2_INST_SETGE_DX10 ? SpvOpFOrdGreaterThanEqual :
				op == ALU_OP2_INST_SETNE_DX10 ? SpvOpFOrdNotEqual : SpvOpFOrdEqual;
			Assign(inst, Select(FCmp(cmp, a, b), KI(-1), KI(0), DT_I));
		}
		else if (op == ALU_OP2_INST_SETE_INT || op == ALU_OP2_INST_SETNE_INT || op == ALU_OP2_INST_SETGT_INT || op == ALU_OP2_INST_SETGE_INT)
		{
			Val a = Operand(inst, 0, DT_I);
			Val b = Operand(inst, 1, DT_I);
			uint32 cmp = op == ALU_OP2_INST_SETE_INT ? SpvOpIEqual : op == ALU_OP2_INST_SETNE_INT ? SpvOpINotEqual :
				op == ALU_OP2_INST_SETGT_INT ? SpvOpSGreaterThan : SpvOpSGreaterThanEqual;
			Assign(inst, Select(ICmp(cmp, a, b), KI(-1), KI(0), DT_I));
		}
		else if (op == ALU_OP2_INST_SETGE_UINT || op == ALU_OP2_INST_SETGT_UINT)
		{
			Val a = Operand(inst, 0, DT_U);
			Val b = Operand(inst, 1, DT_U);
			Val c = ICmp(op == ALU_OP2_INST_SETGE_UINT ? SpvOpUGreaterThanEqual : SpvOpUGreaterThan, a, b);
			// int(0xFFFFFFFF) : int(0x0)
			Assign(inst, Select(c, KI(-1), KI(0), DT_I));
		}
		else if ((op >= ALU_OP2_INST_PRED_SETE && op <= ALU_OP2_INST_PRED_SETNE) || (op >= ALU_OP2_INST_PRED_SETE_INT && op <= ALU_OP2_INST_PRED_SETNE_INT))
		{
			// predResult = (a op b); then into the stack for a push, or the
			// break for a break clause
			const bool isInt = op >= ALU_OP2_INST_PRED_SETE_INT;
			Val a = Operand(inst, 0, isInt ? DT_I : DT_F);
			Val b = Operand(inst, 1, isInt ? DT_I : DT_F);
			const uint32 kind = isInt ? op - ALU_OP2_INST_PRED_SETE_INT : op - ALU_OP2_INST_PRED_SETE;
			static const uint32 fCmp[4] = { SpvOpFOrdEqual, SpvOpFOrdGreaterThan, SpvOpFOrdGreaterThanEqual, SpvOpFOrdNotEqual };
			static const uint32 iCmp[4] = { SpvOpIEqual, SpvOpSGreaterThan, SpvOpSGreaterThanEqual, SpvOpINotEqual };
			Val c = isInt ? ICmp(iCmp[kind], a, b) : FCmp(fCmp[kind], a, b);
			Store(predVar, c.id);
			const sint32 d = cf.activeStackDepth;
			if (cf.type == GPU7_CF_INST_ALU_PUSH_BEFORE)
			{
				MaskStore(false, d, Load(tBool, predVar));
				uint32 v = AndMaskC(IsTrue(Load(tBool, predVar)), d);
				MaskStore(true, d + 1, v);
			}
			else if (cf.type == GPU7_CF_INST_ALU_BREAK && inst.updateExecuteMask)
			{
				// aluBreak = aluBreak || (predResult == false);
				uint32 brk = Load(tBool, aluBreakVar);
				uint32 notPred = Op(SpvOpLogicalEqual, tBool, { Load(tBool, predVar), spv_const_bool(w, 0) });
				Store(aluBreakVar, Op(SpvOpLogicalOr, tBool, { brk, notPred }));
			}
		}
		else if (op == ALU_OP2_INST_KILLE_INT || op == ALU_OP2_INST_KILLNE_INT || op == ALU_OP2_INST_KILLGT_INT)
		{
			Val a = Operand(inst, 0, DT_I);
			Val b = Operand(inst, 1, DT_I);
			uint32 cmp = op == ALU_OP2_INST_KILLE_INT ? SpvOpIEqual : op == ALU_OP2_INST_KILLNE_INT ? SpvOpINotEqual : SpvOpSGreaterThan;
			Val c = ICmp(cmp, a, b);
			if (Ok())
				KillIf(c.id);
		}
		else if (op == ALU_OP2_INST_KILLGT || op == ALU_OP2_INST_KILLGE || op == ALU_OP2_INST_KILLE)
		{
			Val a = Operand(inst, 0, DT_F);
			Val b = Operand(inst, 1, DT_F);
			uint32 cmp = op == ALU_OP2_INST_KILLGT ? SpvOpFOrdGreaterThan : op == ALU_OP2_INST_KILLGE ? SpvOpFOrdGreaterThanEqual : SpvOpFOrdEqual;
			Val c = FCmp(cmp, a, b);
			if (Ok())
				KillIf(c.id);
		}
		else
			return Decline("ALU OP2 instruction");
		return Ok();
	}

	bool SpvShader::EmitOP3(LatteDecompilerALUInstruction& inst)
	{
		const uint32 op = inst.opcode;
		if (op == ALU_OP3_INST_CMOVGE || op == ALU_OP3_INST_CMOVE || op == ALU_OP3_INST_CMOVGT ||
			op == ALU_OP3_INST_CNDE_INT || op == ALU_OP3_INST_CNDGT_INT || op == ALU_OP3_INST_CMOVGE_INT)
		{
			auto& a = inst.sourceOperand[1];
			auto& b = inst.sourceOperand[2];
			bool same = a.sel == b.sel && GPU7_ALU_SRC_IS_GPR(a.sel) && a.chan == b.chan && a.abs == b.abs && a.neg == b.neg && a.rel == b.rel;
			if (same && a.abs == 0 && a.neg == 0)
			{
				Assign(inst, Operand(inst, 1, dt));
				return Ok();
			}
		}
		if (op == ALU_OP3_INST_MULADD || op == ALU_OP3_INST_MULADD_D2 || op == ALU_OP3_INST_MULADD_M2 || op == ALU_OP3_INST_MULADD_M4 || op == ALU_OP3_INST_MULADD_IEEE)
		{
			auto& s0 = inst.sourceOperand[0];
			auto& s1 = inst.sourceOperand[1];
			bool useDefaultMul = GPU7_ALU_SRC_IS_LITERAL(s0.sel) || GPU7_ALU_SRC_IS_LITERAL(s1.sel) ||
				GPU7_ALU_SRC_IS_ANY_CONST(s0.sel) || GPU7_ALU_SRC_IS_ANY_CONST(s1.sel) || op == ALU_OP3_INST_MULADD_IEEE;
			Val a = Operand(inst, 0, DT_F);
			Val b = Operand(inst, 1, DT_F);
			Val m;
			if (ctx->options->strictMul && !useDefaultMul)
				m = MulNonIEEE(a, b);
			else
				m = FBin(SpvOpFMul, a, b);
			Val c = Operand(inst, 2, DT_F);
			Val r = FBin(SpvOpFAdd, m, c);
			if (op == ALU_OP3_INST_MULADD_D2)
				r = FBin(SpvOpFDiv, r, KF(2.0));
			else if (op == ALU_OP3_INST_MULADD_M2)
				r = FBin(SpvOpFMul, r, KF(2.0));
			else if (op == ALU_OP3_INST_MULADD_M4)
				r = FBin(SpvOpFMul, r, KF(4.0));
			Assign(inst, r);
		}
		else if (op == ALU_OP3_INST_CNDE_INT || op == ALU_OP3_INST_CNDGT_INT || op == ALU_OP3_INST_CMOVGE_INT)
		{
			bool requiresFloatResult = inst.sourceOperand[1].neg != 0 || inst.sourceOperand[2].neg != 0;
			sint32 type = requiresFloatResult ? DT_F : DT_I;
			Val c = Operand(inst, 0, DT_I);
			uint32 cmp = op == ALU_OP3_INST_CNDE_INT ? SpvOpIEqual : op == ALU_OP3_INST_CNDGT_INT ? SpvOpSGreaterThan : SpvOpSGreaterThanEqual;
			Val cond = ICmp(cmp, c, KI(0));
			size_t from = BodyAt();
			Val a = Operand(inst, 1, type);
			size_t aEnd = BodyAt();
			Val b = Operand(inst, 2, type);
			Assign(inst, Ternary(cond, a, aEnd, b, from, type));
		}
		else if (op == ALU_OP3_INST_CMOVGE || op == ALU_OP3_INST_CMOVE || op == ALU_OP3_INST_CMOVGT)
		{
			Val c = Operand(inst, 0, DT_F);
			uint32 cmp = op == ALU_OP3_INST_CMOVE ? SpvOpFOrdEqual : op == ALU_OP3_INST_CMOVGE ? SpvOpFOrdGreaterThanEqual : SpvOpFOrdGreaterThan;
			Val cond = FCmp(cmp, c, KF(0.0));
			size_t from = BodyAt();
			Val a = Operand(inst, 1, DT_I);
			size_t aEnd = BodyAt();
			Val b = Operand(inst, 2, DT_I);
			Assign(inst, Ternary(cond, a, aEnd, b, from, DT_I));
		}
		else
			return Decline("ALU OP3 instruction");
		return Ok();
	}

	// CUBE: redcCUBE(vec4(src0s), vec4(src1s), cubeMapSTM, cubeMapFaceId);
	// then S, T, the major axis and float(faceId) into the four outputs
	bool SpvShader::EmitCube(LatteDecompilerALUInstruction* redc[4])
	{
		if (!cubeSTMVar)
			return Decline("CUBE without its variables");
		Val a[4], b[4];
		for (int i = 0; i < 4; i++)
			a[i] = Operand(*redc[i], 0, DT_F);
		for (int i = 0; i < 4; i++)
			b[i] = Operand(*redc[i], 1, DT_F);
		if (!Ok())
			return false;
		const uint32 tV3F = spv_type_vector(w, tF, 3);
		uint32 args[4] = { CoordVector(a, 4), CoordVector(b, 4), 0, 0 };
		const uint32 types[4] = { tV4F, tV4F, tV3F, tI };
		uint32 temps[4];
		CallTyped(RedcCUBEFunction(), tVoid, types, args, 4, temps);
		Store(cubeSTMVar, Load(tV3F, temps[2]));
		Store(cubeFaceVar, Load(tI, temps[3]));
		for (uint32 i = 0; i < 3; i++)
			Assign(*redc[i], Run(Op(SpvOpCompositeExtract, tF, { Load(tV3F, cubeSTMVar), i }), DT_F));
		Assign(*redc[3], Run(Op(SpvOpConvertSToF, tF, { Load(tI, cubeFaceVar) }), DT_F));
		return Ok();
	}

	bool SpvShader::EmitReduction(LatteDecompilerALUInstruction* redc[4])
	{
		if (redc[0]->opcode == ALU_OP2_INST_CUBE)
			return EmitCube(redc);
		Val a[4], b[4];
		for (int i = 0; i < 4; i++)
			a[i] = Operand(*redc[i], 0, DT_F);
		for (int i = 0; i < 4; i++)
			b[i] = Operand(*redc[i], 1, DT_F);
		bool allConst = true;
		for (int i = 0; i < 4; i++)
			allConst = allConst && a[i].k && b[i].k;
		if (allConst)
		{
			// rslc_cval_dot
			double r = 0.0;
			for (int i = 0; i < 4; i++)
				r += a[i].d * b[i].d;
			Assign(*redc[0], KF(r));
			return Ok();
		}
		uint32 va = Op(SpvOpCompositeConstruct, tV4F, { a[0].id, a[1].id, a[2].id, a[3].id });
		uint32 vb = Op(SpvOpCompositeConstruct, tV4F, { b[0].id, b[1].id, b[2].id, b[3].id });
		Assign(*redc[0], Run(Op(SpvOpDot, tF, { va, vb }), DT_F));
		return Ok();
	}

	bool SpvShader::EmitALUClause(LatteDecompilerCFInstruction& cf)
	{
		for (auto& it : pvps)
			it = PVPSAlias{};
		temporaryCount = 0;
		LatteDecompilerALUInstruction* redc[4];
		size_t groupStart = 0;
		auto& list = cf.instructionsALU;
		for (size_t i = 0; i < list.size(); i++)
		{
			LatteDecompilerALUInstruction& inst = list[i];
			if (inst.indexInGroup == 0)
			{
				if (i > 0)
					TrackGroupOutputPVPS(list.data() + groupStart, i - groupStart);
				groupStart = i;
				size_t groupSize = 1;
				while (i + groupSize < list.size() && list[i + groupSize].instructionGroupIndex == inst.instructionGroupIndex)
					groupSize++;
				if (!CreateGPRTemporaries(list.data() + i, groupSize))
					return false;
			}
			bool isReduction = IsReduction(inst);
			if (isReduction)
			{
				if (i + 4 > list.size())
					return Decline("reduction cut short");
				for (int r = 0; r < 4; r++)
					redc[r] = &list[i + r];
				for (int r = 1; r < 4; r++)
				{
					if (redc[r]->isOP3 != redc[0]->isOP3 || redc[r]->opcode != redc[0]->opcode || redc[r]->omod != redc[0]->omod || redc[r]->destClamp != redc[0]->destClamp)
						return Decline("mixed reduction");
				}
				if (!EmitReduction(redc))
					return false;
				i += 3;
			}
			else if (inst.isOP3)
			{
				if (!EmitOP3(inst))
					return false;
			}
			else
			{
				if (inst.opcode == ALU_OP2_INST_NOP)
					continue;
				if (inst.opcode == ALU_OP2_INST_MOVA_FLOOR || inst.opcode == ALU_OP2_INST_MOVA_INT)
				{
					if (!EmitMova(inst))
						return false;
				}
				else
				if (!EmitOP2(cf, inst))
					return false;
			}
			// omod and clamp on the instruction's output - a reduction's first,
			// as in the GLSL emitter
			LatteDecompilerALUInstruction& last = inst;
			if (last.omod != ALU_OMOD_NONE)
			{
				uint32 v = ReadOutput(last);
				uint32 f = dt == DT_F ? v : Op(SpvOpBitcast, tF, { v });
				uint32 r;
				if (last.omod == ALU_OMOD_MUL2)
					r = Op(SpvOpFMul, tF, { f, ConstF(2.0f) });
				else if (last.omod == ALU_OMOD_MUL4)
					r = Op(SpvOpFMul, tF, { f, ConstF(4.0f) });
				else
					r = Op(SpvOpFDiv, tF, { f, ConstF(2.0f) });
				Store(OutputPtr(last), dt == DT_F ? r : Op(SpvOpBitcast, tI, { r }));
			}
			if (last.destClamp != 0)
			{
				uint32 v = ReadOutput(last);
				uint32 r;
				if (dt == DT_F)
					r = Ext(GLSLstd450FClamp, tF, { v, ConstF(0.0f), ConstF(1.0f) });
				else
					r = Call(ClampFI32Function(), tI, DT_I, tI, { v }).id;
				Store(OutputPtr(last), r);
			}
			if (isReduction && redc[0]->opcode == ALU_OP2_INST_CUBE)
			{
				// CUBE: a PV written for itself, from itself
				for (int f = 0; f < 4; f++)
				{
					if (redc[f]->writeMask == 0)
						Store(PVPSOutputPtr(*redc[f]), ReadOutput(*redc[f]));
				}
			}
			else if (isReduction)
			{
				// DOT4: the result into the other three outputs and PVs, read
				// again for each as the GLSL emitter does
				for (int f = 0; f < 4; f++)
				{
					if (redc[f]->writeMask == 0)
						Store(PVPSOutputPtr(*redc[f]), ReadOutput(*redc[0]));
					else if (f != 0)
						Store(OutputPtr(*redc[f]), ReadOutput(*redc[0]));
				}
			}
			if (!Ok())
				return false;
		}
		return Ok();
	}

	Val SpvShader::ExportComponent(LatteDecompilerCFInstruction& cf, uint32 burst, uint32 i)
	{
		uint32 sel = cf.exportComponentSel[i];
		if (sel < 4)
		{
			uint32 v = ReadChannel(GPR(cf.exportSourceGPR + burst), sel);
			return Convert({ v, dt }, DT_F);
		}
		if (sel == 5)
			return { ConstF(1.0f), DT_F };
		if (sel == 4 || sel == 7)
			return { ConstF(0.0f), DT_F };
		Decline("export select");
		return { ConstF(0.0f), DT_F };
	}

	bool SpvShader::EmitExport(LatteDecompilerCFInstruction& cf)
	{
		if (cf.exportType != 0 || cf.exportArrayBase >= 8)
			return Decline("pixel export other than color");
		for (uint32 i = 0; i < cf.exportBurstCount + 1; i++)
		{
			sint32 colorIndex = LatteDecompiler_getColorOutputIndexFromExportIndex(ctx, cf.exportArrayBase + i);
			if (colorIndex < 0 || !colorOut[colorIndex])
				return Decline("color export not matched");
			bool alphaTestEnable = ctx->contextRegistersNew->SX_ALPHA_TEST_CONTROL.get_ALPHA_TEST_ENABLE();
			auto alphaTestFunc = ctx->contextRegistersNew->SX_ALPHA_TEST_CONTROL.get_ALPHA_FUNC();
			if (colorIndex == 0 && alphaTestEnable && alphaTestFunc == Latte::E_COMPAREFUNC::NEVER)
			{
				// discard; what follows it in a block of its own, unreached
				Kill();
			}
			else if (colorIndex == 0 && alphaTestEnable && alphaTestFunc != Latte::E_COMPAREFUNC::ALWAYS)
			{
				uint32 c[4];
				for (uint32 k = 0; k < 4; k++)
					c[k] = ExportComponent(cf, i, k).id;
				uint32 alpha = c[3];
				uint32 ptr = spv_type_pointer(w, SpvStorageClassUniform, tF);
				// uf_alphaTestRef, after the register array where there is one
				uint32 refPtr = Op(SpvOpAccessChain, ptr, { ufBlockVar, ConstI(alphaRefMember) });
				uint32 ref = Load(tF, refPtr);
				uint32 cmp;
				switch (alphaTestFunc)
				{
				case Latte::E_COMPAREFUNC::LESS: cmp = SpvOpFOrdLessThan; break;
				case Latte::E_COMPAREFUNC::EQUAL: cmp = SpvOpFOrdEqual; break;
				case Latte::E_COMPAREFUNC::LEQUAL: cmp = SpvOpFOrdLessThanEqual; break;
				case Latte::E_COMPAREFUNC::GREATER: cmp = SpvOpFOrdGreaterThan; break;
				case Latte::E_COMPAREFUNC::NOTEQUAL: cmp = SpvOpFOrdNotEqual; break;
				case Latte::E_COMPAREFUNC::GEQUAL: cmp = SpvOpFOrdGreaterThanEqual; break;
				default: return Decline("alpha test function");
				}
				uint32 pass = Op(cmp, tBool, { alpha, ref });
				KillIf(Op(SpvOpLogicalEqual, tBool, { pass, spv_const_bool(w, 0) }));
			}
			uint32 c[4];
			for (uint32 k = 0; k < 4; k++)
				c[k] = ExportComponent(cf, i, k).id;
			Store(colorOut[colorIndex], Op(SpvOpCompositeConstruct, tV4F, { c[0], c[1], c[2], c[3] }));
		}
		return Ok();
	}

	// _emitTextureDefinitions: a sampler per texture unit used
	bool SpvShader::DeclareTextures()
	{
		for (sint32 i = 0; i < LATTE_NUM_MAX_TEX_UNITS; i++)
		{
			if (!ctx->output->textureUnitMask[i])
				continue;
			const Latte::E_DIM dim = shader->textureUnitDim[i];
			const bool isInt = shader->textureIsIntegerFormat[i];
			const bool shadow = shader->textureUsesDepthCompare[i];
			if (shadow && (isInt || (dim != Latte::E_DIM::DIM_2D && dim != Latte::E_DIM::DIM_2D_ARRAY && dim != Latte::E_DIM::DIM_1D)))
				return Decline("shadow sampler");
			uint32 spvDim;
			bool arrayed = false;
			if (dim == Latte::E_DIM::DIM_2D)
				spvDim = SpvDimension2D;
			else if (dim == Latte::E_DIM::DIM_1D)
			{
				spvDim = SpvDimension1D;
				sampled1D = true;
			}
			else if (dim == Latte::E_DIM::DIM_CUBEMAP && !isInt && !shadow)
			{
				// samplerCubeArray, its layer cubeMapArrayIndex<n>
				spvDim = SpvDimensionCube;
				arrayed = true;
				sampledCubeArray = true;
			}
			else if (dim == Latte::E_DIM::DIM_2D_ARRAY && !isInt)
			{
				spvDim = SpvDimension2D;
				arrayed = true;
			}
			else if (dim == Latte::E_DIM::DIM_3D && !isInt)
				spvDim = SpvDimension3D;
			else
				return Decline("texture dimension");
			uint32 image = shadow ? spv_type_depth_image(w, tF, spvDim, arrayed ? 1 : 0) :
				spv_type_image(w, isInt ? tU : tF, spvDim, arrayed ? 1 : 0);
			texImage[i] = image;
			texType[i] = spv_type_sampled_image(w, image);
			uint32 ptr = spv_type_pointer(w, SpvStorageClassUniformConstant, texType[i]);
			texVar[i] = spv_global_variable(w, ptr, SpvStorageClassUniformConstant);
			Decorate(texVar[i], SpvDecorationDescriptorSet, { (uint32)ctx->output->resourceMappingVK.setIndex });
			Decorate(texVar[i], SpvDecorationBinding, { (uint32)ctx->output->resourceMappingVK.textureUnitToBindingPoint[i] });
		}
		// vec4 gradH, gradV: declared, not set
		if (ctx->analyzer.hasGradientLookup)
		{
			gradVar[0] = Local(tV4F, 0);
			gradVar[1] = Local(tV4F, 0);
		}
		return true;
	}

	// _emitTEXSampleCoordInputComponent: a register's channel or a
	// constant, as a float or an int
	Val SpvShader::TexCoordAs(LatteDecompilerTEXInstruction& t, sint32 component, sint32 type)
	{
		sint32 sel = t.textureFetch.srcSel[component];
		if (sel < 4)
			return Convert({ ReadChannel(GPR(t.srcGpr), (uint32)sel), dt }, type);
		if (sel == 4 || sel == 5)
		{
			// floatBitsToInt(0.0) as an int: a bit cast rslc does not fold
			Val k = KF(sel == 5 ? 1.0 : 0.0);
			return type == DT_F ? k : Convert(k, type);
		}
		Decline("texture coordinate select");
		return KF(0.0);
	}

	Val SpvShader::TexCoord(LatteDecompilerTEXInstruction& t, sint32 component)
	{
		return TexCoordAs(t, component, DT_F);
	}

	// _getTexGPRAccess: the register's channels swizzled, as floats
	uint32 SpvShader::TexGPRAccess(LatteDecompilerTEXInstruction& t, const uint8* sel, uint32 n)
	{
		for (uint32 i = 0; i < n; i++)
		{
			if (sel[i] >= 4)
			{
				Decline("texture coordinate select");
				return ConstF(0.0f);
			}
		}
		if (dt != DT_I && dt != DT_F)
		{
			Decline("texture coordinate type");
			return ConstF(0.0f);
		}
		uint32 v = Swizzle(Load(Vec4Of(dt), GPR(t.srcGpr)), TypeOf(dt), sel, n);
		if (dt == DT_I)
			v = Op(SpvOpBitcast, n == 1 ? tF : spv_type_vector(w, tF, n), { v });
		return v;
	}

	// vecN(c...) of the coordinates: a constant when they all are
	uint32 SpvShader::CoordVector(const Val* c, uint32 n)
	{
		bool allConst = true;
		uint32 parts[4];
		for (uint32 i = 0; i < n; i++)
		{
			allConst = allConst && c[i].k;
			parts[i] = c[i].id;
		}
		uint32 type = spv_type_vector(w, tF, n);
		if (allConst)
			return spv_const_composite(w, type, parts, n);
		return Construct(type, parts, n);
	}

	// the image op's words: result, sampled image or image, coordinate, an
	// extra operand (the reference, the component) and the image operands
	uint32 SpvShader::ImageOp(uint32 opcode, uint32 resultType, uint32 image, uint32 coord, uint32 extra, uint32 mask, const uint32* operands, uint32 nOperands)
	{
		uint32 words[10];
		size_t k = 0;
		uint32 id = spv_id(w);
		words[k++] = resultType;
		words[k++] = id;
		words[k++] = image;
		words[k++] = coord;
		if (extra)
			words[k++] = extra;
		if (mask)
		{
			words[k++] = mask;
			for (uint32 i = 0; i < nOperands; i++)
				words[k++] = operands[i];
		}
		EnsureBlock();
		spv_inst(w, SPV_SEC_BODY, opcode, words, k);
		return id;
	}

	// the offset's ivec2, ivec3 for a 3D texture: halves of the fields
	uint32 SpvShader::TexOffset(LatteDecompilerTEXInstruction& t, Latte::E_DIM dim)
	{
		if (dim == Latte::E_DIM::DIM_1D)
			return ConstI(t.textureFetch.offsetX / 2);
		const uint32 m = dim == Latte::E_DIM::DIM_3D ? 3 : 2;
		uint32 parts[3] = { ConstI(t.textureFetch.offsetX / 2), ConstI(t.textureFetch.offsetY / 2), ConstI(t.textureFetch.offsetZ / 2) };
		return spv_const_composite(w, spv_type_vector(w, tI, m), parts, m);
	}

	// the components written, in order; none written: declined
	uint32 SpvShader::TexWritten(LatteDecompilerTEXInstruction& t, uint32 written[4])
	{
		uint32 count = 0;
		for (uint32 f = 0; f < 4; f++)
		{
			if (t.dstSel[f] < 4)
				written[count++] = f;
		}
		if (count == 0)
			Decline("texture result written nowhere");
		return count;
	}

	// the result's selected components into the destination: converted as
	// the GLSL does (a bit cast of the whole, or ivecN(uvecN) by
	// component), one channel through its access chain, more loaded,
	// inserted and stored whole as rslc writes a swizzle
	bool SpvShader::StoreTexResult(LatteDecompilerTEXInstruction& t, uint32 v, sint32 sampleType, uint32 count, const uint32* written)
	{
		const bool perComponent = sampleType == DT_U && dt == DT_I && count > 1;
		if (sampleType != dt && !perComponent)
		{
			uint32 to = count == 1 ? TypeOf(dt) : spv_type_vector(w, TypeOf(dt), count);
			v = Op(SpvOpBitcast, to, { v });
		}
		if (count == 1)
		{
			Store(Channel(GPR(t.dstGpr), dt, written[0]), v);
			return Ok();
		}
		if (perComponent)
		{
			uint32 parts[4];
			for (uint32 i = 0; i < count; i++)
				parts[i] = Op(SpvOpBitcast, tI, { Op(SpvOpCompositeExtract, tU, { v, i }) });
			v = Construct(spv_type_vector(w, tI, count), parts, count);
		}
		uint32 reg = Load(Vec4Of(dt), GPR(t.dstGpr));
		for (uint32 i = 0; i < count; i++)
			reg = Op(SpvOpCompositeInsert, Vec4Of(dt), { Op(SpvOpCompositeExtract, TypeOf(dt), { v, i }), reg, written[i] });
		Store(GPR(t.dstGpr), reg);
		return Ok();
	}

	// the written components' selects of a vec4 result, through a
	// reordering (FETCH4's) or none
	uint32 SpvShader::TexSwizzle(LatteDecompilerTEXInstruction& t, uint32 v, uint32 elem, uint32 count, const uint32* written, const uint8* order)
	{
		uint8 sel[4];
		for (uint32 i = 0; i < count; i++)
		{
			uint8 s = t.dstSel[written[i]];
			sel[i] = order ? order[s] : s;
		}
		return Swizzle(v, elem, sel, count);
	}

	// _emitTEXSampleTextureCode: SAMPLE, SAMPLE_L, SAMPLE_LZ and SAMPLE_LB
	bool SpvShader::EmitSample(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		const uint32 op = t.opcode;
		if (shader->textureUsesDepthCompare[unit])
			return Decline("sample of a shadow sampler");
		const Latte::E_DIM dim = shader->textureUnitDim[unit];
		const bool isInt = shader->textureIsIntegerFormat[unit];
		uint32 written[4];
		const uint32 count = TexWritten(t, written);
		if (!count)
			return false;
		const bool hasOffset = t.textureFetch.offsetX != 0 || t.textureFetch.offsetY != 0 || t.textureFetch.offsetZ != 0;
		const bool three = dim == Latte::E_DIM::DIM_3D || dim == Latte::E_DIM::DIM_2D_ARRAY;
		const bool one = dim == Latte::E_DIM::DIM_1D;
		const bool cube = dim == Latte::E_DIM::DIM_CUBEMAP;
		if (!three && !one && !cube && ActiveSettings::ForceSamplerRoundToPrecision())
			return Decline("sampler round to precision");
		if (cube && hasOffset)
			return Decline("cube map offset"); // no offset argument written
		// sampler, coordinates, then the level or bias, as the call reads them
		uint32 sampler = Load(texType[unit], texVar[unit]);
		uint32 coord;
		if (cube)
			coord = CubeCoord(t);
		else if (one)
			coord = TexGPRAccess1(t, 0); // a 1D's: the register's channel as it is
		else
		{
			Val c[3];
			const uint32 n = three ? 3 : 2;
			for (uint32 i = 0; i < n; i++)
				c[i] = TexCoord(t, (sint32)i);
			coord = CoordVector(c, n);
		}
		uint32 mask = 0;
		uint32 operands[2];
		uint32 nOperands = 0;
		bool explicitLod = false;
		if (NotPixel() && (op == GPU7_TEX_INST_SAMPLE || op == GPU7_TEX_INST_SAMPLE_LB))
		{
			// texture() outside a fragment shader: level 0, the bias read
			// as the call's argument and dropped, as rslc lowers it
			if (op == GPU7_TEX_INST_SAMPLE_LB)
				TexCoord(t, 3);
			mask |= SpvImageOperandsLodMask;
			operands[nOperands++] = ConstF(0.0f);
			explicitLod = true;
		}
		else if (op == GPU7_TEX_INST_SAMPLE_LB)
		{
			mask |= SpvImageOperandsBiasMask;
			operands[nOperands++] = TexCoord(t, 3).id;
		}
		else if (op == GPU7_TEX_INST_SAMPLE_L)
		{
			mask |= SpvImageOperandsLodMask;
			operands[nOperands++] = TexCoord(t, 3).id;
			explicitLod = true;
		}
		else if (op == GPU7_TEX_INST_SAMPLE_LZ)
		{
			mask |= SpvImageOperandsLodMask;
			operands[nOperands++] = ConstF(0.0f);
			explicitLod = true;
		}
		if (hasOffset)
		{
			mask |= SpvImageOperandsConstOffsetMask;
			operands[nOperands++] = TexOffset(t, dim);
		}
		if (!Ok())
			return false;
		const uint32 elem = isInt ? tU : tF;
		uint32 sample = ImageOp(explicitLod ? SpvOpImageSampleExplicitLod : SpvOpImageSampleImplicitLod,
			isInt ? tV4U : tV4F, sampler, coord, 0, mask, operands, nOperands);
		return StoreTexResult(t, TexSwizzle(t, sample, elem, count, written, nullptr), isInt ? DT_U : DT_F, count, written);
	}

	// SAMPLE_C, SAMPLE_C_L and SAMPLE_C_LZ: a 2D's coordinates and
	// reference as a vec3, a 2D array's as a vec4; texture() or textureLod()
	// of the shadow sampler, its float spread over the components written
	bool SpvShader::EmitShadowSample(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		uint32 op = t.opcode;
		if (NotPixel() && op == GPU7_TEX_INST_SAMPLE_C_LZ)
			op = GPU7_TEX_INST_SAMPLE_C; // the vertex shader's level is zero
		if (!shader->textureUsesDepthCompare[unit])
			return Decline("compare of a sampler without one");
		const Latte::E_DIM dim = shader->textureUnitDim[unit];
		uint32 written[4];
		const uint32 count = TexWritten(t, written);
		if (!count)
			return false;
		const bool hasOffset = t.textureFetch.offsetX != 0 || t.textureFetch.offsetY != 0 || t.textureFetch.offsetZ != 0;
		const bool isArray = dim == Latte::E_DIM::DIM_2D_ARRAY;
		const bool one = dim == Latte::E_DIM::DIM_1D;
		// no textureLod() of an array's shadow sampler, no textureLodOffset()
		// written for SAMPLE_C_L, nor an offset of an array's
		if (isArray && (op != GPU7_TEX_INST_SAMPLE_C || hasOffset))
			return Decline("shadow array level or offset");
		if (op == GPU7_TEX_INST_SAMPLE_C_L && hasOffset)
			return Decline("shadow level with an offset");
		// textureLod()'s result swizzled by the selects: only x is there
		if (op == GPU7_TEX_INST_SAMPLE_C_L)
		{
			for (uint32 i = 0; i < count; i++)
			{
				if (t.dstSel[written[i]] != 0)
					return Decline("shadow result component");
			}
		}
		uint32 sampler = Load(texType[unit], texVar[unit]);
		uint32 coord;
		if (one)
		{
			// vec3(x, 0.0, the reference): w the reference for every kind
			uint32 parts[3] = { TexGPRAccess1(t, 0), ConstF(0.0f), TexGPRAccess1(t, 3) };
			coord = Construct(spv_type_vector(w, tF, 3), parts, 3);
		}
		else if (isArray)
		{
			Val c[3];
			for (uint32 i = 0; i < 3; i++)
				c[i] = TexCoord(t, (sint32)i);
			uint8 refSel = (uint8)t.textureFetch.srcSel[3];
			uint32 ref = TexGPRAccess(t, &refSel, 1);
			uint32 parts[4] = { c[0].id, c[1].id, c[2].id, ref };
			coord = Construct(tV4F, parts, 4);
		}
		else
		{
			uint8 xy[2] = { (uint8)t.textureFetch.srcSel[0], (uint8)t.textureFetch.srcSel[1] };
			uint32 v = TexGPRAccess(t, xy, 2);
			uint8 refSel = (uint8)(op == GPU7_TEX_INST_SAMPLE_C_L ? t.textureFetch.srcSel[2] : t.textureFetch.srcSel[3]);
			uint32 ref = TexGPRAccess(t, &refSel, 1);
			uint32 parts[2] = { v, ref };
			coord = Construct(spv_type_vector(w, tF, 3), parts, 2);
		}
		uint32 mask = 0;
		uint32 operands[2];
		uint32 nOperands = 0;
		bool explicitLod = true;
		if (op == GPU7_TEX_INST_SAMPLE_C_L)
		{
			mask |= SpvImageOperandsLodMask;
			operands[nOperands++] = TexCoord(t, 3).id;
		}
		else if (op == GPU7_TEX_INST_SAMPLE_C_LZ || NotPixel())
		{
			mask |= SpvImageOperandsLodMask;
			operands[nOperands++] = ConstF(0.0f);
		}
		else
			explicitLod = false;
		if (hasOffset)
		{
			mask |= SpvImageOperandsConstOffsetMask;
			operands[nOperands++] = TexOffset(t, dim);
		}
		if (!Ok())
			return false;
		// the reference: the coordinate's last component, as rslc takes it
		uint32 ref = Op(SpvOpCompositeExtract, tF, { coord, isArray ? 3u : 2u });
		uint32 v = ImageOp(explicitLod ? SpvOpImageSampleDrefExplicitLod : SpvOpImageSampleDrefImplicitLod,
			tF, sampler, coord, ref, mask, operands, nOperands);
		if (count > 1)
		{
			uint32 parts[4] = { v, v, v, v };
			v = Construct(spv_type_vector(w, tF, count), parts, count);
		}
		return StoreTexResult(t, v, DT_F, count, written);
	}

	// FETCH4: textureGather() of a 2D, nudged by vec2(0.0001), or of a 2D
	// array; its components in FETCH4's order
	bool SpvShader::EmitGather(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		const Latte::E_DIM dim = shader->textureUnitDim[unit];
		const bool isInt = shader->textureIsIntegerFormat[unit];
		if (shader->textureUsesDepthCompare[unit])
			return Decline("gather of a shadow sampler");
		if (dim == Latte::E_DIM::DIM_1D && !isInt)
		{
			// no FETCH4 of a 1D: vecN(0.0), as the GLSL writes it
			uint32 written[4];
			const uint32 count = TexWritten(t, written);
			if (!count)
				return false;
			return StoreTexResult(t, count == 1 ? ConstF(0.0f) : SplatF(0.0f, count), DT_F, count, written);
		}
		const bool cube = dim == Latte::E_DIM::DIM_CUBEMAP;
		if (dim != Latte::E_DIM::DIM_2D && dim != Latte::E_DIM::DIM_2D_ARRAY && !cube)
			return Decline("gather dimension");
		if (t.textureFetch.offsetX != 0 || t.textureFetch.offsetY != 0 || t.textureFetch.offsetZ != 0)
			return Decline("gather offset");
		const bool isArray = dim == Latte::E_DIM::DIM_2D_ARRAY || cube;
		if (!isArray && ActiveSettings::ForceSamplerRoundToPrecision())
			return Decline("sampler round to precision");
		uint32 written[4];
		const uint32 count = TexWritten(t, written);
		if (!count)
			return false;
		uint32 sampler = Load(texType[unit], texVar[unit]);
		uint32 coord;
		if (cube)
			coord = CubeCoord(t);
		else
		{
			Val c[3];
			const uint32 n = isArray ? 3 : 2;
			for (uint32 i = 0; i < n; i++)
				c[i] = TexCoord(t, (sint32)i);
			if (!isArray && c[0].k && c[1].k)
				return Decline("constant gather coordinate"); // the sum folds
			coord = CoordVector(c, n);
		}
		if (!isArray)
			coord = Op(SpvOpFAdd, tV2F, { SplatF(0.0001f, 2), coord });
		if (!Ok())
			return false;
		uint32 v = ImageOp(SpvOpImageGather, isInt ? tV4U : tV4F, sampler, coord, ConstI(0), 0, nullptr, 0);
		// textureGather()'s xyzw from FETCH4's yzxw
		static const uint8 fetchToGather[8] = { 2, 0, 1, 3, 0, 0, 0, 0 };
		return StoreTexResult(t, TexSwizzle(t, v, isInt ? tU : tF, count, written, fetchToGather), isInt ? DT_U : DT_F, count, written);
	}

	// LD and SAMPLE of texel coordinates: texelFetch() of the coordinates
	// scaled by uf_tex<n>Scale, of LD's level or level 0
	bool SpvShader::EmitTexelFetch(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		const Latte::E_DIM dim = shader->textureUnitDim[unit];
		const bool isInt = shader->textureIsIntegerFormat[unit];
		const bool isLD = t.opcode == GPU7_TEX_INST_LD;
		if (shader->textureUsesDepthCompare[unit])
			return Decline("texel fetch of a shadow sampler");
		const bool one = dim == Latte::E_DIM::DIM_1D;
		if (dim != Latte::E_DIM::DIM_2D && !one)
			return Decline("texel fetch dimension");
		if (t.textureFetch.offsetX != 0 || t.textureFetch.offsetY != 0 || t.textureFetch.offsetZ != 0)
			return Decline("texel fetch offset");
		if (texScaleMember[unit] < 0)
			return Decline("texel scale not declared");
		uint32 written[4];
		const uint32 count = TexWritten(t, written);
		if (!count)
			return false;
		uint32 sampler = Load(texType[unit], texVar[unit]);
		const sint32 type = isLD ? DT_I : DT_F;
		uint32 scalePtr = 0;
		if (one)
		{
			// int(float(x)*uf_tex<n>Scale.x)
			Val x = TexCoordAs(t, 0, type);
			uint32 xf = isLD ? Op(SpvOpConvertSToF, tF, { x.id }) : x.id;
			scalePtr = Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassUniform, tV2F), { ufBlockVar, ConstI(texScaleMember[unit]) });
			uint32 sx = Op(SpvOpCompositeExtract, tF, { Load(tV2F, scalePtr), 0 });
			uint32 coord1 = Op(SpvOpConvertFToS, tI, { Op(SpvOpFMul, tF, { xf, sx }) });
			uint32 lod1 = isLD ? TexCoordAs(t, 3, DT_I).id : ConstI(0);
			if (!Ok())
				return false;
			uint32 image1 = Op(SpvOpImage, texImage[unit], { sampler });
			uint32 v1 = ImageOp(SpvOpImageFetch, isInt ? tV4U : tV4F, image1, coord1, 0, SpvImageOperandsLodMask, &lod1, 1);
			return StoreTexResult(t, TexSwizzle(t, v1, isInt ? tU : tF, count, written, nullptr), isInt ? DT_U : DT_F, count, written);
		}
		// ivec2(vec2(x, y)*uf_tex<n>Scale): by component
		Val c[2] = { TexCoordAs(t, 0, type), TexCoordAs(t, 1, type) };
		uint32 xy;
		if (isLD)
		{
			uint32 parts[2] = { Op(SpvOpConvertSToF, tF, { c[0].id }), Op(SpvOpConvertSToF, tF, { c[1].id }) };
			xy = Construct(tV2F, parts, 2);
		}
		else
			xy = CoordVector(c, 2);
		scalePtr = Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassUniform, tV2F), { ufBlockVar, ConstI(texScaleMember[unit]) });
		uint32 scaled = Op(SpvOpFMul, tV2F, { xy, Load(tV2F, scalePtr) });
		uint32 tV2I = spv_type_vector(w, tI, 2);
		uint32 ip[2] = { Op(SpvOpConvertFToS, tI, { Op(SpvOpCompositeExtract, tF, { scaled, 0 }) }),
			Op(SpvOpConvertFToS, tI, { Op(SpvOpCompositeExtract, tF, { scaled, 1 }) }) };
		uint32 coord = Construct(tV2I, ip, 2);
		uint32 lod = isLD ? TexCoordAs(t, 3, DT_I).id : ConstI(0);
		if (!Ok())
			return false;
		uint32 image = Op(SpvOpImage, texImage[unit], { sampler });
		uint32 v = ImageOp(SpvOpImageFetch, isInt ? tV4U : tV4F, image, coord, 0, SpvImageOperandsLodMask, &lod, 1);
		return StoreTexResult(t, TexSwizzle(t, v, isInt ? tU : tF, count, written, nullptr), isInt ? DT_U : DT_F, count, written);
	}

	// one converted value into a channel, as the GLSL's per-channel lines
	void SpvShader::StoreTexChannel(LatteDecompilerTEXInstruction& t, uint32 chan, Val v)
	{
		Store(Channel(GPR(t.dstGpr), dt, chan), Convert(v, dt).id);
	}

	// _emitTEXGetTextureResInfoCode: the level's size and the level count
	bool SpvShader::EmitResInfo(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		const Latte::E_DIM dim = shader->textureUnitDim[unit];
		imageQuery = true;
		uint32 image = Op(SpvOpImage, texImage[unit], { Load(texType[unit], texVar[unit]) });
		uint32 levels = Op(SpvOpImageQueryLevels, tI, { image });
		uint32 mipIn = TexCoordAs(t, 3, DT_I).id;
		uint32 mip = Ext(GLSLstd450SClamp, tI, { mipIn, ConstI(0), Op(SpvOpISub, tI, { levels, ConstI(1) }) });
		image = Op(SpvOpImage, texImage[unit], { Load(texType[unit], texVar[unit]) });
		const bool cube = dim == Latte::E_DIM::DIM_CUBEMAP;
		const bool two = dim == Latte::E_DIM::DIM_2D || cube;
		const bool one = dim == Latte::E_DIM::DIM_1D;
		uint32 size = Op(SpvOpImageQuerySizeLod, one ? tI : spv_type_vector(w, tI, two && !cube ? 2 : 3), { image, mip });
		if (cube)
		{
			// textureSize(...).xy
			static const uint8 xy[2] = { 0, 1 };
			size = Swizzle(size, tI, xy, 2);
		}
		uint32 info;
		if (one)
		{
			uint32 parts[4] = { size, ConstI(0), ConstI(0), levels };
			info = Construct(tV4I, parts, 4);
		}
		else
		{
			uint32 parts[3] = { size, two ? ConstI(0) : levels, levels };
			info = Construct(tV4I, parts, two ? 3 : 2);
		}
		for (uint32 f = 0; f < 4; f++)
		{
			const uint8 sel = t.dstSel[f];
			if (sel == 7)
				continue;
			Val v = sel < 4 ? Run(Op(SpvOpCompositeExtract, tI, { info, sel }), DT_I) : K(DT_I, sel == 5 ? 1u : 0u);
			StoreTexChannel(t, f, v);
		}
		return Ok();
	}

	// _emitTEXGetCompTexLodCode: textureQueryLod()'s level, and it rounded
	// and clamped to the levels there are
	bool SpvShader::EmitCompTexLod(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		const Latte::E_DIM dim = shader->textureUnitDim[unit];
		if (NotPixel())
			return Decline("texture level query in a vertex shader");
		imageQuery = true;
		uint32 sampler = Load(texType[unit], texVar[unit]);
		const uint32 n = dim == Latte::E_DIM::DIM_3D || dim == Latte::E_DIM::DIM_CUBEMAP ? 3 : dim == Latte::E_DIM::DIM_1D ? 1 : 2;
		Val c[3];
		for (uint32 i = 0; i < n; i++)
			c[i] = TexCoord(t, (sint32)i);
		uint32 coord = n == 1 ? c[0].id : CoordVector(c, n);
		uint32 lod = Op(SpvOpCompositeExtract, tF, { Op(SpvOpImageQueryLod, tV2F, { sampler, coord }), 1 });
		uint32 rounded = Ext(GLSLstd450Floor, tF, { Op(SpvOpFAdd, tF, { lod, ConstF(0.5f) }) });
		uint32 image = Op(SpvOpImage, texImage[unit], { Load(texType[unit], texVar[unit]) });
		uint32 top = Op(SpvOpConvertSToF, tF, { Op(SpvOpISub, tI, { Op(SpvOpImageQueryLevels, tI, { image }), ConstI(1) }) });
		uint32 level = Ext(GLSLstd450FClamp, tF, { rounded, ConstF(0.0f), top });
		for (uint32 f = 0; f < 4; f++)
		{
			const uint8 sel = t.dstSel[f];
			if (sel == 7)
				continue;
			Val v = sel < 4 ? Run((sel & 1) ? level : lod, DT_F) : KF(sel == 5 ? 1.0 : 0.0);
			StoreTexChannel(t, f, v);
		}
		return Ok();
	}

	// _getTexGPRAccess of one channel: a 1D's coordinate, a reference
	uint32 SpvShader::TexGPRAccess1(LatteDecompilerTEXInstruction& t, sint32 component)
	{
		uint8 sel = (uint8)t.textureFetch.srcSel[component];
		return TexGPRAccess(t, &sel, 1);
	}

	// SAMPLE_G: textureGrad() of a 1D or 2D with gradH and gradV's x or xy
	bool SpvShader::EmitGradSample(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		const Latte::E_DIM dim = shader->textureUnitDim[unit];
		const bool isInt = shader->textureIsIntegerFormat[unit];
		const bool one = dim == Latte::E_DIM::DIM_1D;
		if (shader->textureUsesDepthCompare[unit])
			return Decline("gradients of a shadow sampler");
		if (dim != Latte::E_DIM::DIM_2D && !one)
			return Decline("gradient dimension");
		if (!gradVar[0])
			return Decline("gradients never set");
		if (!one && ActiveSettings::ForceSamplerRoundToPrecision())
			return Decline("sampler round to precision");
		uint32 written[4];
		const uint32 count = TexWritten(t, written);
		if (!count)
			return false;
		const bool hasOffset = t.textureFetch.offsetX != 0 || t.textureFetch.offsetY != 0 || t.textureFetch.offsetZ != 0;
		uint32 sampler = Load(texType[unit], texVar[unit]);
		uint32 coord;
		if (one)
			coord = TexGPRAccess1(t, 0);
		else
		{
			Val c[2] = { TexCoord(t, 0), TexCoord(t, 1) };
			coord = CoordVector(c, 2);
		}
		uint32 operands[3];
		uint32 nOperands = 0;
		static const uint8 xy[2] = { 0, 1 };
		for (uint32 g = 0; g < 2; g++)
			operands[nOperands++] = Swizzle(Load(tV4F, gradVar[g]), tF, xy, one ? 1 : 2);
		uint32 mask = SpvImageOperandsGradMask;
		if (hasOffset)
		{
			mask |= SpvImageOperandsConstOffsetMask;
			operands[nOperands++] = TexOffset(t, dim);
		}
		if (!Ok())
			return false;
		uint32 sample = ImageOp(SpvOpImageSampleExplicitLod, isInt ? tV4U : tV4F, sampler, coord, 0, mask, operands, nOperands);
		return StoreTexResult(t, TexSwizzle(t, sample, isInt ? tU : tF, count, written, nullptr), isInt ? DT_U : DT_F, count, written);
	}

	// _emitTEXGetGradientsHV: dFdx() or dFdy() of the coordinates' vec4
	bool SpvShader::EmitGetGradients(LatteDecompilerTEXInstruction& t)
	{
		if (NotPixel())
			return Decline("gradients in a vertex shader");
		Val c[4];
		for (uint32 i = 0; i < 4; i++)
			c[i] = TexCoord(t, (sint32)i);
		if (c[0].k && c[1].k && c[2].k && c[3].k)
			return Decline("gradient of constants"); // folded
		uint32 g = Op(t.opcode == GPU7_TEX_INST_GET_GRADIENTS_H ? SpvOpDPdx : SpvOpDPdy, tV4F, { CoordVector(c, 4) });
		for (uint32 f = 0; f < 4; f++)
		{
			const uint8 sel = t.dstSel[f];
			if (sel == 7)
				continue;
			Val v = sel < 4 ? Run(Op(SpvOpCompositeExtract, tF, { g, sel }), DT_F) : KF(sel == 5 ? 1.0 : 0.0);
			StoreTexChannel(t, f, v);
		}
		return Ok();
	}

	// _emitTEXSetGradientsHV: gradH or gradV = vec4(the coordinates)
	bool SpvShader::EmitSetGradients(LatteDecompilerTEXInstruction& t)
	{
		Val c[4];
		for (uint32 i = 0; i < 4; i++)
			c[i] = TexCoord(t, (sint32)i);
		Store(gradVar[t.opcode == GPU7_TEX_INST_SET_GRADIENTS_H ? 0 : 1], CoordVector(c, 4));
		return Ok();
	}

	// vec4(redcCUBEReverse(the register's xy, int(z)), cubeMapArrayIndex<n>)
	uint32 SpvShader::CubeCoord(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		uint8 xy[2] = { (uint8)t.textureFetch.srcSel[0], (uint8)t.textureFetch.srcSel[1] };
		uint32 st = TexGPRAccess(t, xy, 2);
		Val z = TexCoord(t, 2);
		// int() of a constant folds
		uint32 face = z.k ? ConstI((sint32)z.d) : Op(SpvOpConvertFToS, tI, { z.id });
		const uint32 types[2] = { spv_type_vector(w, tF, 2), tI };
		const uint32 args[2] = { st, face };
		uint32 temps[2];
		uint32 dir = CallTyped(CubeReverseFunction(), spv_type_vector(w, tF, 3), types, args, 2, temps);
		uint32 parts[2] = { dir, Load(tF, cubeIndexVar[unit]) };
		return Construct(tV4F, parts, 2);
	}

	// _emitTEXSetCubemapIndexCode: cubeMapArrayIndex<n> = the register's x
	bool SpvShader::EmitSetCubeIndex(LatteDecompilerTEXInstruction& t)
	{
		const sint32 unit = t.textureFetch.textureIndex;
		if (unit < 0 || unit >= LATTE_NUM_MAX_TEX_UNITS || !cubeIndexVar[unit])
			return Decline("cube map index of no cube map");
		uint8 sel = (uint8)t.textureFetch.srcSel[0];
		if (sel >= 4)
			Store(cubeIndexVar[unit], ConstF(sel == 5 ? 1.0f : 0.0f)); // a constant select
		else
			Store(cubeIndexVar[unit], TexGPRAccess(t, &sel, 1));
		return Ok();
	}

	// _emitTEXClauseCode: each instruction by its kind
	bool SpvShader::EmitTEXClause(LatteDecompilerCFInstruction& cf)
	{
		for (auto& t : cf.instructionsTEX)
		{
			const uint32 op = t.opcode;
			const sint32 unit = t.textureFetch.textureIndex;
			const bool sample = op == GPU7_TEX_INST_SAMPLE || op == GPU7_TEX_INST_SAMPLE_L || op == GPU7_TEX_INST_SAMPLE_LB ||
				op == GPU7_TEX_INST_SAMPLE_LZ || op == GPU7_TEX_INST_SAMPLE_C || op == GPU7_TEX_INST_SAMPLE_C_L ||
				op == GPU7_TEX_INST_SAMPLE_C_LZ || op == GPU7_TEX_INST_FETCH4 || op == GPU7_TEX_INST_SAMPLE_G || op == GPU7_TEX_INST_LD;
			const bool query = op == GPU7_TEX_INST_GET_TEXTURE_RESINFO || op == GPU7_TEX_INST_GET_COMP_TEX_LOD;
			if (op == GPU7_TEX_INST_GET_GRADIENTS_H || op == GPU7_TEX_INST_GET_GRADIENTS_V)
			{
				if (!EmitGetGradients(t))
					return false;
				continue;
			}
			if (op == GPU7_TEX_INST_SET_CUBEMAP_INDEX)
			{
				if (!EmitSetCubeIndex(t))
					return false;
				continue;
			}
			if (op == GPU7_TEX_INST_SET_GRADIENTS_H || op == GPU7_TEX_INST_SET_GRADIENTS_V)
			{
				if (!EmitSetGradients(t))
					return false;
				continue;
			}
			if (op == GPU7_TEX_INST_VFETCH)
			{
				// a uniform buffer's vec4 at a register's index
				if (unit >= 0x80 && unit <= 0x8F)
				{
					if (!EmitUniformFetch(t))
						return false;
					continue;
				}
				// a geometry shader's read of the ring; vertex fetches else
				if (!isGeometry || unit != 0x9F)
					return Decline("vertex fetch");
				if (!EmitRingRead(t))
					return false;
				continue;
			}
			if (!sample && !query)
				return Decline("texture instruction kind");
			if (unit < 0 || unit >= LATTE_NUM_MAX_TEX_UNITS)
			{
				if (sample)
					continue; // the GLSL emitter writes nothing for it
				return Decline("texture unit out of range");
			}
			if (!texVar[unit])
				return Decline("texture unit not declared");
			bool ok;
			const bool texel = op == GPU7_TEX_INST_LD || (op == GPU7_TEX_INST_SAMPLE && t.textureFetch.unnormalized[0] &&
				t.textureFetch.unnormalized[1] && t.textureFetch.unnormalized[2] && t.textureFetch.unnormalized[3]);
			if (texel)
				ok = EmitTexelFetch(t);
			else if (op == GPU7_TEX_INST_SAMPLE_C || op == GPU7_TEX_INST_SAMPLE_C_L || op == GPU7_TEX_INST_SAMPLE_C_LZ)
				ok = EmitShadowSample(t);
			else if (op == GPU7_TEX_INST_FETCH4)
				ok = EmitGather(t);
			else if (op == GPU7_TEX_INST_GET_TEXTURE_RESINFO)
				ok = EmitResInfo(t);
			else if (op == GPU7_TEX_INST_GET_COMP_TEX_LOD)
				ok = EmitCompTexLod(t);
			else if (op == GPU7_TEX_INST_SAMPLE_G)
				ok = EmitGradSample(t);
			else
				ok = EmitSample(t);
			if (!ok)
				return false;
		}
		return Ok();
	}

	uint32 SpvShader::BuiltinInput(uint32 builtin)
	{
		uint32& var = builtin == SpvBuiltInVertexIndex ? vertexIndexVar : instanceIndexVar;
		if (!var)
		{
			var = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassInput, tI), SpvStorageClassInput);
			Decorate(var, SpvDecorationBuiltIn, { builtin });
			interfaceIds[interfaceCount++] = var;
		}
		return var;
	}

	// _emitAttributes, gl_PerVertex and _emitVSExports
	bool SpvShader::DeclareVertexIO()
	{
		uint32 ptrIn = spv_type_pointer(w, SpvStorageClassInput, tV4U);
		for (uint32 i = 0; i < LATTE_NUM_MAX_ATTRIBUTE_LOCATIONS; i++)
		{
			if (!ctx->analyzer.inputAttributSemanticMask[i])
				continue;
			attrVar[i] = spv_global_variable(w, ptrIn, SpvStorageClassInput);
			Decorate(attrVar[i], SpvDecorationLocation, { (uint32)ctx->output->resourceMappingVK.attributeMapping[i] });
			interfaceIds[interfaceCount++] = attrVar[i];
		}
		if (usesGS)
		{
			// V2G_LAYOUT out Vertex { ivec4 passV2GParameter0; ... } v2g; and
			// nothing else: the geometry shader makes the vertices
			v2gCount = shader->ringParameterCount;
			if (v2gCount == 0)
				return Ok();
			if (v2gCount > 32)
				return Decline("ring parameter count");
			uint32 members[32];
			for (uint32 i = 0; i < v2gCount; i++)
				members[i] = tV4I;
			uint32 vblock = spv_type_struct(w, members, v2gCount);
			Decorate(vblock, SpvDecorationBlock);
			v2gVar = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassOutput, vblock), SpvStorageClassOutput);
			Decorate(v2gVar, SpvDecorationLocation, { 0 });
			interfaceIds[interfaceCount++] = v2gVar;
			return Ok();
		}
		// out gl_PerVertex { vec4 gl_Position; float gl_PointSize; }, the
		// point size where the primitives are points
		const uint32 members[2] = { tV4F, tF };
		const bool pointSize = ctx->analyzer.outputPointSize;
		uint32 block = spv_type_struct(w, members, pointSize ? 2 : 1);
		Decorate(block, SpvDecorationBlock);
		uint32 member[4] = { block, 0, SpvDecorationBuiltIn, SpvBuiltInPosition };
		spv_inst(w, SPV_SEC_DECORATION, SpvOpMemberDecorate, member, 4);
		if (pointSize)
		{
			uint32 sizeMember[4] = { block, 1, SpvDecorationBuiltIn, SpvBuiltInPointSize };
			spv_inst(w, SPV_SEC_DECORATION, SpvOpMemberDecorate, sizeMember, 4);
		}
		perVertexVar = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassOutput, block), SpvStorageClassOutput);
		interfaceIds[interfaceCount++] = perVertexVar;
		LatteShaderPSInputTable* psInputTable = LatteSHRC_GetPSInputTable();
		uint32 ptrOut = spv_type_pointer(w, SpvStorageClassOutput, tV4F);
		for (uint32 i = 0; i < 32; i++)
		{
			if ((shader->outputParameterMask & (1u << i)) == 0)
				continue;
			uint32 semanticId = VertexOutSemantic(i);
			if (semanticId > LATTE_ANALYZER_IMPORT_INDEX_PARAM_MAX)
				continue;
			sint32 psInputIndex = -1;
			for (sint32 f = 0; f < psInputTable->count; f++)
			{
				if (psInputTable->import[f].semanticId == semanticId)
				{
					psInputIndex = f;
					break;
				}
			}
			if (psInputIndex < 0)
				continue;
			if (paramVar[semanticId])
				return Decline("parameter declared twice");
			uint32 v = spv_global_variable(w, ptrOut, SpvStorageClassOutput);
			Decorate(v, SpvDecorationLocation, { (uint32)psInputIndex });
			if (psInputTable->import[psInputIndex].isFlat)
				Decorate(v, SpvDecorationFlat);
			if (psInputTable->import[psInputIndex].isNoPerspective)
				Decorate(v, SpvDecorationNoPerspective);
			paramVar[semanticId] = v;
			interfaceIds[interfaceCount++] = v;
		}
		return true;
	}

	// _getVertexShaderOutParamSemanticId
	uint32 SpvShader::VertexOutSemantic(uint32 index)
	{
		uint32 semanticId = (ctx->contextRegisters[mmSPI_VS_OUT_ID_0 + (index / 4)] >> (8 * (index % 4))) & 0xFF;
		LatteShaderPSInputTable* psInputTable = LatteSHRC_GetPSInputTable();
		for (sint32 i = 0; i < psInputTable->count; i++)
		{
			if (psInputTable->import[i].semanticId == semanticId)
				return semanticId;
		}
		return 0xFF;
	}

	// LatteDecompiler_emitAttributeDecodeGLSL: attrDecoder from the
	// attribute, statement for statement
	bool SpvShader::EmitAttributeDecode(LatteParsedFetchShaderAttribute& attrib)
	{
		using E = Latte::E_HWFMT;
		if (attrib.attributeBufferIndex >= Latte::GPU_LIMITS::NUM_VERTEX_BUFFERS)
			return Decline("attribute of no vertex buffer");
		const uint32 a = attrib.semanticId;
		if (a >= LATTE_NUM_MAX_ATTRIBUTE_LOCATIONS || !attrVar[a])
			return Decline("attribute not declared");
		const E fmt = attrib.format;
		const uint8 nfa = attrib.nfa;
		const bool sgn = attrib.isSigned != 0;
		auto le32 = [&](uint32 n) {
			// attrDecoder = attrDataSem; or uvec4(attrDataSem.xyz,0) and the like
			if (n == 4)
				DecWrite("xyzw", AttrRead(a, "xyzw"), true);
			else
				DecWrite("xyzw", PadU(AttrRead(a, n == 3 ? "xyz" : n == 2 ? "xy" : "x"), n, 4), true);
		};
		auto be32 = [&](uint32 n) {
			if (n == 4)
			{
				DecWrite("xyzw", AttrRead(a, "xyzw"), true);
				DecSwap32("xyzw", true);
				return;
			}
			const char* c = n == 3 ? "xyz" : n == 2 ? "xy" : "x";
			DecWrite(c, AttrRead(a, c));
			DecSwap32(c, false);
			for (uint32 i = n; i < 4; i++)
				DecWrite(i == 1 ? "y" : i == 2 ? "z" : "w", ConstU(0));
		};
		auto be16x1 = [&] {
			DecWrite("xy", AttrRead(a, "xy"));
			DecSwap16("x", false);
			DecWrite("y", ConstU(0));
			DecWrite("z", ConstU(0));
			DecWrite("w", ConstU(0));
		};
		auto be16x2 = [&] {
			DecWrite("xy", AttrRead(a, "xy"));
			DecSwap16("xy", false);
			DecWrite("z", ConstU(0));
			DecWrite("w", ConstU(0));
		};
		auto be16x4 = [&] {
			DecWrite("xyzw", AttrRead(a, "xyzw"), true);
			DecSwap16("xyzw", true);
		};
		auto signExtend = [&](const char* chans, uint32 bit, uint32 ext) {
			for (const char* c = chans; *c; c++)
			{
				char one[2] = { *c, 0 };
				DecSignExtend(one, bit, ext);
			}
		};
		auto snorm = [&](const char* chans, const char* d) {
			for (const char* c = chans; *c; c++)
			{
				char one[2] = { *c, 0 };
				DecSNorm(one, d);
			}
		};
		// attrDecoder.c = floatBitsToUint(vecN(<uvec>)/d) and kin
		auto unormVec = [&](uint32 v, uint32 n, const char* d) {
			uint32 f = ToFloat(v, n, false);
			if (d)
				f = Op(SpvOpFDiv, VecF(n), { f, n == 1 ? KFText(d).id : SplatF((float)std::strtod(d, nullptr), n) });
			return BitcastU(f, n);
		};
		// floatBitsToInt(vec4(unpackHalf2x16(..),unpackHalf2x16(..))) into uvec4
		auto half4 = [&] {
			uint32 lo = DecUnpackHalf("x", "y");
			uint32 hi = DecUnpackHalf("z", "w");
			uint32 parts[4] = { Op(SpvOpCompositeExtract, tF, { lo, 0 }), Op(SpvOpCompositeExtract, tF, { lo, 1 }),
				Op(SpvOpCompositeExtract, tF, { hi, 0 }), Op(SpvOpCompositeExtract, tF, { hi, 1 }) };
			uint32 v = Construct(tV4F, parts, 4);
			DecWrite("xyzw", Op(SpvOpBitcast, tV4U, { Op(SpvOpBitcast, tV4I, { v }) }), true);
		};
		auto half2 = [&] {
			DecWrite("xy", Op(SpvOpBitcast, tV2U, { DecUnpackHalf("x", "y") }));
			DecWrite("zw", SplatU(0, 2));
		};
		switch (attrib.endianSwap)
		{
		case LatteConst::VertexFetchEndianMode::SWAP_U32:
			if (fmt == E::HWFMT_32_32_32_32_FLOAT && nfa == 2) be32(4);
			else if (fmt == E::HWFMT_32_32_32_FLOAT && nfa == 2) be32(3);
			else if (fmt == E::HWFMT_32_32_FLOAT && nfa == 2) be32(2);
			else if (fmt == E::HWFMT_32_FLOAT && nfa == 2) be32(1);
			else if (fmt == E::HWFMT_2_10_10_10 && nfa == 0)
			{
				be32(1);
				Dec1010102(sgn ? "511.0" : "1023.0", sgn);
			}
			else if (fmt == E::HWFMT_32_32_32_32 && nfa == 1 && !sgn) be32(4);
			else if (fmt == E::HWFMT_32_32_32 && nfa == 1 && !sgn) be32(3);
			else if (fmt == E::HWFMT_32_32 && nfa == 1 && !sgn) be32(2);
			else if (fmt == E::HWFMT_32 && nfa == 1) be32(1);
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 0 && !sgn)
				DecWrite("xyzw", unormVec(AttrRead(a, "wzyx"), 4, "255.0"), true);
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 0 && sgn)
			{
				DecWrite("xyzw", AttrRead(a, "wzyx"), true);
				signExtend("xyzw", 0x80, 0xFFFFFF00);
				snorm("xyzw", "127.0");
			}
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 1 && !sgn)
				DecWrite("xyzw", AttrRead(a, "wzyx"), true);
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 2 && !sgn)
				DecWrite("xyzw", unormVec(AttrRead(a, "wzyx"), 4, nullptr), true);
			else
				return Decline("attribute format");
			break;
		case LatteConst::VertexFetchEndianMode::SWAP_NONE:
			if (fmt == E::HWFMT_32_32_32_32_FLOAT && nfa == 2) le32(4);
			else if (fmt == E::HWFMT_32_32_32_FLOAT && nfa == 2) le32(3);
			else if (fmt == E::HWFMT_32_32_FLOAT && nfa == 2) le32(2);
			else if (fmt == E::HWFMT_32 && nfa == 1 && !sgn) le32(1);
			else if (fmt == E::HWFMT_2_10_10_10 && nfa == 0 && !sgn)
			{
				le32(1);
				Dec1010102("1023.0", false);
			}
			else if (fmt == E::HWFMT_16_16_16_16 && nfa == 0 && sgn)
			{
				le32(4);
				signExtend("xyzw", 0x8000, 0xFFFF0000);
				snorm("xyzw", "32767.0");
			}
			else if (fmt == E::HWFMT_16_16_16_16 && nfa == 2 && attrib.isSigned == 1)
			{
				le32(4);
				signExtend("xyzw", 0x8000, 0xFFFF0000);
				// floatBitsToUint(vec4(ivec4(attrDecoder)))
				uint32 v = DecRead("xyzw");
				uint32 parts[4];
				for (uint32 i = 0; i < 4; i++)
					parts[i] = Op(SpvOpBitcast, tI, { Op(SpvOpCompositeExtract, tU, { v, i }) });
				uint32 iv = Construct(tV4I, parts, 4);
				DecWrite("xyzw", BitcastU(ToFloat(iv, 4, true), 4), true);
			}
			else if (fmt == E::HWFMT_16_16_16_16_FLOAT && nfa == 2)
			{
				le32(4);
				half4();
			}
			else if (fmt == E::HWFMT_16_16 && nfa == 0 && sgn)
			{
				le32(2);
				signExtend("xy", 0x8000, 0xFFFF0000);
				snorm("xy", "32767.0");
			}
			else if (fmt == E::HWFMT_16_16_FLOAT && nfa == 2)
			{
				le32(2);
				half2();
			}
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 0 && !sgn)
				DecWrite("xyzw", unormVec(AttrRead(a, "xyzw"), 4, "255.0"), true);
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 0 && sgn)
			{
				DecWrite("xyzw", AttrRead(a, "xyzw"), true);
				signExtend("xyzw", 0x80, 0xFFFFFF00);
				snorm("xyzw", "127.0");
			}
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 1 && !sgn)
				DecWrite("xyzw", AttrRead(a, "xyzw"), true);
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 1 && sgn)
			{
				DecWrite("xyzw", AttrRead(a, "xyzw"), true);
				signExtend("xyzw", 0x80, 0xFFFFFF00);
			}
			else if (fmt == E::HWFMT_8_8_8_8 && nfa == 2 && !sgn)
			{
				// floatBitsToInt(vec4(attrDataSem.xyzw)), into the uvec4
				uint32 f = ToFloat(AttrRead(a, "xyzw"), 4, false);
				DecWrite("xyzw", Op(SpvOpBitcast, tV4U, { Op(SpvOpBitcast, tV4I, { f }) }), true);
			}
			else if (fmt == E::HWFMT_8_8 && nfa == 0 && !sgn)
			{
				DecWrite("xy", unormVec(AttrRead(a, "xy"), 2, "255.0"));
				DecWrite("zw", SplatU(0, 2));
			}
			else if (fmt == E::HWFMT_8_8 && nfa == 2 && !sgn)
			{
				DecWrite("xy", unormVec(AttrRead(a, "xy"), 2, nullptr));
				DecWrite("zw", SplatU(0, 2));
			}
			else if (fmt == E::HWFMT_8_8 && nfa == 0 && sgn)
			{
				DecWrite("xy", AttrRead(a, "xy"));
				signExtend("xy", 0x80, 0xFFFFFF00);
				snorm("xy", "127.0");
				DecWrite("zw", SplatU(0, 2));
			}
			else if (fmt == E::HWFMT_8_8 && nfa == 1 && !sgn)
				DecWrite("xyzw", PadU(AttrRead(a, "xy"), 2, 4), true);
			else if (fmt == E::HWFMT_8 && nfa == 0 && !sgn)
			{
				DecWrite("x", unormVec(AttrRead(a, "x"), 1, "255.0"));
				DecWrite("yzw", SplatU(0, 3));
			}
			else if (fmt == E::HWFMT_8 && nfa == 1 && !sgn)
				DecWrite("xyzw", PadU(AttrRead(a, "x"), 1, 4), true);
			else
				return Decline("attribute format");
			break;
		case LatteConst::VertexFetchEndianMode::SWAP_U16:
			if (fmt == E::HWFMT_16_16_16_16_FLOAT && nfa == 2)
			{
				be16x4();
				half4();
			}
			else if (fmt == E::HWFMT_16_16_16_16 && nfa == 0 && sgn)
			{
				be16x4();
				signExtend("xyzw", 0x8000, 0xFFFF0000);
				snorm("xyzw", "32767.0");
			}
			else if (fmt == E::HWFMT_16_16_16_16 && nfa == 0 && !sgn)
			{
				be16x4();
				DecIntToFloat("x", "65535.0");
				DecIntToFloat("y", "65535.0");
				DecIntToFloat("z", "65535.0");
				DecIntToFloat("w", "65535.0");
			}
			else if (fmt == E::HWFMT_16_16_16_16 && nfa == 2 && sgn)
			{
				be16x4();
				signExtend("xyzw", 0x8000, 0xFFFF0000);
				DecIntToFloat("x", nullptr);
				DecIntToFloat("y", nullptr);
				DecIntToFloat("z", nullptr);
				DecIntToFloat("w", nullptr);
			}
			else if (fmt == E::HWFMT_16_16_16_16 && nfa == 1 && sgn)
			{
				be16x4();
				signExtend("xyzw", 0x8000, 0xFFFF0000);
			}
			else if (fmt == E::HWFMT_16_16_16_16 && nfa == 1 && !sgn)
				be16x4();
			else if (fmt == E::HWFMT_16_16_FLOAT && nfa == 2)
			{
				be16x2();
				half2();
			}
			else if (fmt == E::HWFMT_16_16 && nfa == 0 && !sgn)
			{
				// floatBitsToUint(vec2(float(attrDecoder.x), float(attrDecoder.y))/65535.0)
				be16x2();
				uint32 parts[2] = { Op(SpvOpConvertUToF, tF, { DecRead("x") }), Op(SpvOpConvertUToF, tF, { DecRead("y") }) };
				uint32 v = Construct(tV2F, parts, 2);
				DecWrite("xy", BitcastU(Op(SpvOpFDiv, tV2F, { v, SplatF(65535.0f, 2) }), 2));
				DecWrite("zw", SplatU(0, 2));
			}
			else if (fmt == E::HWFMT_16_16 && nfa == 0 && sgn)
			{
				be16x2();
				signExtend("xy", 0x8000, 0xFFFF0000);
				snorm("xy", "32767.0");
				DecWrite("zw", SplatU(0, 2));
			}
			else if (fmt == E::HWFMT_16_16 && nfa == 1 && !sgn)
				be16x2();
			else if (fmt == E::HWFMT_16_16 && nfa == 1 && sgn)
			{
				be16x2();
				signExtend("xy", 0x8000, 0xFFFF0000);
				DecWrite("zw", SplatU(0, 2));
			}
			else if (fmt == E::HWFMT_16_16 && nfa == 2 && !sgn)
			{
				be16x2();
				uint32 parts[2] = { Op(SpvOpConvertUToF, tF, { DecRead("x") }), Op(SpvOpConvertUToF, tF, { DecRead("y") }) };
				DecWrite("xy", BitcastU(Construct(tV2F, parts, 2), 2));
				DecWrite("zw", SplatU(0, 2));
			}
			else if (fmt == E::HWFMT_16_16 && nfa == 2 && sgn)
			{
				be16x2();
				signExtend("xy", 0x8000, 0xFFFF0000);
				uint32 parts[2] = { Op(SpvOpConvertSToF, tF, { Op(SpvOpBitcast, tI, { DecRead("x") }) }),
					Op(SpvOpConvertSToF, tF, { Op(SpvOpBitcast, tI, { DecRead("y") }) }) };
				DecWrite("xy", BitcastU(Construct(tV2F, parts, 2), 2));
				DecWrite("zw", SplatU(0, 2));
			}
			else if (fmt == E::HWFMT_16 && nfa == 1 && !sgn)
				be16x1();
			else if (fmt == E::HWFMT_16 && nfa == 0 && !sgn)
			{
				be16x1();
				DecIntToFloat("x", "65535.0");
			}
			else
				return Decline("attribute format");
			break;
		default:
			return Decline("attribute endian swap");
		}
		return Ok();
	}

	// LatteDecompiler_emitAttributeImport: decoded into its register
	bool SpvShader::EmitAttributeImport(LatteParsedFetchShaderAttribute& attrib)
	{
		uint32 attributeShaderLoc = 0xFFFFFFFF;
		for (uint32 f = 0; f < 32; f++)
		{
			if (ctx->contextRegisters[mmSQ_VTX_SEMANTIC_0 + f] == attrib.semanticId)
			{
				attributeShaderLoc = f;
				break;
			}
		}
		if (attributeShaderLoc == 0xFFFFFFFF)
			return true;
		uint32 registerIndex = attributeShaderLoc + 1;
		if ((ctx->analyzer.gprUseMask[registerIndex / 8] & (1 << (registerIndex % 8))) == 0)
			return true;
		if (!EmitAttributeDecode(attrib))
			return false;
		// ivec4(int(attrDecoder.x), ..., 0, 0x3f800000), each through
		// intBitsToFloat into float registers
		uint32 parts[4];
		for (uint32 f = 0; f < 4; f++)
		{
			uint8 ds = attrib.ds[f];
			if (ds >= 6)
				ds = 4;
			uint32 v;
			if (ds < 4)
			{
				static const char* chan[4] = { "x", "y", "z", "w" };
				v = Op(SpvOpBitcast, tI, { DecRead(chan[ds]) });
			}
			else if (ds == 4)
				v = ConstI(0);
			else
				v = ConstI(attrib.nfa != 1 ? 0x3f800000 : 1);
			if (dt == DT_F)
				v = Op(SpvOpBitcast, tF, { v });
			parts[f] = v;
		}
		Store(GPR(registerIndex), Construct(Vec4Of(dt), parts, 4));
		return Ok();
	}

	// the vertex shader's inputs: R0 from the vertex and instance index,
	// then each attribute
	bool SpvShader::EmitVertexInputs()
	{
		if ((ctx->analyzer.gprUseMask[0] & 1) != 0)
		{
			uint32 parts[4] = { Load(tI, BuiltinInput(SpvBuiltInVertexIndex)), ConstI(0), ConstI(0), Load(tI, BuiltinInput(SpvBuiltInInstanceIndex)) };
			uint32 v = Construct(tV4I, parts, 4);
			if (dt == DT_F)
			{
				// R0f = floatBitsToInt(ivec4(...)): the ivec4 converted to
				// floats for the call, its result converted again for R0f,
				// as rslc takes the GLSL emitter's line
				v = Op(SpvOpConvertSToF, tV4F, { Op(SpvOpBitcast, tV4I, { Op(SpvOpConvertSToF, tV4F, { v }) }) });
			}
			Store(GPR(0), v);
		}
		LatteFetchShader* fs = ctx->fetchShader;
		if (!fs)
			return Decline("no fetch shader");
		for (auto& group : fs->bufferGroups)
		{
			for (sint32 i = 0; i < group.attribCount; i++)
			{
				if (!EmitAttributeImport(group.attrib[i]))
					return false;
			}
		}
		for (auto& group : fs->bufferGroupsInvalid)
		{
			for (sint32 i = 0; i < group.attribCount; i++)
			{
				if (!EmitAttributeImport(group.attrib[i]))
					return false;
			}
		}
		return Ok();
	}

	// _emitExportCode for a vertex shader: the position, as SET_POSITION
	// writes it, and the parameters
	bool SpvShader::EmitVertexExport(LatteDecompilerCFInstruction& cf)
	{
		if (usesGS)
			return Decline("vertex export with a geometry shader");
		if (cf.exportType == 1 && cf.exportArrayBase == GPU7_DECOMPILER_CF_EXPORT_BASE_POSITION)
		{
			uint32 c[4];
			for (uint32 k = 0; k < 4; k++)
				c[k] = ExportComponent(cf, 0, k).id;
			uint32 pos = Construct(tV4F, c, 4);
			if (windowMember >= 0)
			{
				// vec4 finalPos = ...; finalPos.xy = finalPos.xy *
				// uf_windowSpaceToClipSpaceTransform - vec2(1.0,1.0);
				const uint32 tV2F = spv_type_vector(w, tF, 2);
				uint32 finalPos = Local(tV4F, 0);
				Store(finalPos, pos);
				uint32 cur = Load(tV4F, finalPos);
				uint32 v2 = Op(SpvOpVectorShuffle, tV2F, { cur, cur, 0, 1 });
				uint32 tPtr = Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassUniform, tV2F), { ufBlockVar, ConstI(windowMember) });
				uint32 scaled = Op(SpvOpFMul, tV2F, { v2, Load(tV2F, tPtr) });
				uint32 ones[2] = { ConstF(1.0f), ConstF(1.0f) };
				uint32 moved = Op(SpvOpFSub, tV2F, { scaled, spv_const_composite(w, tV2F, ones, 2) });
				uint32 old = Load(tV4F, finalPos);
				Store(finalPos, Op(SpvOpVectorShuffle, tV4F, { old, moved, 4, 5, 2, 3 }));
				pos = Load(tV4F, finalPos);
			}
			SetPosition(pos);
			return Ok();
		}
		if (cf.exportType == 2 && cf.exportArrayBase < 32)
		{
			for (uint32 burst = 0; burst < cf.exportBurstCount + 1; burst++)
			{
				uint32 paramIndex = cf.exportArrayBase + burst;
				if (paramIndex >= 32)
					return Decline("parameter index");
				uint32 semanticId = VertexOutSemantic(paramIndex);
				if (semanticId == 0xFF)
					continue;
				uint8 sel[4];
				uint32 count = 0;
				for (uint32 k = 0; k < 4; k++)
				{
					if (cf.exportComponentSel[k] != 7)
						sel[count++] = (uint8)k;
				}
				if (count == 0)
					continue;
				if (semanticId > LATTE_ANALYZER_IMPORT_INDEX_PARAM_MAX || !paramVar[semanticId])
					return Decline("parameter not declared");
				uint32 c[4];
				for (uint32 k = 0; k < 4; k++)
					c[k] = ExportComponent(cf, burst, k).id;
				uint32 v = Construct(tV4F, c, 4);
				if (count == 4)
					Store(paramVar[semanticId], v);
				else
				{
					// passParameterSemN.<mask> = vec4(...).<mask>
					uint32 part = Swizzle(v, tF, sel, count);
					uint32 reg = Load(tV4F, paramVar[semanticId]);
					for (uint32 i = 0; i < count; i++)
						reg = Op(SpvOpCompositeInsert, tV4F, { count == 1 ? part : Op(SpvOpCompositeExtract, tF, { part, i }), reg, sel[i] });
					Store(paramVar[semanticId], reg);
				}
			}
			return Ok();
		}
		if (cf.exportType == 1 && cf.exportArrayBase == GPU7_DECOMPILER_CF_EXPORT_POINT_SIZE)
		{
			// gl_PointSize = (vec4(...)).x, where the primitives are points
			if (!ctx->analyzer.outputPointSize)
				return Ok();
			uint32 c[4];
			for (uint32 k = 0; k < 4; k++)
				c[k] = ExportComponent(cf, 0, k).id;
			uint32 v = Op(SpvOpCompositeExtract, tF, { Construct(tV4F, c, 4), 0 });
			Store(Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassOutput, tF), { perVertexVar, ConstI(1) }), v);
			return Ok();
		}
		return Decline("vertex export");
	}

	// a geometry shader's interface: rslc's own gl_PerVertex (the position,
	// the point size, a clip and a cull distance) in gl_in[] and out,
	// V2G_LAYOUT in Vertex { ivec4 passV2GParameter<n>; } v2g[], and the
	// copy shader's parameters as passG2PParameter<n> at n's location
	bool SpvShader::DeclareGeometryIO()
	{
		if (!ctx->parsedGSCopyShader)
			return Decline("no copy shader");
		switch (ctx->contextRegistersNew->VGT_PRIMITIVE_TYPE.get_PRIMITIVE_MODE())
		{
		case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::POINTS: gsInputVertices = 1; break;
		case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::TRIANGLES: gsInputVertices = 3; break;
		case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::LINE_STRIP: gsInputVertices = 4; break;
		default: return Decline("geometry input primitive");
		}
		const uint32 gsOutPrimType = ctx->contextRegisters[mmVGT_GS_OUT_PRIM_TYPE];
		if (gsOutPrimType > 2)
			return Decline("geometry output primitive");
		const uint32 floatArr = spv_type_array(w, tF, ConstI(1));
		const uint32 members[4] = { tV4F, tF, floatArr, floatArr };
		auto perVertexBlock = [&] {
			uint32 block = spv_type_struct(w, members, 4);
			Decorate(block, SpvDecorationBlock);
			const uint32 builtins[4] = { SpvBuiltInPosition, SpvBuiltInPointSize, SpvBuiltInClipDistance, SpvBuiltInCullDistance };
			for (uint32 i = 0; i < 4; i++)
			{
				uint32 words[4] = { block, i, SpvDecorationBuiltIn, builtins[i] };
				spv_inst(w, SPV_SEC_DECORATION, SpvOpMemberDecorate, words, 4);
			}
			return block;
		};
		const uint32 inCount = ConstI((sint32)gsInputVertices);
		uint32 glIn = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassInput, spv_type_array(w, perVertexBlock(), inCount)), SpvStorageClassInput);
		interfaceIds[interfaceCount++] = glIn;
		v2gCount = shader->ringParameterCountFromPrevStage;
		if (v2gCount == 0 || v2gCount > 32)
			return Decline("ring parameter count");
		uint32 vmembers[32];
		for (uint32 i = 0; i < v2gCount; i++)
			vmembers[i] = tV4I;
		uint32 vblock = spv_type_struct(w, vmembers, v2gCount);
		Decorate(vblock, SpvDecorationBlock);
		v2gVar = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassInput, spv_type_array(w, vblock, inCount)), SpvStorageClassInput);
		Decorate(v2gVar, SpvDecorationLocation, { 0 });
		interfaceIds[interfaceCount++] = v2gVar;
		const uint32 ptrOut = spv_type_pointer(w, SpvStorageClassOutput, tV4F);
		for (sint32 i = 0; i < ctx->parsedGSCopyShader->numParam; i++)
		{
			auto& m = ctx->parsedGSCopyShader->paramMapping[i];
			if (m.exportType != 2)
				continue;
			if (g2pVar[m.exportParam])
				return Decline("parameter declared twice");
			for (uint32 j = 0; j < 256; j++)
			{
				if (g2pVar[j] && (j & 0x7F) == (m.exportParam & 0x7F))
					return Decline("parameter declared twice");
			}
			uint32 v = spv_global_variable(w, ptrOut, SpvStorageClassOutput);
			Decorate(v, SpvDecorationLocation, { (uint32)(m.exportParam & 0x7F) });
			g2pVar[m.exportParam] = v;
			interfaceIds[interfaceCount++] = v;
		}
		perVertexVar = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassOutput, perVertexBlock()), SpvStorageClassOutput);
		interfaceIds[interfaceCount++] = perVertexVar;
		return Ok();
	}

	// _emitGSReadInputVFetchCode: R<d>.<the written> = (v2g[0].
	// passV2GParameter<offset/16>.<their selects>), the int vector converted
	// to the registers' type (the first vertex's, as the GLSL emitter reads)
	bool SpvShader::EmitRingRead(LatteDecompilerTEXInstruction& t)
	{
		if (t.textureFetch.srcSel[0] >= 4 || t.textureFetch.srcSel[1] >= 4)
			return Decline("ring read select");
		const uint32 param = t.textureFetch.offset / 16;
		if (param >= v2gCount || !v2gVar)
			return Decline("ring parameter not declared");
		uint32 dst[4], src[4];
		uint32 count = 0;
		for (uint32 f = 0; f < 4; f++)
		{
			if (t.dstSel[f] < 4)
			{
				dst[count] = f;
				src[count++] = t.dstSel[f];
			}
			else if (t.dstSel[f] != 7)
				return Decline("ring read constant");
		}
		if (count == 0)
			return Decline("ring read of no channel");
		uint32 member = Load(tV4I, Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassInput, tV4I), { v2gVar, ConstI(0), ConstI((sint32)param) }));
		const uint32 tVnI = count == 1 ? tI : spv_type_vector(w, tI, count);
		uint32 v;
		if (count == 1)
			v = Op(SpvOpCompositeExtract, tI, { member, src[0] });
		else if (count == 2)
			v = Op(SpvOpVectorShuffle, tVnI, { member, member, src[0], src[1] });
		else if (count == 3)
			v = Op(SpvOpVectorShuffle, tVnI, { member, member, src[0], src[1], src[2] });
		else
			v = Op(SpvOpVectorShuffle, tVnI, { member, member, src[0], src[1], src[2], src[3] });
		if (dt == DT_F)
			v = Op(SpvOpBitcast, count == 1 ? tF : spv_type_vector(w, tF, count), { v });
		const uint32 elem = TypeOf(dt);
		for (uint32 i = 0; i < count; i++)
			Store(Channel(GPR(t.dstGpr), dt, dst[i]), count == 1 ? v : Op(SpvOpCompositeExtract, elem, { v, i }));
		return Ok();
	}

	// _emitUniformBuffers: each buffer read, as the tracker sizes it, a block
	// of a vec4 array (std140) at its binding
	bool SpvShader::DeclareUniformBuffers()
	{
		if (shader->uniformMode != LATTE_DECOMPILER_UNIFORM_MODE_FULL_CBANK)
			return Ok();
		uint32 decorated[LATTE_NUM_MAX_UNIFORM_BUFFERS];
		uint32 decoratedCount = 0;
		for (uint32 i = 0; i < LATTE_NUM_MAX_UNIFORM_BUFFERS; i++)
		{
			if (!ctx->analyzer.uniformBufferAccessTracker[i].HasAccess())
				continue;
			const sint32 binding = ctx->output->resourceMappingVK.uniformBuffersBindingPoint[i];
			if (binding < 0)
				return Decline("uniform buffer binding");
			const uint32 size = ctx->analyzer.uniformBufferAccessTracker[i].DetermineSize(ctx->shaderBaseHash, LATTE_GLSL_DYNAMIC_UNIFORM_BLOCK_SIZE);
			const uint32 arr = spv_type_array(w, tV4F, ConstI((sint32)size));
			// (an array type of a size another buffer has decorated once)
			bool seen = false;
			for (uint32 d = 0; d < decoratedCount; d++)
				seen |= decorated[d] == arr;
			if (!seen)
			{
				Decorate(arr, SpvDecorationArrayStride, { 16 });
				decorated[decoratedCount++] = arr;
			}
			uint32 block = spv_type_struct(w, &arr, 1);
			uint32 member[4] = { block, 0, SpvDecorationOffset, 0 };
			spv_inst(w, SPV_SEC_DECORATION, SpvOpMemberDecorate, member, 4);
			Decorate(block, SpvDecorationBlock);
			ubVar[i] = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassUniform, block), SpvStorageClassUniform);
			Decorate(ubVar[i], SpvDecorationDescriptorSet, { (uint32)ctx->output->resourceMappingVK.setIndex });
			Decorate(ubVar[i], SpvDecorationBinding, { (uint32)binding });
		}
		return Ok();
	}

	// _emitTEXVFetchCode of a uniform buffer: R<d>.<the written> =
	// floatBitsToInt(uf_block<stage><n>[<the index register's channel, as an
	// int>].<their selects>), the float vector where the registers are floats
	bool SpvShader::EmitUniformFetch(LatteDecompilerTEXInstruction& t)
	{
		const uint32 buffer = (uint32)t.textureFetch.textureIndex - 0x80;
		if (!ubVar[buffer])
			return Decline("uniform buffer not declared");
		if (t.textureFetch.srcSel[0] >= 4)
			return Decline("uniform fetch index select");
		uint32 dst[4], src[4];
		uint32 count = 0;
		for (uint32 f = 0; f < 4; f++)
		{
			if (t.dstSel[f] < 4)
			{
				dst[count] = f;
				src[count++] = t.dstSel[f];
			}
			else if (t.dstSel[f] != 7)
				return Decline("uniform fetch constant");
		}
		if (count == 0)
			return Decline("uniform fetch of no channel");
		uint32 index = Convert({ ReadChannel(GPR(t.srcGpr), t.textureFetch.srcSel[0]), dt }, DT_I).id;
		uint32 vec = Load(tV4F, Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassUniform, tV4F), { ubVar[buffer], ConstI(0), index }));
		const uint32 tVnF = count == 1 ? tF : spv_type_vector(w, tF, count);
		uint32 v;
		if (count == 1)
			v = Op(SpvOpCompositeExtract, tF, { vec, src[0] });
		else if (count == 2)
			v = Op(SpvOpVectorShuffle, tVnF, { vec, vec, src[0], src[1] });
		else if (count == 3)
			v = Op(SpvOpVectorShuffle, tVnF, { vec, vec, src[0], src[1], src[2] });
		else
			v = Op(SpvOpVectorShuffle, tVnF, { vec, vec, src[0], src[1], src[2], src[3] });
		if (dt == DT_I)
			v = Op(SpvOpBitcast, count == 1 ? tI : spv_type_vector(w, tI, count), { v });
		const uint32 elem = TypeOf(dt);
		for (uint32 i = 0; i < count; i++)
			Store(Channel(GPR(t.dstGpr), dt, dst[i]), count == 1 ? v : Op(SpvOpCompositeExtract, elem, { v, i }));
		return Ok();
	}

	// layout(set, binding) buffer StreamoutBuffer { int sb_buffer[]; };, as
	// rslc makes it (a BufferBlock in the Uniform storage class), where
	// streamout is on and the shader writes it
	bool SpvShader::DeclareStreamout()
	{
		auto& a = ctx->analyzer;
		if ((a.hasStreamoutWrite || a.hasStreamoutEnable) && !ctx->options->useTFViaSSBO)
			return Decline("transform feedback outputs");
		if (isVertex && usesGS && a.useSSBOForStreamout)
			return Decline("streamout through a geometry shader's vertex shader");
		if (!NotPixel() || !a.hasStreamoutEnable || !a.useSSBOForStreamout || !a.hasStreamoutWrite)
			return Ok();
		const uint32 arr = spv_type_runtime_array(w, tI);
		Decorate(arr, SpvDecorationArrayStride, { 4 });
		uint32 block = spv_type_struct(w, &arr, 1);
		uint32 member[4] = { block, 0, SpvDecorationOffset, 0 };
		spv_inst(w, SPV_SEC_DECORATION, SpvOpMemberDecorate, member, 4);
		Decorate(block, SpvDecorationBufferBlock);
		sbVar = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassUniform, block), SpvStorageClassUniform);
		Decorate(sbVar, SpvDecorationDescriptorSet, { (uint32)ctx->output->resourceMappingVK.setIndex });
		Decorate(sbVar, SpvDecorationBinding, { (uint32)ctx->output->resourceMappingVK.getTFStorageBufferBindingPoint() });
		return Ok();
	}

	// int sbBase<n> = uf_streamoutBufferBase<n>/4 + (gl_VertexID +
	// uf_verticesPerInstance * gl_InstanceID)*<stride>; a geometry shader's
	// from (gl_PrimitiveIDIn * <its vertices>)
	bool SpvShader::EmitStreamoutBases()
	{
		if (!ctx->analyzer.useSSBOForStreamout || !NotPixel())
			return Ok();
		uint32 maxVertices = 0;
		if (isGeometry)
		{
			const uint32 bytesPerVertex = ctx->contextRegisters[mmSQ_GS_VERT_ITEMSIZE] * 4;
			if (bytesPerVertex == 0)
				return Decline("geometry vertex size");
			maxVertices = ((ctx->contextRegisters[mmSQ_GSVS_RING_ITEMSIZE] & 0x7FFF) * 4) / bytesPerVertex;
		}
		for (uint32 i = 0; i < LATTE_NUM_STREAMOUT_BUFFER; i++)
		{
			if (!ctx->output->streamoutBufferWriteMask[i])
				continue;
			if (sbBaseMember[i] < 0 || verticesPerInstanceMember < 0)
				return Decline("streamout base not declared");
			const uint32 ptrI = spv_type_pointer(w, SpvStorageClassUniform, tI);
			uint32 base = Op(SpvOpSDiv, tI, { Load(tI, Op(SpvOpAccessChain, ptrI, { ufBlockVar, ConstI(sbBaseMember[i]) })), ConstI(4) });
			uint32 index;
			if (isGeometry)
			{
				if (!primitiveIdVar)
				{
					primitiveIdVar = spv_global_variable(w, spv_type_pointer(w, SpvStorageClassInput, tI), SpvStorageClassInput);
					Decorate(primitiveIdVar, SpvDecorationBuiltIn, { SpvBuiltInPrimitiveId });
					interfaceIds[interfaceCount++] = primitiveIdVar;
				}
				index = Op(SpvOpIMul, tI, { Load(tI, primitiveIdVar), ConstI((sint32)maxVertices) });
			}
			else
			{
				uint32 vid = Load(tI, BuiltinInput(SpvBuiltInVertexIndex));
				uint32 perInstance = Load(tI, Op(SpvOpAccessChain, ptrI, { ufBlockVar, ConstI(verticesPerInstanceMember) }));
				uint32 iid = Load(tI, BuiltinInput(SpvBuiltInInstanceIndex));
				index = Op(SpvOpIAdd, tI, { vid, Op(SpvOpIMul, tI, { perInstance, iid }) });
			}
			const sint32 stride = (sint32)(ctx->output->streamoutBufferStride[i] / 4);
			sbBaseVar[i] = Local(tI, Op(SpvOpIAdd, tI, { base, Op(SpvOpIMul, tI, { index, ConstI(stride) }) }));
		}
		return Ok();
	}

	// sb_buffer[sbBase<buffer> + dword] = value
	void SpvShader::StreamStore(uint32 buffer, uint32 dword, uint32 value)
	{
		if (!sbVar || buffer >= LATTE_NUM_STREAMOUT_BUFFER || !sbBaseVar[buffer])
		{
			Decline("streamout buffer not declared");
			return;
		}
		uint32 at = Op(SpvOpIAdd, tI, { Load(tI, sbBaseVar[buffer]), ConstI((sint32)dword) });
		Store(Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassUniform, tI), { sbVar, ConstI(0), at }), value);
	}

	// _emitStreamWriteCode: a vertex shader's MEM_STREAM<n>_WRITE, each
	// channel in the mask (of the array's) to sb_buffer[sbBase<n> + <base>
	// + i]; nothing where streamout is off
	bool SpvShader::EmitStreamWrite(LatteDecompilerCFInstruction& cf)
	{
		if (!ctx->analyzer.hasStreamoutEnable || !isVertex)
			return Ok();
		const uint32 buffer = cf.type == GPU7_CF_INST_MEM_STREAM0_WRITE ? 0 : 1;
		const uint32 arraySize = cf.memWriteArraySize + 1;
		for (uint32 i = 0; i < arraySize && i < 4; i++)
		{
			if ((cf.memWriteCompMask & (1u << i)) == 0)
				continue;
			uint32 v = Convert({ ReadChannel(GPR(cf.exportSourceGPR), i), dt }, DT_I).id;
			StreamStore(buffer, cf.exportArrayBase + i, v);
		}
		return Ok();
	}

	// SET_POSITION(pos): gl_Position = pos, and its z to 0..1 where the clip
	// space is OpenGL's: gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0
	void SpvShader::SetPosition(uint32 pos)
	{
		uint32 posPtr = Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassOutput, tV4F), { perVertexVar, ConstI(0) });
		Store(posPtr, pos);
		if (!ctx->contextRegistersNew->PA_CL_CLIP_CNTL.get_DX_CLIP_SPACE_DEF())
		{
			uint32 ptrF = spv_type_pointer(w, SpvStorageClassOutput, tF);
			uint32 z = Load(tF, Op(SpvOpAccessChain, ptrF, { perVertexVar, ConstI(0), ConstI(2) }));
			uint32 wv = Load(tF, Op(SpvOpAccessChain, ptrF, { perVertexVar, ConstI(0), ConstI(3) }));
			uint32 r = Op(SpvOpFDiv, tF, { Op(SpvOpFAdd, tF, { z, wv }), ConstF(2.0f) });
			Store(Op(SpvOpAccessChain, ptrF, { perVertexVar, ConstI(0), ConstI(2) }), r);
		}
	}

	// _emitCFRingWriteCode of a geometry shader: each of the burst to the
	// output the copy shader exports from its offset in the vertex, the
	// position as { vec4 pos = vec4(0.0,0.0,0.0,1.0); pos.<mask> = ...;
	// SET_POSITION(pos); }, a parameter as passG2PParameter<n>.<mask> = ...
	bool SpvShader::EmitGeometryRingWrite(LatteDecompilerCFInstruction& cf)
	{
		if (cf.memWriteElemSize != 3 || (cf.exportArrayBase & 3) != 0)
			return Decline("ring write layout");
		uint32 bytesPerVertex = ctx->contextRegisters[mmSQ_GS_VERT_ITEMSIZE] * 4;
		if (bytesPerVertex == 0)
			bytesPerVertex = 1;
		if (ctx->analyzer.hasStreamoutEnable)
		{
			// to the stream the copy shader writes from that offset instead:
			// sb_buffer[sbBase<n> + <its dword> + i] = the register's channel
			// i, each masked in; none past an offset it does not write
			for (uint32 burst = 0; burst < cf.exportBurstCount + 1; burst++)
			{
				const uint32 offset = (cf.exportArrayBase * 4 + burst * 0x10) % bytesPerVertex;
				const LatteGSCopyShaderStreamWrite_t* sw = nullptr;
				for (auto& it : ctx->parsedGSCopyShader->list_streamWrites)
				{
					if (it.offset == offset)
					{
						sw = &it;
						break;
					}
				}
				if (!sw)
					return Ok();
				for (uint32 i = 0; i < 4; i++)
				{
					if ((cf.memWriteCompMask & (1u << i)) == 0)
						continue;
					uint32 v = Convert({ ReadChannel(GPR(cf.exportSourceGPR + burst), i), dt }, DT_I).id;
					StreamStore(sw->bufferIndex, sw->exportArrayBase + i, v);
				}
			}
			return Ok();
		}
		const uint32 parameterOffset = (cf.exportArrayBase * 4) % bytesPerVertex;
		uint8 sel[4];
		uint32 count = 0;
		for (uint32 k = 0; k < 4; k++)
		{
			if (cf.memWriteCompMask & (1u << k))
				sel[count++] = (uint8)k;
		}
		if (count == 0)
			return Decline("ring write of no channel");
		const uint32 tVnF = count == 1 ? tF : spv_type_vector(w, tF, count);
		for (uint32 burst = 0; burst < cf.exportBurstCount + 1; burst++)
		{
			uint32 exportType = 0, exportBase = 0;
			if (!LatteGSCopyShaderParser_getExportTypeByOffset(ctx->parsedGSCopyShader, parameterOffset + burst * 16, &exportType, &exportBase))
				return Decline("ring write the copy shader does not export");
			const bool position = exportType == 1 && exportBase == GPU7_DECOMPILER_CF_EXPORT_BASE_POSITION;
			if (!position && !(exportType == 2 && exportBase < 16))
				continue; // the GLSL emitter writes nothing for it
			// vecN(the register's masked channels, as floats)
			uint32 c[4];
			for (uint32 i = 0; i < count; i++)
				c[i] = Convert({ ReadChannel(GPR(cf.exportSourceGPR + burst), sel[i]), dt }, DT_F).id;
			uint32 part = count == 1 ? c[0] : Construct(tVnF, c, count);
			uint32 target;
			if (position)
			{
				const uint32 init[4] = { ConstF(0.0f), ConstF(0.0f), ConstF(0.0f), ConstF(1.0f) };
				target = Local(tV4F, spv_const_composite(w, tV4F, init, 4));
			}
			else
			{
				target = g2pVar[exportBase];
				if (!target)
					return Decline("geometry output not declared");
			}
			uint32 reg = Load(tV4F, target);
			for (uint32 i = 0; i < count; i++)
				reg = Op(SpvOpCompositeInsert, tV4F, { count == 1 ? part : Op(SpvOpCompositeExtract, tF, { part, i }), reg, sel[i] });
			Store(target, reg);
			if (position)
				SetPosition(Load(tV4F, target));
		}
		return Ok();
	}

	// EMIT_VERTEX, under the active mask: gl_PointSize = uf_pointSize where
	// the render state's, then EmitVertex()
	bool SpvShader::EmitEmitVertex(LatteDecompilerCFInstruction& cf)
	{
		if (!isGeometry)
			return Decline("emit vertex outside a geometry shader");
		auto emit = [&] {
			if (pointSizeMember >= 0)
			{
				uint32 ref = Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassUniform, tF), { ufBlockVar, ConstI(pointSizeMember) });
				Store(Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassOutput, tF), { perVertexVar, ConstI(1) }), Load(tF, ref));
			}
			OpVoid(SpvOpEmitVertex, {});
			// sbBase<n> += the stride, a vertex on
			if (ctx->analyzer.useSSBOForStreamout)
			{
				for (uint32 i = 0; i < LATTE_NUM_STREAMOUT_BUFFER; i++)
				{
					if (!ctx->output->streamoutBufferWriteMask[i] || !sbBaseVar[i])
						continue;
					const sint32 stride = (sint32)(ctx->output->streamoutBufferStride[i] / 4);
					Store(sbBaseVar[i], Op(SpvOpIAdd, tI, { Load(tI, sbBaseVar[i]), ConstI(stride) }));
				}
			}
		};
		if (hasActiveMask)
			If(MaskIsTrue(true, cf.activeStackDepth + 1), emit);
		else
			emit();
		return Ok();
	}

	// _emitCFRingWriteCode of a vertex shader: v2g.passV2GParameter<n>.<mask>
	// = ivecN(the register's masked channels), for each of the burst
	bool SpvShader::EmitRingWrite(LatteDecompilerCFInstruction& cf)
	{
		if (isGeometry)
			return EmitGeometryRingWrite(cf);
		if (!isVertex || !usesGS)
			return Decline("ring write");
		if (cf.memWriteElemSize != 3 || (cf.exportArrayBase & 3) != 0)
			return Decline("ring write layout");
		uint8 sel[4];
		uint32 count = 0;
		for (uint32 k = 0; k < 4; k++)
		{
			if (cf.memWriteCompMask & (1u << k))
				sel[count++] = (uint8)k;
		}
		if (count == 0)
			return Decline("ring write of no channel");
		const uint32 tV4Ip = spv_type_pointer(w, SpvStorageClassOutput, tV4I);
		for (uint32 burst = 0; burst < cf.exportBurstCount + 1; burst++)
		{
			const uint32 index = cf.exportArrayBase / 4 + burst;
			if (index >= v2gCount || !v2gVar)
				return Decline("ring parameter not declared");
			uint32 c[4];
			for (uint32 i = 0; i < count; i++)
				c[i] = Convert({ ReadChannel(GPR(cf.exportSourceGPR + burst), sel[i]), dt }, DT_I).id;
			uint32 member = Op(SpvOpAccessChain, tV4Ip, { v2gVar, ConstI((sint32)index) });
			if (count == 4)
			{
				Store(member, Construct(tV4I, c, 4));
				continue;
			}
			// the masked channels into the vector, the rest kept
			uint32 part = count == 1 ? c[0] : Construct(spv_type_vector(w, tI, count), c, count);
			uint32 reg = Load(tV4I, member);
			for (uint32 i = 0; i < count; i++)
				reg = Op(SpvOpCompositeInsert, tV4I, { count == 1 ? part : Op(SpvOpCompositeExtract, tI, { part, i }), reg, sel[i] });
			Store(member, reg);
		}
		return Ok();
	}

	// LatteDecompiler_emitClauseCode's ALU clause: in an if on the active
	// mask, with the push before it and the pop or else after it
	bool SpvShader::EmitControlledALUClause(LatteDecompilerCFInstruction& cf)
	{
		const sint32 d = cf.activeStackDepth;
		const bool push = cf.type == GPU7_CF_INST_ALU_PUSH_BEFORE;
		const bool isBreak = cf.type == GPU7_CF_INST_ALU_BREAK;
		auto clause = [&] {
			if (push)
			{
				MaskStore(false, d, Load(tBool, MaskPtr(false, d - 1)));
				MaskStore(true, d + 1, Load(tBool, MaskPtr(true, d)));
			}
			if (isBreak)
			{
				aluBreakVar = Local(tBool, 0);
				Store(aluBreakVar, spv_const_bool(w, 0));
			}
			if (!EmitALUClause(cf))
				return;
			if (isBreak)
				If(Load(tBool, aluBreakVar), [&] { Break(); });
		};
		uint32 cond = MaskIsTrue(true, push ? d : d + 1);
		if (push)
		{
			IfElse(cond, clause, [&] {
				MaskStore(false, d, spv_const_bool(w, 0));
				MaskStore(true, d + 1, spv_const_bool(w, 0));
			}, true);
		}
		else
			If(cond, clause);
		if (cf.type == GPU7_CF_INST_ALU_POP_AFTER)
			Pop(d, 1);
		else if (cf.type == GPU7_CF_INST_ALU_POP2_AFTER)
			Pop(d, 2);
		else if (cf.type == GPU7_CF_INST_ALU_ELSE_AFTER)
			Else(d);
		return Ok();
	}

	// { bool loopActive = C; [int loopCounter = 0;] while (C == true [&& loopCounter < 500]) { [loopCounter++;]
	bool SpvShader::LoopStart(sint32 d)
	{
		if (loopDepth == sizeof(loops) / sizeof(loops[0]))
			return Decline("loops nested too deep");
		Loop& l = loops[loopDepth];
		const bool limited = ActiveSettings::ShaderPreventInfiniteLoopsEnabled();
		l.active = Local(tBool, 0);
		Store(l.active, Load(tBool, MaskPtr(true, d + 1)));
		l.counter = 0;
		if (limited)
		{
			l.counter = Local(tI, 0);
			Store(l.counter, ConstI(0));
		}
		l.header = spv_id(w);
		l.check = spv_id(w);
		l.body = spv_id(w);
		l.cont = spv_id(w);
		l.merge = spv_id(w);
		Branch(l.header);
		Label(l.header);
		OpVoid(SpvOpLoopMerge, { l.merge, l.cont, SpvLoopControlMaskNone });
		Branch(l.check);
		Label(l.check);
		uint32 cond = MaskIsTrue(true, d + 1);
		if (limited)
		{
			uint32 below = Op(SpvOpSLessThan, tBool, { Load(tI, l.counter), ConstI(500) });
			cond = Op(SpvOpLogicalAnd, tBool, { cond, below });
		}
		OpVoid(SpvOpBranchConditional, { cond, l.body, l.merge });
		curBlock = 0;
		Label(l.body);
		if (limited)
			Store(l.counter, Op(SpvOpIAdd, tI, { Load(tI, l.counter), ConstI(1) }));
		loopDepth++;
		return Ok();
	}

	// } C = loopActive; }
	bool SpvShader::LoopEnd(sint32 d)
	{
		if (loopDepth == 0)
			return Decline("loop end outside a loop");
		Loop& l = loops[--loopDepth];
		if (curBlock)
			Branch(l.cont);
		Label(l.cont);
		Branch(l.header);
		Label(l.merge);
		MaskStore(true, d + 1, Load(tBool, l.active));
		return Ok();
	}

	bool SpvShader::EmitClause(LatteDecompilerCFInstruction& cf)
	{
		const sint32 d = cf.activeStackDepth;
		switch (cf.type)
		{
		case GPU7_CF_INST_ALU:
			if (!hasActiveMask)
				return EmitALUClause(cf);
			[[fallthrough]];
		case GPU7_CF_INST_ALU_PUSH_BEFORE:
		case GPU7_CF_INST_ALU_POP_AFTER:
		case GPU7_CF_INST_ALU_POP2_AFTER:
		case GPU7_CF_INST_ALU_BREAK:
		case GPU7_CF_INST_ALU_ELSE_AFTER:
			if (!hasActiveMask)
				return Decline("control flow without an active mask");
			return EmitControlledALUClause(cf);
		case GPU7_CF_INST_EXPORT:
		case GPU7_CF_INST_EXPORT_DONE:
			return isVertex ? EmitVertexExport(cf) : EmitExport(cf);
		case GPU7_CF_INST_MEM_RING_WRITE:
			return EmitRingWrite(cf);
		case GPU7_CF_INST_EMIT_VERTEX:
			return EmitEmitVertex(cf);
		case GPU7_CF_INST_MEM_STREAM0_WRITE:
		case GPU7_CF_INST_MEM_STREAM1_WRITE:
			return EmitStreamWrite(cf);
		case GPU7_CF_INST_TEX:
			if (hasActiveMask)
			{
				If(MaskIsTrue(true, d + 1), [&] { EmitTEXClause(cf); });
				return Ok();
			}
			return EmitTEXClause(cf);
		case GPU7_CF_INST_ELSE:
			Else(d);
			return Ok();
		case GPU7_CF_INST_POP:
			Pop(d, (sint32)cf.popCount);
			return Ok();
		case GPU7_CF_INST_LOOP_START_DX10:
		case GPU7_CF_INST_LOOP_START_NO_AL:
			return LoopStart(d);
		case GPU7_CF_INST_LOOP_END:
			return LoopEnd(d);
		case GPU7_CF_INST_LOOP_BREAK:
			If(MaskIsTrue(true, d + 1), [&] { Break(); });
			return Ok();
		default:
			return Decline("control flow instruction");
		}
	}

	// the writer each decompiling thread keeps
	struct ThreadWriter
	{
		spv_writer_t w;
		ThreadWriter() { spv_writer_init(&w); }
		~ThreadWriter() { spv_writer_free(&w); }
	};
}

static thread_local const char* s_lastDecline;

const char* LatteDecompiler_SPIRVDeclineReason()
{
	return s_lastDecline;
}

bool LatteDecompiler_emitSPIRVShader(LatteDecompilerShaderContext* shaderContext, LatteDecompilerShader* shader)
{
	thread_local ThreadWriter s_writer;
	spv_writer_t* w = &s_writer.w;
	spv_writer_reset(w);
	SpvShader s{};
	s.w = w;
	s.ctx = shaderContext;
	s.shader = shader;
	s_lastDecline = nullptr;
	auto decline = [&](const char* why) {
		s_lastDecline = why;
		return false;
	};
	// what this emitter covers so far
	if (shader->shaderType != LatteConst::ShaderType::Pixel && shader->shaderType != LatteConst::ShaderType::Vertex &&
		shader->shaderType != LatteConst::ShaderType::Geometry)
		return decline("shader type");
	s.isVertex = shader->shaderType == LatteConst::ShaderType::Vertex;
	s.isGeometry = shader->shaderType == LatteConst::ShaderType::Geometry;
	s.usesGS = shaderContext->options->usesGeometryShader;
#ifdef __ANDROID__
	// the unused output locations the GLSL emitter writes there
	if (s.isVertex)
		return decline("vertex shader on Android");
#endif
	if (!shaderContext->list_subroutines.empty())
		return decline("subroutines");
	if (shader->hasError)
		return decline("shader error");
	s.dt = shaderContext->typeTracker.defaultDataType;
	if (s.dt != DT_F && s.dt != DT_I)
		return decline("register type");
	s.Setup();
	// the uniform block, and the offsets the renderer writes it with
	LatteDecompilerOutputUniformOffsets offsetsVK, offsetsGL;
	if (!s.DeclareUniforms(offsetsVK, RendererAPI::Vulkan, true) || !s.DeclareUniforms(offsetsGL, RendererAPI::OpenGL, false))
		return decline(s.decline);
	if (!(s.isVertex ? s.DeclareVertexIO() : s.isGeometry ? s.DeclareGeometryIO() : s.DeclareIO()) || !s.DeclareTextures() ||
		!s.DeclareStreamout() || !s.DeclareUniformBuffers())
		return decline(s.decline);
	// main
	uint32 fnType = spv_type_function(w, s.tVoid, nullptr, 0);
	uint32 mainId = spv_id(w);
	uint32 fnWords[4] = { s.tVoid, mainId, SpvFunctionControlMaskNone, fnType };
	spv_inst(w, SPV_SEC_FUNCTION, SpvOpFunction, fnWords, 4);
	uint32 label = spv_id(w);
	spv_inst(w, SPV_SEC_FUNCTION, SpvOpLabel, &label, 1);
	s.curBlock = label;
	// the GLSL emitter's locals, as it declares and initializes them
	uint32 zero = s.dt == DT_F ? s.ConstF(0.0f) : s.ConstI(0);
	uint32 scalar = s.TypeOf(s.dt);
	if (shaderContext->typeTracker.useArrayGPRs)
	{
		// ivec4 Ri[128]; then Ri[i] = ivec4(0); for each
		s.gprArrayVar = s.Local(spv_type_array(w, s.Vec4Of(s.dt), s.ConstI(LATTE_NUM_GPR)), 0);
		for (sint32 i = 0; i < LATTE_NUM_GPR; i++)
			s.Store(s.GPR((uint32)i), spv_const_null(w, s.Vec4Of(s.dt)));
	}
	else
	{
		for (sint32 i = 0; i < LATTE_NUM_GPR; i++)
		{
			if (shaderContext->analyzer.usesRelativeGPRRead || (shaderContext->analyzer.gprUseMask[i / 8] & (1 << (i & 7))) != 0)
				s.gprVar[i] = s.Local(s.Vec4Of(s.dt), spv_const_null(w, s.Vec4Of(s.dt)));
		}
	}
	if (s.isVertex)
		s.decVar = s.Local(s.tV4U, 0); // uvec4 attrDecoder;
	for (int b = 0; b < 5; b++)
		s.backupVar[b] = s.Local(scalar, 0);
	for (int p = 0; p < 2; p++)
	{
		for (int u = 0; u < 4; u++)
			s.pvVar[p][u] = s.Local(scalar, zero);
	}
	for (int p = 0; p < 2; p++)
		s.psVar[p] = s.Local(scalar, zero);
	s.tempVar = s.Local(s.Vec4Of(s.dt), spv_const_null(w, s.Vec4Of(s.dt)));
	s.tempResultFVar = s.Local(s.tF, 0);
	s.tempResultI = s.Local(s.tI, 0); // int tempResulti;
	s.ariVar = s.Local(s.tV4I, spv_const_null(w, s.tV4I)); // ivec4 ARi = ivec4(0);
	// bool predResult = true; and the active mask stacks
	s.predVar = s.Local(s.tBool, 0);
	s.Store(s.predVar, spv_const_bool(w, 1));
	// vec3 cubeMapSTM; int cubeMapFaceId; float cubeMapArrayIndex<n> = 0.0;
	if (shaderContext->analyzer.hasRedcCUBE)
	{
		s.cubeSTMVar = s.Local(spv_type_vector(w, s.tF, 3), 0);
		s.cubeFaceVar = s.Local(s.tI, 0);
	}
	for (sint32 i = 0; i < LATTE_NUM_MAX_TEX_UNITS; i++)
	{
		if (s.texVar[i] && shader->textureUnitDim[i] == Latte::E_DIM::DIM_CUBEMAP)
			s.cubeIndexVar[i] = s.Local(s.tF, s.ConstF(0.0f));
	}
	if (shaderContext->analyzer.modifiesPixelActiveState)
	{
		const sint32 depth = shaderContext->analyzer.activeStackMaxDepth;
		s.hasActiveMask = true;
		s.maskLen = depth + 1;
		s.maskCLen = depth + 2;
		s.maskVar = s.Local(spv_type_array(w, s.tBool, s.ConstI(s.maskLen)), 0);
		s.maskCVar = s.Local(spv_type_array(w, s.tBool, s.ConstI(s.maskCLen)), 0);
		for (sint32 i = 0; i < depth; i++)
			s.MaskStore(false, i, spv_const_bool(w, 0));
		for (sint32 i = 0; i < depth + 1; i++)
			s.MaskStore(true, i, spv_const_bool(w, 0));
		s.MaskStore(false, 0, spv_const_bool(w, 1));
		s.MaskStore(true, 0, spv_const_bool(w, 1));
		s.MaskStore(true, 1, spv_const_bool(w, 1));
	}
	// the streamout buffers' bases; the inputs (a geometry shader reads the
	// ring where it fetches from it)
	if (!s.EmitStreamoutBases() || !(s.isVertex ? s.EmitVertexInputs() : s.isGeometry ? true : s.EmitInputs()))
		return decline(s.decline);
	for (auto& cf : shaderContext->cfInstructions)
	{
		if (!s.EmitClause(cf))
			return decline(s.decline ? s.decline : "out of memory");
	}
	if (s.loopDepth != 0)
		return decline("loop without an end");
	if (s.curBlock && s.isGeometry)
		s.OpVoid(SpvOpEndPrimitive, {}); // EndPrimitive(); at the end
	if (s.curBlock && s.isVertex && s.pointSizeMember >= 0)
	{
		// gl_PointSize = uf_pointSize; at the end
		uint32 ref = s.Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassUniform, s.tF), { s.ufBlockVar, s.ConstI(s.pointSizeMember) });
		s.Store(s.Op(SpvOpAccessChain, spv_type_pointer(w, SpvStorageClassOutput, s.tF), { s.perVertexVar, s.ConstI(1) }), s.Load(s.tF, ref));
	}
	if (s.curBlock)
		s.OpVoid(SpvOpReturn, {});
	s.OpVoid(SpvOpFunctionEnd, {});
	// entry point and modes
	uint32 epWords[2] = { s.isVertex ? (uint32)SpvExecutionModelVertex : s.isGeometry ? (uint32)SpvExecutionModelGeometry :
		(uint32)SpvExecutionModelFragment, mainId };
	spv_inst_str(w, SPV_SEC_ENTRY_POINT, SpvOpEntryPoint, epWords, 2, "main", s.interfaceIds, s.interfaceCount);
	if (s.isGeometry)
	{
		// layout(<input>) in; layout(<output>, max_vertices=<n>) out; as rslc
		// writes them, with its Invocations 1
		uint32 cap = SpvCapabilityGeometry;
		spv_inst(w, SPV_SEC_CAPABILITY, SpvOpCapability, &cap, 1);
		const uint32 inMode = s.gsInputVertices == 1 ? SpvExecutionModeInputPoints : s.gsInputVertices == 3 ?
			SpvExecutionModeTriangles : SpvExecutionModeInputLinesAdjacency;
		const uint32 gsOutPrimType = shaderContext->contextRegisters[mmVGT_GS_OUT_PRIM_TYPE];
		const uint32 outMode = gsOutPrimType == 0 ? SpvExecutionModeOutputPoints : gsOutPrimType == 1 ?
			SpvExecutionModeOutputLineStrip : SpvExecutionModeOutputTriangleStrip;
		const uint32 bytesPerVertex = shaderContext->contextRegisters[mmSQ_GS_VERT_ITEMSIZE] * 4;
		if (bytesPerVertex == 0)
			return decline("geometry vertex size");
		uint32 maxVertices = ((shaderContext->contextRegisters[mmSQ_GSVS_RING_ITEMSIZE] & 0x7FFF) * 4) / bytesPerVertex;
		if (!shaderContext->analyzer.hasLoops && shaderContext->analyzer.numEmitVertex < maxVertices)
			maxVertices = shaderContext->analyzer.numEmitVertex;
		const uint32 modes[4][3] = { { mainId, inMode, 0 }, { mainId, SpvExecutionModeInvocations, 1 }, { mainId, outMode, 0 },
			{ mainId, SpvExecutionModeOutputVertices, maxVertices } };
		for (uint32 i = 0; i < 4; i++)
			spv_inst(w, SPV_SEC_EXEC_MODE, SpvOpExecutionMode, modes[i], i == 1 || i == 3 ? 3 : 2);
	}
	else if (!s.isVertex)
	{
		uint32 mode[2] = { mainId, SpvExecutionModeOriginUpperLeft };
		spv_inst(w, SPV_SEC_EXEC_MODE, SpvOpExecutionMode, mode, 2);
	}
	if (s.sampledCubeArray)
	{
		uint32 cap = SpvCapabilitySampledCubeArray;
		spv_inst(w, SPV_SEC_CAPABILITY, SpvOpCapability, &cap, 1);
	}
	if (s.sampled1D)
	{
		uint32 cap = SpvCapabilitySampled1D;
		spv_inst(w, SPV_SEC_CAPABILITY, SpvOpCapability, &cap, 1);
	}
	if (s.imageQuery)
	{
		uint32 cap = SpvCapabilityImageQuery;
		spv_inst(w, SPV_SEC_CAPABILITY, SpvOpCapability, &cap, 1);
	}
	if (shaderContext->options->spirvInstrinsics.hasRoundingModeRTEFloat32)
	{
		uint32 cap = SpvCapabilityRoundingModeRTE;
		spv_inst(w, SPV_SEC_CAPABILITY, SpvOpCapability, &cap, 1);
		spv_inst_str(w, SPV_SEC_EXTENSION, SpvOpExtension, nullptr, 0, "SPV_KHR_float_controls", nullptr, 0);
		uint32 rte[3] = { mainId, SpvExecutionModeRoundingModeRTE, 32 };
		spv_inst(w, SPV_SEC_EXEC_MODE, SpvOpExecutionMode, rte, 3);
	}
	if (!s.Ok())
		return decline(s.decline ? s.decline : "out of memory");
	size_t count = 0;
	const uint32_t* words = spv_writer_finish(w, &count);
	if (!words)
		return decline("out of memory");
	shader->spirv.assign(words, words + count);
	shaderContext->output->uniformOffsetsVK = offsetsVK;
	shaderContext->output->uniformOffsetsGL = offsetsGL;
	return true;
}
