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
// where the right side is no bare variable or constant.
// What it does not cover yet it declines
// (LatteDecompiler_emitSPIRVShader returns false) and the shader takes the
// GLSL path as before; LatteDecompiler_SPIRVDeclineReason says why.
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/Core/LatteShaderAssembly.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "Cafe/HW/Latte/Core/Latte.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
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
		uint32 pvVar[2][4]{}, psVar[2]{}, backupVar[5]{}, tempVar{}, tempResultF{}, tempResultI{}, ariVar{}, predVar{};
		uint32 ufBlockVar{};
		// the active mask stacks (activeMaskStack, activeMaskStackC) and
		// their lengths, when the shader has control flow
		bool hasActiveMask{};
		uint32 maskVar{}, maskCVar{};
		sint32 maskLen{}, maskCLen{};
		uint32 aluBreakVar{}; // the ALU_BREAK clause's aluBreak
		// the loops open: rslc's blocks for each, its loopActive and counter
		struct Loop
		{
			uint32 header, check, body, cont, merge, active, counter;
		};
		Loop loops[16];
		uint32 loopDepth{};
		uint32 fnMulNonIEEE{}, fnClampFI32{};
		uint32 tempResultFVar{};
		uint32 ufMember[8]{};
		uint32 inputVar[GPU7_PS_MAX_INPUTS]{};
		uint32 colorOut[LATTE_NUM_COLOR_TARGET]{};
		uint32 interfaceIds[64];
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

		// a function variable, initialized as the GLSL emitter's declaration
		uint32 Local(uint32 type, uint32 init)
		{
			uint32 ptr = spv_type_pointer(w, SpvStorageClassFunction, type);
			uint32 words[4] = { ptr, spv_id(w), SpvStorageClassFunction, init };
			spv_inst(w, SPV_SEC_LOCALS, SpvOpVariable, words, init ? 4 : 3);
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

		uint32 GPR(uint32 index)
		{
			if (!gprVar[index])
				gprVar[index] = Local(Vec4Of(dt), spv_const_null(w, Vec4Of(dt)));
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
			Label(aLabel);
			for (size_t i = 0; i < aWords;)
			{
				uint32 n = s_sides[i] >> 16;
				spv_inst(w, SPV_SEC_BODY, s_sides[i] & 0xFFFF, s_sides.data() + i + 1, n - 1);
				i += n;
			}
			OpVoid(SpvOpBranch, { mergeLabel });
			Label(bLabel);
			for (size_t i = aWords; i < s_sides.size();)
			{
				uint32 n = s_sides[i] >> 16;
				spv_inst(w, SPV_SEC_BODY, s_sides[i] & 0xFFFF, s_sides.data() + i + 1, n - 1);
				i += n;
			}
			OpVoid(SpvOpBranch, { mergeLabel });
			Label(mergeLabel);
			return Run(Op(SpvOpPhi, TypeOf(type), { a.id, aLabel, b.id, bLabel }), type);
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

		bool Setup();
		bool DeclareUniforms(LatteDecompilerOutputUniformOffsets& offsets, RendererAPI api, bool declare);
		bool DeclareIO();
		bool EmitInputs();
		bool EmitClause(LatteDecompilerCFInstruction& cf);
		bool EmitALUClause(LatteDecompilerCFInstruction& cf);
		bool EmitExport(LatteDecompilerCFInstruction& cf);
		Val Operand(LatteDecompilerALUInstruction& inst, sint32 index, sint32 requiredType);
		Val UniformOperand(LatteDecompilerALUInstruction& inst, sint32 index, sint32 requiredType);
		Val RegisterRead(LatteDecompilerALUInstruction& inst, sint32 index);
		Val PVPSRead(LatteDecompilerALUInstruction& inst, uint8 unit);
		uint32 OutputPtr(LatteDecompilerALUInstruction& inst);
		uint32 PVPSOutputPtr(LatteDecompilerALUInstruction& inst);
		void Assign(LatteDecompilerALUInstruction& inst, Val v);
		bool EmitOP2(LatteDecompilerCFInstruction& cf, LatteDecompilerALUInstruction& inst);
		bool EmitOP3(LatteDecompilerALUInstruction& inst);
		bool EmitReduction(LatteDecompilerALUInstruction* redc[4]);
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
		uint32 memberTypes[8];
		uint32 memberOffsets[8];
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
		else if (shader->uniformMode != LATTE_DECOMPILER_UNIFORM_MODE_NONE)
			return Decline("uniform buffers");
		// special uniforms (a pixel shader's)
		bool alphaTestEnable = dc->contextRegistersNew->SX_ALPHA_TEST_CONTROL.get_ALPHA_TEST_ENABLE();
		if (alphaTestEnable)
		{
			uniformCurrentOffset = (uniformCurrentOffset + 3) & ~3;
			memberTypes[memberCount] = tF;
			memberOffsets[memberCount++] = uniformCurrentOffset;
			uniformOffsets.offset_alphaTestRef = uniformCurrentOffset;
			uniformCurrentOffset += 4;
		}
		if (dc->analyzer.hasFragCoordAccess)
			return Decline("gl_FragCoord");
		for (sint32 t = 0; t < LATTE_NUM_MAX_TEX_UNITS; t++)
		{
			if (dc->analyzer.texUnitUsesTexelCoordinates.test(t))
				return Decline("texel coordinates");
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
			uint32 v = spv_global_variable(w, ptrIn, SpvStorageClassInput);
			Decorate(v, SpvDecorationLocation, { (uint32)i });
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
			uint32 v = Load(TypeOf(dt), Channel(GPR(temporaries[i].gpr), dt, temporaries[i].chan));
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
		return { Load(TypeOf(dt), Channel(GPR(gpr), dt, inst.sourceOperand[index].chan)), dt };
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
			return { Load(TypeOf(dt), Channel(GPR(pvps[unit].index), dt, pvps[unit].unit)), dt };
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
		if (op.rel != 0)
		{
			Decline("relative uniform access");
			return { ConstI(0), requiredType };
		}
		bool isRegister = GPU7_ALU_SRC_IS_CFILE(op.sel);
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
				else if (it.kcacheBankId == (uint32)bufferIndex && it.index == (uint32)uniformOffset)
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
		uint32 ptr = spv_type_pointer(w, SpvStorageClassUniform, tI);
		uint32 p = Op(SpvOpAccessChain, ptr, { ufBlockVar, ConstI(0), ConstI(arrayIndex), ConstI(op.chan) });
		return Convert({ Load(tI, p), DT_I }, requiredType);
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
			if (op.rel != 0)
			{
				Decline("relative GPR read");
				return KI(0);
			}
			Val r = RegisterRead(inst, index);
			if (requiredType == DT_U && r.type == DT_F)
			{
				Decline("float register read as uint");
				return KU(0);
			}
			v = Convert(r, requiredType);
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
		{
			Decline("relative GPR write");
			return PVPSOutputPtr(inst);
		}
		return Channel(GPR(inst.destGpr), dt, inst.destElem);
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
		if (inst.destRel != 0)
			return Decline("relative GPR write");
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

	bool SpvShader::EmitReduction(LatteDecompilerALUInstruction* redc[4])
	{
		if (redc[0]->opcode == ALU_OP2_INST_CUBE)
			return Decline("CUBE");
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
					return Decline("MOVA");
				if (!EmitOP2(cf, inst))
					return false;
			}
			// omod and clamp on the instruction's output - a reduction's first,
			// as in the GLSL emitter
			LatteDecompilerALUInstruction& last = inst;
			if (last.omod != ALU_OMOD_NONE)
			{
				uint32 ptr = OutputPtr(last);
				uint32 v = Load(TypeOf(dt), ptr);
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
				uint32 ptr = OutputPtr(last);
				uint32 v = Load(TypeOf(dt), ptr);
				uint32 r;
				if (dt == DT_F)
					r = Ext(GLSLstd450FClamp, tF, { v, ConstF(0.0f), ConstF(1.0f) });
				else
					r = Call(ClampFI32Function(), tI, DT_I, tI, { v }).id;
				Store(OutputPtr(last), r);
			}
			if (isReduction)
			{
				// DOT4: the result into the other three outputs and PVs, read
				// again for each as the GLSL emitter does
				for (int f = 0; f < 4; f++)
				{
					if (redc[f]->writeMask == 0)
						Store(PVPSOutputPtr(*redc[f]), Load(TypeOf(dt), OutputPtr(*redc[0])));
					else if (f != 0)
						Store(OutputPtr(*redc[f]), Load(TypeOf(dt), OutputPtr(*redc[0])));
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
			uint32 v = Load(TypeOf(dt), Channel(GPR(cf.exportSourceGPR + burst), dt, sel));
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
				uint32 refPtr = Op(SpvOpAccessChain, ptr, { ufBlockVar, ConstI(shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_NONE ? 0 : 1) });
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
			return EmitExport(cf);
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
	if (shader->shaderType != LatteConst::ShaderType::Pixel)
		return decline("not a pixel shader");
	if (shaderContext->options->usesGeometryShader)
		return decline("geometry shader");
	if (shaderContext->typeTracker.useArrayGPRs)
		return decline("GPR array");
	if (!shaderContext->list_subroutines.empty())
		return decline("subroutines");
	if (shaderContext->output->textureUnitMask.any())
		return decline("textures");
	if (shaderContext->analyzer.usesRelativeGPRRead || shaderContext->analyzer.usesRelativeGPRWrite)
		return decline("relative GPR access");
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
	if (!s.DeclareIO())
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
	for (int p = 0; p < 2; p++)
	{
		for (int u = 0; u < 4; u++)
			s.pvVar[p][u] = s.Local(scalar, zero);
		s.psVar[p] = s.Local(scalar, zero);
	}
	for (int b = 0; b < 5; b++)
		s.backupVar[b] = s.Local(scalar, 0);
	s.tempVar = s.Local(s.Vec4Of(s.dt), spv_const_null(w, s.Vec4Of(s.dt)));
	s.tempResultFVar = s.Local(s.tF, 0);
	// bool predResult = true; and the active mask stacks
	s.predVar = s.Local(s.tBool, 0);
	s.Store(s.predVar, spv_const_bool(w, 1));
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
	if (!s.EmitInputs())
		return decline(s.decline);
	for (auto& cf : shaderContext->cfInstructions)
	{
		if (!s.EmitClause(cf))
			return decline(s.decline ? s.decline : "out of memory");
	}
	if (s.loopDepth != 0)
		return decline("loop without an end");
	if (s.curBlock)
		s.OpVoid(SpvOpReturn, {});
	s.OpVoid(SpvOpFunctionEnd, {});
	// entry point and modes
	uint32 epWords[2] = { SpvExecutionModelFragment, mainId };
	spv_inst_str(w, SPV_SEC_ENTRY_POINT, SpvOpEntryPoint, epWords, 2, "main", s.interfaceIds, s.interfaceCount);
	uint32 mode[2] = { mainId, SpvExecutionModeOriginUpperLeft };
	spv_inst(w, SPV_SEC_EXEC_MODE, SpvOpExecutionMode, mode, 2);
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
