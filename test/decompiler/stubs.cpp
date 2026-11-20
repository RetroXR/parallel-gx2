// What the shader decompiler takes from the rest of the core: the renderer
// it decompiles for (Vulkan), the GPU state it reads beside the context
// registers it is given, the pixel shader's input table and the settings
// that change what it writes - each as the test sets it.
#include "Cemu/Logging/CemuLogging.h"
#include "Cafe/HW/Latte/Core/Latte.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/Core/LatteCachedFBO.h"
#include "Cafe/HW/Latte/Core/LattePerformanceMonitor.h"
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.h"
#include "config/ActiveSettings.h"

#include <chrono>

uint64 s_loggingFlagMask = 0;
bool cemuLog_log(LogType type, std::string_view text)
{
	return false;
}

LatteGPUState_t LatteGPUState = {};
performanceMonitor_t performanceMonitor;

LatteShaderPSInputTable g_testPSInputTable;
LatteShaderPSInputTable* LatteSHRC_GetPSInputTable()
{
	return &g_testPSInputTable;
}

bool ActiveSettings::DumpShadersEnabled()
{
	return false;
}
// the vertex shaders' fetch shaders are the test's own, never registered
#include "Cafe/HW/Latte/Core/FetchShader.h"
LatteFetchShader::~LatteFetchShader()
{
}

// the test turns it on and off: loops are written either way
bool g_testPreventInfiniteLoops = false;
bool ActiveSettings::ShaderPreventInfiniteLoopsEnabled()
{
	return g_testPreventInfiniteLoops;
}
// the titles sampling on texel edges' nudge: the test turns it on and off
bool g_testRoundToPrecision = false;
bool ActiveSettings::ForceSamplerRoundToPrecision()
{
	return g_testRoundToPrecision;
}

// one color buffer, 32 bit float per channel (the test reads the bits back)
uint8 LatteMRT::GetActiveColorBufferMask(const LatteDecompilerShader* pixelShader, const LatteContextRegister& lcr)
{
	return 1;
}
bool LatteMRT::GetActiveDepthBufferMask(const LatteContextRegister& lcr)
{
	return false;
}
Latte::E_GX2SURFFMT LatteMRT::GetColorBufferFormat(const uint32 index, const LatteContextRegister& lcr)
{
	return Latte::E_GX2SURFFMT::R32_G32_B32_A32_FLOAT;
}

Latte::E_GX2SURFFMT LatteTexture_ReconstructGX2Format(const Latte::LATTE_SQ_TEX_RESOURCE_WORD1_N& texUnitWord1, const Latte::LATTE_SQ_TEX_RESOURCE_WORD4_N& texUnitWord4)
{
	return Latte::E_GX2SURFFMT::R8_G8_B8_A8_UNORM;
}

uint64 PPCTimer_getRawTsc()
{
	return (uint64)std::chrono::steady_clock::now().time_since_epoch().count();
}

#include "../test_renderer.h"
