#pragma once

#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.h"

// Vertex and pixel shaders decompiled ahead of the draw that first uses
// them. As the GPU thread starts a command buffer from the ring, a compile
// thread reads it ahead (LattePM4Scan.h) from a copy of the registers as
// they are then, and at each draw whose shaders the GPU thread has never
// seen, it decompiles them from the copy as they are at that draw - a
// vertex shader with its fetch shader parsed there. The draw, when the GPU
// thread gets to it, takes a result only where the shader's base and aux
// hashes and every register its decompile reads are what its own
// decompile would have had (LattePM4_PixelShaderInputs,
// LattePM4_VertexShaderInputs), and for a vertex shader its own fetch
// shader parsed the same (LatteFetchShader::Fingerprint); otherwise, or
// with the compile thread not done, it decompiles the shader itself, as
// before. Vulkan, whose compile threads it runs on; geometry
// shaders' draws are left to the draw. Nothing allocated while a game plays:
// the copies and slots are made at the start (LatteShaderAhead_Init).

// GPU thread, at the renderer's start and end (the end waits for a read
// under way and frees what no draw took)
void LatteShaderAhead_Init();
void LatteShaderAhead_Shutdown();

// GPU thread: a command buffer the ring calls, about to be processed
void LatteShaderAhead_OnCommandBuffer(uint32 physAddr, uint32 words);

// GPU thread: the pixel shader for baseHash decompiled ahead, if it was, for
// the registers as they are now; false to decompile it here
bool LatteShaderAhead_TakePixelShader(uint64 baseHash, LatteDecompilerOutput_t& output);

// GPU thread: the vertex shader for baseHash decompiled ahead, if it was, for
// the registers as they are now and the draw's fetch shader, which becomes
// its compatible one; false to decompile it here
bool LatteShaderAhead_TakeVertexShader(uint64 baseHash, struct LatteFetchShader* fetchShader, LatteDecompilerOutput_t& output);

// GPU thread: a shader now known, never decompiled ahead again
void LatteShaderAhead_NotePixelShader(uint64 baseHash);
void LatteShaderAhead_NoteVertexShader(uint64 baseHash);
