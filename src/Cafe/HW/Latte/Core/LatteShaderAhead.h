#pragma once

#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.h"

// Shaders decompiled ahead of the draw that first uses them. As the GPU
// thread starts a command buffer from the ring, a compile thread reads it
// ahead (LattePM4Scan.h) from a copy of the registers as they are then -
// and on into the buffers the guest has submitted to the ring since, the
// registers carried from one to the next, where the GPU thread is behind;
// a buffer read so is not read again when the GPU thread starts it, unless
// the read stopped before it. At each draw whose shaders the GPU thread
// has never seen, it hands each to a compile thread of its own with a copy
// of the registers its decompile reads (or decompiles it there, with none
// free), so that a read's new shaders are decompiled side by side - a
// vertex shader with its fetch shader parsed there. The draw, when the GPU
// thread gets to it, takes a result only where the shader's base and aux
// hashes and every register its decompile reads are what its own
// decompile would have had (LattePM4_PixelShaderInputs,
// LattePM4_VertexShaderInputs, LattePM4_GeometryShaderInputs), and for a
// vertex shader its own fetch shader parsed the same
// (LatteFetchShader::Fingerprint); otherwise, or with the compile thread
// not done, it decompiles the shader itself, as before. A geometry shader's
// draw has its three decompiled where its vertex (export) shader is, whose
// ring the geometry shader's key and decompile take. Vulkan, whose compile
// threads it runs on. Nothing allocated while a game plays: the copies and
// slots are made at the start (LatteShaderAhead_Init).

// GPU thread, at the renderer's start and end (the end waits for a read
// under way and frees what no draw took)
void LatteShaderAhead_Init();
void LatteShaderAhead_Shutdown();

// GPU thread, before each draw's shaders are looked up: the parts of the
// shaders made ahead since queued for the compile threads, for every
// template seen so far (PipelineCompiler::PrepareAheadPartsFor), so that
// their first pipeline links. One atomic load where there are none.
void LatteShaderAhead_Poll();

// GPU thread: a command buffer the ring calls, about to be processed
void LatteShaderAhead_OnCommandBuffer(uint32 physAddr, uint32 words);

// GPU thread: gone idle with nothing in the ring (ringAt: its read index),
// its registers then offered for a read of the buffer the guest writes
// next; or with work again, the offer withdrawn
void LatteShaderAhead_OnIdle(bool idle, uint32 ringAt);

// The guest's thread: words of the command buffer it is writing, not yet
// submitted (ringAt: the ring's write index). Read ahead from the idle GPU
// thread's registers where they are offered and it has submitted nothing
// since - where a GPU thread that is never behind gives the reads at the
// start of a buffer no lead - and no read of the guest's own is under way.
void LatteShaderAhead_OnGuestCommands(uint32 physAddr, uint32 words, uint32 ringAt);

// GPU thread: the pixel shader for baseHash decompiled ahead, if it was, for
// the registers as they are now; false to decompile it here
bool LatteShaderAhead_TakePixelShader(uint64 baseHash, LatteDecompilerOutput_t& output);

// GPU thread: the vertex shader for baseHash decompiled ahead, if it was, for
// the registers as they are now and the draw's fetch shader, which becomes
// its compatible one; false to decompile it here
bool LatteShaderAhead_TakeVertexShader(uint64 baseHash, struct LatteFetchShader* fetchShader, LatteDecompilerOutput_t& output);

// GPU thread: the geometry shader for baseHash decompiled ahead, if it was,
// for the registers as they are now, the ring its vertex shader writes and
// its copy shader
bool LatteShaderAhead_TakeGeometryShader(uint64 baseHash, uint32 vsRingParameterCount, const uint8* copyProgram, uint32 copyProgramSize, LatteDecompilerOutput_t& output);

// GPU thread: a shader now known, never decompiled ahead again
void LatteShaderAhead_NotePixelShader(uint64 baseHash);
void LatteShaderAhead_NoteVertexShader(uint64 baseHash);
void LatteShaderAhead_NoteGeometryShader(uint64 baseHash);
// GPU thread: a pipeline it has (VulkanRenderer::draw_calculateGraphicsPipelineHash),
// made or loaded from the cache: never made ahead
void LatteShaderAhead_NotePipeline(uint64 pipelineHash);

// GPU thread: whether a read queued the pipeline ahead of its draw
bool LatteShaderAhead_PipelineQueuedAhead(uint64 pipelineHash);

// GPU thread: what new pipelines' draws met, logged and reset as the title
// stops (LatteShaderAhead_Shutdown)
struct LatteDrawPathCounts& LatteShaderAhead_DrawPathCounts();
