#pragma once
// A renderer for the tests whose code only asks which one it is (the
// decompiler, fetch shader keys): the one TU of a test that defines g_renderer
// includes this.
#include "Cafe/HW/Latte/Renderer/Renderer.h"

class TestRenderer : public Renderer
{
public:
	TestRenderer() : Renderer(RendererAPI::Vulkan) {}
	bool IsPadWindowActive() override { return false; }
	void ClearColorbuffer(bool) override {}
	void DrawEmptyFrame(bool) override {}
	void SwapBuffers(bool, bool) override {}
	void DrawBackbufferQuad(LatteTextureView*, RendererOutputShader*, bool, sint32, sint32, sint32, sint32, bool, bool) override {}
	bool BeginFrame(bool) override { return false; }
	void Flush(bool) override {}
	void NotifyLatteCommandProcessorIdle() override {}
	void renderTarget_setViewport(float, float, float, float, float, float, bool) override {}
	void renderTarget_setScissor(sint32, sint32, sint32, sint32) override {}
	LatteCachedFBO* rendertarget_createCachedFBO(uint64) override { return nullptr; }
	void rendertarget_deleteCachedFBO(LatteCachedFBO*) override {}
	void rendertarget_bindFramebufferObject(LatteCachedFBO*) override {}
	void* texture_acquireTextureUploadBuffer(uint32) override { return nullptr; }
	void texture_releaseTextureUploadBuffer(uint8*) override {}
	TextureDecoder* texture_chooseDecodedFormat(Latte::E_GX2SURFFMT, bool, Latte::E_DIM, uint32, uint32) override { return nullptr; }
	void texture_clearSlice(LatteTexture*, sint32, sint32) override {}
	void texture_loadSlice(LatteTexture*, sint32, sint32, sint32, void*, sint32, sint32, uint32) override {}
	void texture_clearColorSlice(LatteTexture*, sint32, sint32, float, float, float, float) override {}
	void texture_clearDepthSlice(LatteTexture*, uint32, sint32, bool, bool, float, uint32) override {}
	LatteTexture* texture_createTextureEx(Latte::E_DIM, MPTR, MPTR, Latte::E_GX2SURFFMT, uint32, uint32, uint32, uint32, uint32, uint32, Latte::E_HWTILEMODE, bool) override { return nullptr; }
	void texture_setLatteTexture(LatteTextureView*, uint32) override {}
	void texture_copyImageSubData(LatteTexture*, sint32, sint32, sint32, sint32, LatteTexture*, sint32, sint32, sint32, sint32, sint32, sint32, sint32) override {}
	LatteTextureReadbackInfo* texture_createReadback(LatteTextureView*) override { return nullptr; }
	void surfaceCopy_copySurfaceWithFormatConversion(LatteTexture*, sint32, sint32, LatteTexture*, sint32, sint32, sint32, sint32) override {}
	void bufferCache_init(const sint32) override {}
	void bufferCache_upload(uint8*, sint32, uint32) override {}
	void bufferCache_copy(uint32, uint32, uint32) override {}
	void bufferCache_copyStreamoutToMainBuffer(uint32, uint32, uint32) override {}
	void buffer_bindVertexBuffers(std::span<BindBufferParam>) override {}
	void buffer_bindUniformBuffer(LatteConst::ShaderType, uint32, uint32, uint32) override {}
	RendererShader* shader_create(RendererShader::ShaderType, uint64, uint64, std::string, bool, bool) override { return nullptr; }
	void streamout_setupXfbBuffer(uint32, sint32, uint32, uint32) override {}
	void streamout_begin() override {}
	void streamout_rendererFinishDrawcall() override {}
	void draw_beginSequence() override {}
	void draw_execute(uint32, uint32, uint32, uint32, MPTR, Latte::LATTE_VGT_DMA_INDEX_TYPE::E_INDEX_TYPE, const LatteDrawcallContext&) override {}
	void draw_endSequence() override {}
	IndexAllocation indexData_reserveIndexMemory(uint32) override { return {}; }
	void indexData_releaseIndexMemory(IndexAllocation&) override {}
	void indexData_uploadIndexMemory(IndexAllocation&) override {}
	LatteQueryObject* occlusionQuery_create() override { return nullptr; }
	void occlusionQuery_destroy(LatteQueryObject*) override {}
	void occlusionQuery_flush() override {}
	void occlusionQuery_updateState() override {}
};

void Renderer::Initialize() {}
void Renderer::Shutdown() {}
bool Renderer::GetVRAMInfo(int&, int&) const { return false; }

std::unique_ptr<Renderer> g_renderer = std::make_unique<TestRenderer>();
