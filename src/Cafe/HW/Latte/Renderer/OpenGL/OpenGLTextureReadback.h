#pragma once
#include "Cafe/HW/Latte/Core/LatteTextureReadbackInfo.h"
#include "Common/GLInclude/GLInclude.h"
#include "util/helpers/PoolNew.h"

// from a pool (PoolNew.h), as Vulkan's
class LatteTextureReadbackInfoGL : public LatteTextureReadbackInfo, public PoolNew<LatteTextureReadbackInfoGL, 64>
{
public:
	LatteTextureReadbackInfoGL(LatteTextureView* textureView);

	~LatteTextureReadbackInfoGL();

	void StartTransfer() override;
	bool IsFinished() override;
	void ForceFinish() override;

	uint8* GetData() override;
	void ReleaseData() override;

private:
	bool WaitForCompletion(GLuint64 timeout);
	GLuint m_texFormatGL;
	GLuint m_texDataTypeGL;

	// buffer
	GLuint texImageBufferGL = 0;
	// sync
	GLsync imageCopyFinSync = nullptr;
};

