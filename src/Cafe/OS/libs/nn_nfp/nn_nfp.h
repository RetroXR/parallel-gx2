#pragma once
#include "Cafe/OS/RPL/COSModule.h"

namespace nn::nfp
{
	uint32 NFCGetTagInfo(uint32 index, uint32 timeout, MPTR functionPtr, void* userParam);

	COSModule* GetModule();
}

void nnNfp_load();
void nnNfp_update();

bool nnNfp_isInitialized();
bool nnNfp_touchNfcTagFromFile(const fs::path& filePath, uint32* nfcError);
// libretro: lift the amiibo ahead of its 1.5 s touch, and whether one is on
void nnNfp_releaseTouch();
bool nnNfp_hasActiveAmiibo();

#define NFP_STATE_NONE			(0)
#define NFP_STATE_INIT			(1)
#define NFP_STATE_RW_SEARCH		(2)
#define NFP_STATE_RW_ACTIVE		(3)
#define NFP_STATE_RW_DEACTIVE	(4)
#define NFP_STATE_RW_MOUNT		(5)
#define NFP_STATE_UNEXPECTED	(6)
#define NFP_STATE_RW_MOUNT_ROM	(7)
