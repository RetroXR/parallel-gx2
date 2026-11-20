#pragma once
#include <set>
#include <algorithm>
#include <boost/container/static_vector.hpp>
#include <vector>
#include "Cafe/OS/RPL/COSModule.h"

#if BOOST_OS_WINDOWS
#include <winsock2.h>
#else
#include <sys/socket.h>
#define SOCKET int
#define closesocket close
#endif

typedef signed int WUSOCKET;

WUSOCKET nsysnet_createVirtualSocketFromExistingSocket(SOCKET existingSocket);
void nsysnet_notifyCloseSharedSocket(SOCKET existingSocket);

struct wu_fd_set
{
	uint32be mask;
};

void _translateFDSet(fd_set* hostSet, struct wu_fd_set* fdset, sint32 nfds);
void _translateFDSetRev(struct wu_fd_set* fdset, fd_set* hostSet, sint32 nfds);

sint32 nsysnet_getVirtualSocketHandleFromHostHandle(SOCKET s);

namespace nsysnet
{

#define NSSL_OK (0)
#define NSSL_INVALID_CTX (0xFFD7FFFF)

	struct NSSLInternalState_t
	{
		bool destroyed;
		uint32 sslVersion;
		uint32 clientPKI;
		// in order, each once (as the set it was): both groups' (39) and a
		// game's own fit, a context's PKIs set without allocating
		boost::container::static_vector<uint32, 64> serverPKIs;
		void AddServerPKI(uint32 pki)
		{
			auto it = std::lower_bound(serverPKIs.begin(), serverPKIs.end(), pki);
			if ((it != serverPKIs.end() && *it == pki) || serverPKIs.size() == serverPKIs.capacity())
				return;
			serverPKIs.insert(it, pki);
		}
		std::vector<std::vector<uint8>> serverCustomPKIs;
	};

	NSSLInternalState_t* GetNSSLContext(sint32 index);

	void wuResetFD(struct wu_fd_set* fdset);
	void wuSetFD(struct wu_fd_set* fdset, sint32 fd);

	COSModule* GetModule();
}