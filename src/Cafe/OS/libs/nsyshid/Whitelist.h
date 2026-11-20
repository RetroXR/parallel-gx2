#ifndef CEMU_NSYSHID_WHITELIST_H
#define CEMU_NSYSHID_WHITELIST_H

#include <list>
#include <tuple>
#include <boost/container/static_vector.hpp>

namespace nsyshid
{
	class Whitelist {
	  public:
		static Whitelist& GetInstance();

		Whitelist(const Whitelist&) = delete;

		Whitelist& operator=(const Whitelist&) = delete;

		bool IsDeviceWhitelisted(uint16 vendorId, uint16 productId);

		void AddDevice(uint16 vendorId, uint16 productId);

		void RemoveDevice(uint16 vendorId, uint16 productId);

		std::list<std::tuple<uint16, uint16>> GetDevices();

		void RemoveAllDevices();

	  private:
		Whitelist();

		// vendorId, productId: a few known devices and those added, in place
		boost::container::static_vector<std::tuple<uint16, uint16>, 32> m_devices;
	};
} // namespace nsyshid

#endif // CEMU_NSYSHID_WHITELIST_H
