#pragma once

#if BOOST_OS_WINDOWS

#endif

class Fiber
{
public:
	Fiber(void(*FiberEntryPoint)(void* userParam), void* userParam, void* privateData);
	~Fiber();

	static Fiber* PrepareCurrentThread(void* privateData = nullptr);
	static void Switch(Fiber& targetFiber);
	static void* GetFiberPrivateData();
	// the same, or nullptr on a thread no fiber was prepared on
	static void* TryGetFiberPrivateData();

	// Labels the fiber for crash reports. On Android it names the fiber in the
	// report of a switch to one that is already running; the other backends
	// ignore it.
	void SetDebugName(const char* name);

	// Starts the fiber over at FiberEntryPoint, its stack kept where the
	// backend can (a fiber made with an entry point, not running, and not to
	// be returned to where it stopped)
	void Restart(void(*FiberEntryPoint)(void* userParam), void* userParam);
private:
	Fiber(void* privateData); // fiber from current thread

	void* m_implData{nullptr};
	void* m_privateData;
	void* m_stackPtr{ nullptr };
};