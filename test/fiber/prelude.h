// What the core's precompiled header gives Fiber*.cpp, and no more
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
using uint64 = uint64_t;
enum class LogType { Force };
template<typename... Args>
inline void cemuLog_log(LogType, const char* fmt, Args&&...) { std::fprintf(stderr, "%s\n", fmt); }
inline void cemuLog_waitForFlush() {}
#define cemu_assert_unimplemented() std::abort()
#define cemu_assert_debug(x) do { if (!(x)) std::abort(); } while (0)
