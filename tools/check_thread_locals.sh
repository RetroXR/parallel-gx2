#!/bin/sh
# No thread_local in the core holds anything to free. MinGW (emutls) frees a
# thread_local's storage before it runs the destructor, so one owning a
# buffer, a decoder or a slot read its fields from freed memory as its
# thread ended: a free of a stale pointer, a hazard slot given back by a
# stale index. What a thread keeps for itself is reached by a thread_local
# pointer and freed as the thread ends by thread_end.h (ThreadOwned.h).
#
# A thread_local passes as a pointer, a scalar (bool, char, the integer and
# float types), an array of either, or a type the same file holds trivially
# destructible with static_assert(std::is_trivially_destructible_v<Type>).
cd "$(dirname "$0")/.." || exit 1

bad=$(find src -name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.mm' | sort | while read -r f; do
	grep -n 'thread_local' "$f" | while IFS= read -r line; do
		text=${line#*:}
		# comments
		case "$text" in
		*//*thread_local*|*/\**thread_local*) continue ;;
		esac
		trimmed=$(printf '%s\n' "$text" | sed 's/^[[:space:]]*//')
		case "$trimmed" in
		\**|//*) continue ;;
		esac
		# the type: what follows thread_local, up to the name
		decl=$(printf '%s\n' "$text" | sed -n 's/.*thread_local[[:space:]]\{1,\}\([^;={(]*\).*/\1/p')
		[ -n "$decl" ] || continue
		case "$decl" in
		*\**) continue ;; # a pointer
		esac
		type=$(printf '%s\n' "$decl" | sed 's/[[:space:]]*\[.*//; s/[[:space:]]*[A-Za-z_][A-Za-z_0-9]*[[:space:]]*$//; s/^const[[:space:]]\{1,\}//; s/[[:space:]]*$//')
		case "$type" in
		bool|char|int|unsigned|long|short|float|double|size_t|uint8|uint16|uint32|uint64|sint8|sint16|sint32|sint64|uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t|"unsigned int"|"unsigned char"|"unsigned long"|"long long"|"unsigned long long")
			continue ;;
		esac
		if grep -q "static_assert(std::is_trivially_destructible_v<$type>" "$f"; then
			continue
		fi
		echo "$f:${line%%:*}: thread_local $type"
	done
done)

if [ -n "$bad" ]; then
	echo "$bad"
	echo "a thread_local that holds something to free: MinGW runs its destructor on freed storage; use ThreadOwned.h (thread_end.h)"
	exit 1
fi
echo "every thread_local is a pointer, a scalar or trivially destructible"
