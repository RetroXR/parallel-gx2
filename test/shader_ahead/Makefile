# Pixel shaders decompiled ahead of their draw (src/Cafe/HW/Latte/Core/
# LatteShaderAhead.cpp): the command buffer read ahead (LattePM4Scan.cpp),
# the registers a pixel shader's decompile reads, and the slots and filter
# the reads and the GPU thread hand shaders over in (AheadSlots.h)
# (run_tests.sh). SANITIZER=thread or SANITIZER=address,undefined for the
# sanitizer runs; a MinGW compiler, or MSYS2's g++, for Windows.
LRC := ../../dependencies/libretro-common
SRC := ../../src
DEP := ../../dependencies
CXX ?= g++
FLAGS := -O1 -g -I$(SRC) -I$(LRC)/include -Wno-multichar -I$(DEP)/boost-compat -I$(DEP)/fmt/include \
	-include $(SRC)/Common/precompiled.h
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(or $(findstring mingw,$(CXX)),$(filter Windows_NT,$(OS))))
   LIBS += -static
   EXE := .exe
endif

shader_ahead_test$(EXE): shader_ahead_test.cpp $(SRC)/Cafe/HW/Latte/Core/LattePM4Scan.cpp $(SRC)/Cafe/HW/Latte/Core/LattePM4Scan.h $(SRC)/util/helpers/AheadSlots.h \
	$(SRC)/Cafe/HW/Latte/Core/LatteShaderKeys.h $(SRC)/Cafe/HW/Latte/Core/LatteDrawPathCounts.h $(SRC)/Cafe/HW/Latte/Core/LatteShaderAheadRules.h
	$(CXX) $(FLAGS) -std=gnu++20 shader_ahead_test.cpp $(SRC)/Cafe/HW/Latte/Core/LattePM4Scan.cpp $(LIBS) -o $@

clean:
	rm -f shader_ahead_test shader_ahead_test.exe
