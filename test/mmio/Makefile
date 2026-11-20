# The MMIO handlers (src/Cafe/HW/MMU/MMU.cpp, run_tests.sh): registered as the
# hardware is set up, found by address. SANITIZER=address,undefined for the
# sanitizer run.
SRC := ../../src
DEP := ../../dependencies
LRC := $(DEP)/libretro-common
CXX ?= g++
FLAGS := -O1 -g -DENABLE_VULKAN -DENABLE_LIBRETRO -DRETRO_CORE -DVK_NO_PROTOTYPES \
	-I$(SRC) -I$(SRC)/Cafe -I$(SRC)/gui/interface -I$(SRC)/Common -I$(SRC)/util -I$(SRC)/input -I$(SRC)/config \
	-I$(LRC)/include -I$(DEP)/Vulkan-Headers/include -I$(DEP)/boost-compat -I$(DEP)/fmt/include \
	-include $(SRC)/Common/precompiled.h
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-sanitize=vptr -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -static
   EXE := .exe
endif
SRCS := mmio_test.cpp stubs.cpp $(SRC)/Cafe/HW/MMU/MMU.cpp

mmio_test$(EXE): $(SRCS) $(SRC)/Cafe/HW/MMU/MMU.h
	$(CXX) $(FLAGS) -std=gnu++20 $(SRCS) $(LIBS) -o $@

clean:
	rm -f mmio_test mmio_test.exe
