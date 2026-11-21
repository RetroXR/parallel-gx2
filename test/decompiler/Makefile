# The shader decompiler on random R600 programs (decompiler_test.cpp,
# run_tests.sh): each decompiled to GLSL and compiled by rslc as the Vulkan
# renderer compiles it. SANITIZER=address,undefined for the sanitizer run.
SRC := ../../src
DEP := ../../dependencies
LRC := $(DEP)/libretro-common
CXX ?= g++
CC ?= gcc
FLAGS := -O1 -g -I$(SRC) -I$(SRC)/Cafe -I$(SRC)/Common -I$(SRC)/util -I$(LRC)/include -I$(DEP)/Vulkan-Headers/include \
	-DENABLE_VULKAN -DVK_NO_PROTOTYPES -DRSLC_EXTENDED -DRSLC_NO_TEXT -DHAVE_THREADS
# what the core's precompiled header wants
CXXFLAGS_CORE := -I$(DEP)/boost-compat -I$(DEP)/fmt/include \
	-include $(SRC)/Common/precompiled.h
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
# a shift out of range stops the run: a TEX instruction writing no register
# (SET_GRADIENTS_H/V, dstGpr -1) once shifted by -1 in the analyzer
ifneq (,$(findstring undefined,$(SANITIZER)))
   FLAGS += -fno-sanitize-recover=shift
endif
LIBS := -ldl -lpthread -lm
# a MinGW compiler, or MSYS2's own g++ on Windows
ifneq (,$(or $(findstring mingw,$(CXX)),$(filter Windows_NT,$(OS))))
   LIBS := -static
   EXE := .exe
endif
DEC := $(SRC)/Cafe/HW/Latte/LegacyShaderDecompiler
LRC_SRCS := $(LRC)/rthreads/retro_eventcount.c $(LRC)/rthreads/rthreads.c $(LRC)/gfx/shaders/rslc_front.c $(LRC)/gfx/shaders/rslc_ir.c $(LRC)/gfx/shaders/rslc_dxbc.c \
	$(LRC)/compat/compat_strl.c $(SRC)/util/helpers/obj_pool.c $(SRC)/util/helpers/lf_ring.c $(SRC)/util/spirv/spirv_writer.c \
	$(SRC)/util/helpers/thread_end.c
SRCS := decompiler_test.cpp stubs.cpp $(DEC)/LatteDecompiler.cpp $(DEC)/LatteDecompilerAnalyzer.cpp \
	$(DEC)/LatteDecompilerEmitGLSL.cpp $(DEC)/LatteDecompilerEmitGLSLAttrDecoder.cpp $(DEC)/LatteDecompilerEmitSPIRV.cpp \
	$(DEC)/LatteDecompilerRegisterDataTypeTracker.cpp $(SRC)/Cafe/HW/Latte/Core/LatteGSCopyShaderParser.cpp \
	$(SRC)/Cafe/HW/Latte/Renderer/Vulkan/VulkanAPI.cpp $(SRC)/Cafe/HW/Latte/Core/LattePM4Scan.cpp

decompiler_test$(EXE): $(SRCS) $(LRC_SRCS) $(wildcard $(DEC)/*.h $(DEC)/*.hpp) $(SRC)/util/helpers/StringBuf.h \
	$(SRC)/Cafe/HW/Latte/Renderer/Vulkan/RslcGlsl.h
	for f in $(LRC_SRCS); do $(CC) $(FLAGS) -std=gnu99 -c $$f -o $$(basename $$f .c).o || exit 1; done
	$(CXX) $(FLAGS) $(CXXFLAGS_CORE) -std=gnu++20 $(SRCS) *.o $(LIBS) -o $@
	rm -f *.o

clean:
	rm -rf decompiler_test decompiler_test.exe *.o modules
