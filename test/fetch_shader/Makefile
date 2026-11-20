# Fetch shaders parsed (src/Cafe/HW/Latte/Core/FetchShader.cpp): their
# buffers' groups and attributes in place, and nothing allocated (run_tests.sh).
# SANITIZER=address,undefined for the sanitizer run.
SRC := ../../src
DEP := ../../dependencies
LRC := $(DEP)/libretro-common
CXX ?= g++
CC ?= gcc
FLAGS := -O1 -g -I$(SRC) -I$(SRC)/Cafe -I$(LRC)/include -I$(DEP)/Vulkan-Headers/include -DVK_NO_PROTOTYPES -DENABLE_VULKAN -DHAVE_THREADS
CXXFLAGS_CORE := -I$(DEP)/boost-compat -I$(DEP)/fmt/include -I$(DEP)/glm \
	-include $(SRC)/Common/precompiled.h
ifneq ($(SANITIZER),)
   # vptr checks want the renderer's typeinfo, which is not linked here
   FLAGS += -fsanitize=$(SANITIZER) -fno-sanitize=vptr -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -static
   EXE := .exe
endif
SRCS := fetch_shader_test.cpp stubs.cpp $(SRC)/Cafe/HW/Latte/Core/FetchShader.cpp
C_SRCS := $(SRC)/util/helpers/obj_pool.c $(SRC)/util/helpers/lf_ring.c $(SRC)/util/helpers/u64_map.c $(LRC)/hash/lrc_hash.c

fetch_shader_test$(EXE): $(SRCS) $(C_SRCS) $(SRC)/Cafe/HW/Latte/Core/FetchShader.h
	for f in $(C_SRCS); do $(CC) $(FLAGS) -std=gnu99 -c $$f -o $$(basename $$f .c).o || exit 1; done
	$(CXX) $(FLAGS) $(CXXFLAGS_CORE) -std=gnu++20 $(SRCS) *.o $(LIBS) -o $@
	rm -f *.o

clean:
	rm -f fetch_shader_test fetch_shader_test.exe *.o
