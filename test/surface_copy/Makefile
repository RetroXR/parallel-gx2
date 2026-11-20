# The surface copies' framebuffers and descriptor sets by pipeline, texture,
# slice and mip (src/Cafe/HW/Latte/Renderer/Vulkan/SurfaceCopyTable.h,
# run_tests.sh). SANITIZER=address,undefined or SANITIZER=thread for the
# sanitizer runs.
LRC := ../../dependencies/libretro-common
SRC := ../../src
CC ?= gcc
CXX ?= g++
FLAGS := -O1 -g -I$(SRC) -I$(LRC)/include
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -static
   EXE := .exe
endif
C_SRCS := $(SRC)/util/helpers/obj_pool.c $(SRC)/util/helpers/lf_ring.c $(SRC)/util/helpers/u64_map.c

surface_copy_test$(EXE): surface_copy_test.cpp $(C_SRCS) $(SRC)/Cafe/HW/Latte/Renderer/Vulkan/SurfaceCopyTable.h $(SRC)/util/helpers/U64Table.h $(SRC)/util/helpers/PoolNew.h
	for f in $(C_SRCS); do $(CC) $(FLAGS) -std=gnu99 -c $$f -o $$(basename $$f .c).o || exit 1; done
	$(CXX) $(FLAGS) -std=gnu++20 surface_copy_test.cpp *.o $(LIBS) -o $@
	rm -f *.o

clean:
	rm -f surface_copy_test surface_copy_test.exe *.o
