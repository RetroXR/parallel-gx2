# Pipelines made ahead of the draw or copy that needs them
# (src/Cafe/HW/Latte/Renderer/Vulkan/PreparedPipeline.h,
# prepared_pipeline_test.cpp). SANITIZER=thread or SANITIZER=address,undefined
# for the sanitizer runs.
SRC := ../../src
DEP := ../../dependencies
LRC := $(DEP)/libretro-common
CXX ?= g++
CC ?= gcc
FLAGS := -O1 -g -I$(SRC) -I$(LRC)/include -I$(DEP)/Vulkan-Headers/include -DVK_NO_PROTOTYPES
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
LIBS := -lpthread
include ../file_io.mk
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -static
   EXE := .exe
endif
LRC_SRCS := $(LRC)/rthreads/retro_eventcount.c $(LRC)/rthreads/rthreads.c
# the pool the tickets come from (PoolNew.h)
C_SRCS := $(SRC)/util/helpers/obj_pool.c $(SRC)/util/helpers/lf_ring.c

prepared_pipeline_test$(EXE): prepared_pipeline_test.cpp $(LRC_SRCS) $(C_SRCS) $(SRC)/Cafe/HW/Latte/Renderer/Vulkan/PreparedPipeline.h \
	$(SRC)/util/helpers/ClaimTicket.h $(SRC)/util/helpers/MpmcQueue.h $(SRC)/util/helpers/RecycledBuffers.h $(SRC)/util/helpers/EventWake.h \
	$(SRC)/util/helpers/PoolNew.h $(SRC)/util/helpers/obj_pool.h
	for f in $(LRC_SRCS); do $(CC) $(FLAGS) -std=gnu99 -c $$f -o $$(basename $$f .c).o || exit 1; done
	for f in $(C_SRCS); do $(CC) $(FLAGS) $(C89_FLAGS) -c $$f -o $$(basename $$f .c).o || exit 1; done
	$(CXX) $(FLAGS) -std=gnu++20 prepared_pipeline_test.cpp retro_eventcount.o rthreads.o obj_pool.o lf_ring.o $(LIBS) -o $@
	rm -f retro_eventcount.o rthreads.o obj_pool.o lf_ring.o

clean:
	rm -f prepared_pipeline_test prepared_pipeline_test.exe *.o
