# An emulated controller's mappings read together (EmulatedController's
# get_mappings_down and get_axis_values, src/input/emulated/; run_tests.sh).
# SANITIZER=address,undefined for the sanitizer run.
SRC := ../../src
DEP := ../../dependencies
LRC := $(DEP)/libretro-common
CXX ?= g++
CC ?= gcc
FLAGS := -O1 -g -DENABLE_VULKAN -DENABLE_LIBRETRO -DRETRO_CORE -DVK_NO_PROTOTYPES \
	-I$(SRC) -I$(SRC)/Cafe -I$(SRC)/gui/interface -I$(SRC)/Common -I$(SRC)/util -I$(SRC)/input -I$(SRC)/config \
	-I$(LRC)/include -I$(DEP)/Vulkan-Headers/include -I$(DEP)/boost-compat -I$(DEP)/fmt/include
CXXFLAGS_CORE := -include $(SRC)/Common/precompiled.h
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-sanitize=vptr -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -static
   EXE := .exe
endif
SRCS := emulated_input_test.cpp $(SRC)/input/emulated/EmulatedController.cpp $(SRC)/input/api/Controller.cpp

C_SRCS := $(SRC)/util/helpers/rcu_cell.c $(LRC)/rthreads/retro_eventcount.c $(LRC)/rthreads/rthreads.c

emulated_input_test$(EXE): $(SRCS) $(C_SRCS) $(SRC)/input/emulated/EmulatedController.h
	for f in $(C_SRCS); do $(CC) $(FLAGS) -DHAVE_THREADS -std=gnu99 -c $$f -o $$(basename $$f .c).o || exit 1; done
	$(CXX) $(FLAGS) $(CXXFLAGS_CORE) -std=gnu++20 $(SRCS) *.o $(LIBS) -o $@
	rm -f *.o

clean:
	rm -f emulated_input_test emulated_input_test.exe *.o
