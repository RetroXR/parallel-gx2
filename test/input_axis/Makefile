# The stick dead zone and range (ControllerBase::apply_axis_setting,
# src/input/api/Controller.cpp; run_tests.sh). SANITIZER=address,undefined
# for the sanitizer run.
SRC := ../../src
DEP := ../../dependencies
LRC := $(DEP)/libretro-common
CXX ?= g++
FLAGS := -O2 -g -DENABLE_VULKAN -DENABLE_LIBRETRO -DRETRO_CORE -DVK_NO_PROTOTYPES \
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
SRCS := input_axis_test.cpp $(SRC)/input/api/Controller.cpp

input_axis_test$(EXE): $(SRCS) $(SRC)/input/api/Controller.h $(SRC)/input/api/ControllerState.h
	$(CXX) $(FLAGS) -std=gnu++20 $(SRCS) $(LIBS) -o $@

clean:
	rm -f input_axis_test input_axis_test.exe
