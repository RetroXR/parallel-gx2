# The toy bases taking figures from a frontend (src/Cafe/OS/libs/nsyshid:
# Skylander, Infinity, Dimensions, FigureMail.h, hid_portal's host commands),
# linked against the core's own objects for them: build the core first.
# OBJ is the core's object directory (obj/win-x86_64, obj/unix-x86_64...).
CXX ?= g++
# for this machine: obj/win-x86_64 under MinGW, else the unix one
ifneq (,$(findstring mingw,$(shell $(CXX) -dumpmachine)))
   OBJ ?= ../../obj/win-x86_64
else
   OBJ ?= $(firstword $(wildcard ../../obj/unix-*))
endif
# the core's own defines and include paths (Makefile.common), as its objects were built
FLAGS := -O1 -g -std=gnu++20 -DENABLE_LIBRETRO -DRETRO_CORE -DVK_NO_PROTOTYPES -DENABLE_VULKAN -DENABLE_OPENGL \
	-I../../src -I../../src/Cafe -I../../src/Common -I../../src/util -I../../src/Cemu -I../../src/input \
	-I../../src/config -I../../src/audio -I../../src/libretro -I../../dependencies/Vulkan-Headers/include \
	-I../../dependencies/fmt/include -I../../dependencies/boost-compat \
	-I../../dependencies/libretro-common/include -include $(OBJ)/pch/precompiled.h \
	-Wno-multichar -Wno-invalid-offsetof
NSYSHID := $(OBJ)/src/Cafe/OS/libs/nsyshid
LRC := $(OBJ)/dependencies/libretro-common
OBJS := $(NSYSHID)/Skylander.cpp.o $(NSYSHID)/Infinity.cpp.o $(NSYSHID)/Dimensions.cpp.o \
	$(NSYSHID)/hid_portal.c.o $(OBJ)/src/util/helpers/lf_ring.c.o $(OBJ)/src/util/helpers/mailbox.c.o \
	$(LRC)/rthreads/retro_eventcount.c.o $(LRC)/rthreads/rthreads.c.o
LIBS := -lpthread
ifneq (,$(findstring mingw,$(shell $(CXX) -dumpmachine)))
   FLAGS += -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -DWIN32_LEAN_AND_MEAN
   LIBS += -static
   EXE := .exe
endif

toy_bases_test$(EXE): toy_bases_test.cpp stubs.cpp $(OBJS)
	$(CXX) $(FLAGS) toy_bases_test.cpp stubs.cpp $(OBJS) $(LIBS) -o $@

clean:
	rm -f toy_bases_test toy_bases_test.exe
