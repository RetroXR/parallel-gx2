# The PPC recompiler against the interpreter (run_tests.sh): PPC to IML, the
# passes and register allocator, and the x86-64 backend, as the core builds
# them. SANITIZER=address,undefined or SANITIZER=thread for the sanitizer
# runs.
LRC := ../../dependencies/libretro-common
SRC := ../../src
DEP := ../../dependencies
CC ?= gcc
CXX ?= g++
FLAGS := -O1 -g -I$(SRC) -I$(SRC)/Common -I$(SRC)/Cafe -I$(SRC)/util -I$(LRC)/include -DHAVE_THREADS
# what the core's precompiled header and the recompiler's headers want
CXXFLAGS_CORE := -std=gnu++20 -I$(DEP)/boost-compat -I$(DEP)/fmt/include \
	-I$(DEP)/Vulkan-Headers/include -include $(SRC)/Common/precompiled.h \
	-DENABLE_LIBRETRO -DRETRO_CORE -DVK_NO_PROTOTYPES -DENABLE_VULKAN
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
# undefined behaviour fails the run rather than only being reported
ifneq (,$(findstring undefined,$(SANITIZER)))
   FLAGS += -fno-sanitize-recover=undefined
endif
REC := $(SRC)/Cafe/HW/Espresso/Recompiler
ITP := $(SRC)/Cafe/HW/Espresso/Interpreter
ifneq (,$(findstring mingw,$(CXX)))
   LIBS += -static -lntdll
   EXE := .exe
   MEMMAPPER := $(SRC)/util/MemMapper/MemMapperWin.cpp
else
   LIBS += -lpthread
   MEMMAPPER := $(SRC)/util/MemMapper/MemMapperUnix.cpp
endif
CXX_SRCS := recompiler_test.cpp stubs.cpp \
	$(REC)/PPCRecompiler.cpp $(REC)/PPCRecompilerImlGen.cpp $(REC)/PPCRecompilerImlGenFPU.cpp \
	$(REC)/PPCRecompilerIntermediate.cpp $(REC)/PPCRecompilerWorkQueue.cpp \
	$(wildcard $(REC)/IML/*.cpp) $(wildcard $(REC)/BackendX64/*.cpp) \
	$(wildcard $(ITP)/*.cpp) $(SRC)/Common/cpu_features.cpp $(MEMMAPPER)
C_SRCS := $(REC)/rec_store.c $(SRC)/util/helpers/lf_ring.c \
	$(LRC)/rthreads/rthreads.c $(LRC)/rthreads/retro_eventcount.c
OBJS := $(addprefix obj/,$(notdir $(CXX_SRCS:.cpp=.o)) $(notdir $(C_SRCS:.c=.o)))
vpath %.cpp . $(REC) $(REC)/IML $(REC)/BackendX64 $(ITP) $(SRC)/Common $(SRC)/util/MemMapper
vpath %.c $(REC) $(SRC)/util/helpers $(LRC)/rthreads

recompiler_test$(EXE): $(OBJS)
	$(CXX) $(FLAGS) $(OBJS) $(LIBS) -o $@

obj/%.o: %.cpp | obj
	$(CXX) $(FLAGS) $(CXXFLAGS_CORE) -c $< -o $@

obj/%.o: %.c | obj
	$(CC) $(FLAGS) -std=gnu99 -c $< -o $@

obj:
	mkdir -p obj

clean:
	rm -rf obj recompiler_test recompiler_test.exe
