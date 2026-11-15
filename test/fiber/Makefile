# The fibers' test (run_tests.sh): FIBER=unix (ucontext), FIBER=libco
# (Android's backend, on libretro-common's libco) or, with a MinGW compiler
# or FIBER=win, Windows' own fibers. SANITIZER=address,undefined for the sanitizer run;
# ThreadSanitizer does not follow stack switches, so not that.
LRC := ../../dependencies/libretro-common
CXX ?= g++
CC ?= gcc
FIBER ?= unix
FLAGS := -O1 -g -I../../src -I$(LRC)/include -include prelude.h
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer -fno-sanitize-recover=undefined
endif
LIBS := -lpthread
SRCS := fiber_test.cpp
OBJS :=
ifneq (,$(or $(findstring mingw,$(CXX)),$(filter win,$(FIBER))))
   SRCS += ../../src/util/Fiber/FiberWin.cpp
   LIBS += -static
   EXE := .exe
else ifeq ($(FIBER),libco)
   SRCS += ../../src/util/Fiber/FiberAndroid.cpp
   OBJS += libco.o
else
   SRCS += ../../src/util/Fiber/FiberUnix.cpp
endif

fiber_test$(EXE): $(SRCS) prelude.h ../../src/util/Fiber/Fiber.h
	$(if $(OBJS),$(CC) -O1 -g -I$(LRC)/include -DLIBCO_MP -std=gnu11 -c $(LRC)/libco/libco.c -o libco.o,true)
	$(CXX) $(FLAGS) -std=gnu++20 $(SRCS) $(OBJS) $(LIBS) -o $@
	rm -f *.o

clean:
	rm -f fiber_test fiber_test.exe *.o
