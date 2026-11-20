# The guest's sockets (src/Cafe/OS/libs/nsysnet/nsysnet.cpp): handles made
# and closed in place, a full table refused (run_tests.sh).
# SANITIZER=address,undefined for the sanitizer run.
SRC := ../../src
DEP := ../../dependencies
LRC := $(DEP)/libretro-common
CXX ?= g++
FLAGS := -O1 -g -I$(SRC) -I$(SRC)/Cafe -I$(LRC)/include -I$(DEP)/boost-compat -I$(DEP)/fmt/include -I$(DEP)/glm \
	-include $(SRC)/Common/precompiled.h
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-sanitize=vptr -fno-omit-frame-pointer
endif
# every malloc and free counted (the test's own and nsysnet's)
LIBS := -Wl,--wrap=malloc -Wl,--wrap=free -lpthread
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -Wl,--wrap=malloc -Wl,--wrap=free -static -lws2_32
   EXE := .exe
endif
SRCS := nsysnet_test.cpp stubs.cpp $(SRC)/Cafe/OS/libs/nsysnet/nsysnet.cpp

nsysnet_test$(EXE): $(SRCS)
	$(CXX) $(FLAGS) -std=gnu++20 $(SRCS) $(LIBS) -o $@

clean:
	rm -f nsysnet_test nsysnet_test.exe
