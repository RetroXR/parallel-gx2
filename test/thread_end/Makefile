# What a thread keeps for itself, freed as it ends (src/util/helpers/
# thread_end.c, in C89 as the C89 lane builds it, and ThreadOwned.h), with
# no thread_local destructor: MinGW runs those on storage already freed
# (run_tests.sh). SANITIZER=thread or SANITIZER=address,undefined for the
# sanitizer runs; a MinGW compiler, or MSYS2's g++, for Windows.
LRC := ../../dependencies/libretro-common
SRC := ../../src
CC ?= gcc
CXX ?= g++
FLAGS := -O1 -g -I$(SRC) -I$(LRC)/include
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(or $(findstring mingw,$(CXX)),$(filter Windows_NT,$(OS))))
   LIBS += -static
   EXE := .exe
endif
include ../file_io.mk

thread_end_test$(EXE): thread_end_test.cpp $(SRC)/util/helpers/thread_end.c $(SRC)/util/helpers/thread_end.h $(SRC)/util/helpers/ThreadOwned.h
	$(CC) $(FLAGS) $(C89_FLAGS) -c $(SRC)/util/helpers/thread_end.c -o thread_end.o
	$(CXX) $(FLAGS) -std=gnu++20 thread_end_test.cpp thread_end.o $(LIBS) -o $@
	rm -f thread_end.o

clean:
	rm -f thread_end_test thread_end_test.exe *.o
