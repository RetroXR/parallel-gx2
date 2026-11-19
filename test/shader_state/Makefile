# The shader state cache's fixed table and aux variants
# (src/Cafe/HW/Latte/Core/ShaderStateTable.h, run_tests.sh). SANITIZER=thread or
# SANITIZER=address,undefined for the sanitizer runs.
LRC := ../../dependencies/libretro-common
SRC := ../../src
CC ?= gcc
CXX ?= g++
FLAGS := -O1 -g -I$(SRC) -I$(LRC)/include
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
LIBS :=
ifneq (,$(findstring mingw,$(CXX)))
   LIBS += -static
   EXE := .exe
endif

shader_state_test$(EXE): shader_state_test.cpp $(SRC)/util/helpers/u64_map.c $(SRC)/Cafe/HW/Latte/Core/ShaderStateTable.h $(SRC)/util/helpers/u64_map.h
	$(CC) $(FLAGS) -std=c89 -c $(SRC)/util/helpers/u64_map.c -o u64_map.o
	$(CXX) $(FLAGS) -std=gnu++20 shader_state_test.cpp u64_map.o $(LIBS) -o $@
	rm -f u64_map.o

clean:
	rm -f shader_state_test shader_state_test.exe u64_map.o
