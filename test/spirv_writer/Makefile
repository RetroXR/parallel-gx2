# The SPIR-V writer the shader decompiler writes with
# (src/util/spirv/spirv_writer.c), in C89 as the C89 lane builds it
# (run_tests.sh). SANITIZER=address,undefined for the sanitizer run.
SRC := ../../src
CC ?= gcc
FLAGS := -O1 -g -I$(SRC)
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
ifneq (,$(findstring mingw,$(CC)))
   LIBS += -static
   EXE := .exe
endif
include ../file_io.mk

spirv_writer_test$(EXE): spirv_writer_test.c $(SRC)/util/spirv/spirv_writer.c $(SRC)/util/spirv/spirv_writer.h $(SRC)/util/spirv/spirv_ops.h
	$(CC) $(FLAGS) $(C89_FLAGS) spirv_writer_test.c $(SRC)/util/spirv/spirv_writer.c $(LIBS) -o $@

clean:
	rm -f spirv_writer_test spirv_writer_test.exe module.spv
