# NFC version 0 tags (src/Cafe/OS/libs/nfc/TagV0.cpp, run_tests.sh): read and
# written back the same. SANITIZER=address,undefined for the sanitizer run.
SRC := ../../src
DEP := ../../dependencies
LRC := $(DEP)/libretro-common
CXX ?= g++
FLAGS := -O1 -g -I$(SRC) -I$(SRC)/Cafe -I$(LRC)/include -I$(DEP)/boost-compat -I$(DEP)/fmt/include \
	-include $(SRC)/Common/precompiled.h
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-sanitize=vptr -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -static
   EXE := .exe
endif
NFC := $(SRC)/Cafe/OS/libs/nfc
SRCS := nfc_tag_test.cpp $(NFC)/TagV0.cpp $(NFC)/TLV.cpp $(NFC)/stream.cpp

nfc_tag_test$(EXE): $(SRCS) $(NFC)/TagV0.h
	$(CXX) $(FLAGS) -std=gnu++20 $(SRCS) $(LIBS) -o $@

clean:
	rm -f nfc_tag_test nfc_tag_test.exe
