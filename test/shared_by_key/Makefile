# Objects shared by key, made once by whichever thread asks first
# (src/util/helpers/SharedByKey.h, run_tests.sh). SANITIZER=thread or
# SANITIZER=address,undefined for the sanitizer runs.
SRC := ../../src
CXX ?= g++
FLAGS := -O1 -g -I$(SRC)
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -static
   EXE := .exe
endif

shared_by_key_test$(EXE): shared_by_key_test.cpp $(SRC)/util/helpers/SharedByKey.h
	$(CXX) $(FLAGS) -std=gnu++20 shared_by_key_test.cpp $(LIBS) -o $@

clean:
	rm -f shared_by_key_test shared_by_key_test.exe
