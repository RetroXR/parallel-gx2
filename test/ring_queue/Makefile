# The fixed first-in first-out queue (src/util/helpers/RingQueue.h,
# run_tests.sh). SANITIZER=address,undefined for the sanitizer run.
SRC := ../../src
CXX ?= g++
FLAGS := -O1 -g -I$(SRC)
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
LIBS :=
ifneq (,$(findstring mingw,$(CXX)))
   LIBS := -static
   EXE := .exe
endif

ring_queue_test$(EXE): ring_queue_test.cpp $(SRC)/util/helpers/RingQueue.h
	$(CXX) $(FLAGS) -std=gnu++20 ring_queue_test.cpp $(LIBS) -o $@

clean:
	rm -f ring_queue_test ring_queue_test.exe
