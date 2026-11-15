# coreinit's virtual address range, FSA clients, alarm tables and host thread table (src/Cafe/OS/libs/coreinit/virt_range.c, fsa_clients.c, alarm_table.c, host_thread_table.c), in C89 as the C89
# lane builds it (run_tests.sh). SANITIZER=thread or
# SANITIZER=address,undefined for the sanitizer runs.
LRC := ../../dependencies/libretro-common
SRC := ../../src
CC ?= gcc
FLAGS := -O1 -g -I$(SRC) -I$(LRC)/include -DHAVE_THREADS -DVIRT_RANGE_TEST_HOOKS
ifneq ($(SANITIZER),)
   FLAGS += -fsanitize=$(SANITIZER) -fno-omit-frame-pointer
endif
LIBS := -lpthread
ifneq (,$(findstring mingw,$(CC)))
   LIBS += -static
   EXE := .exe
endif
include ../file_io.mk
LRC_SRCS := $(LRC)/rthreads/rthreads.c
CORE_SRCS := os_tables_test.c $(SRC)/Cafe/OS/libs/coreinit/virt_range.c $(SRC)/Cafe/OS/libs/coreinit/fsa_clients.c $(SRC)/Cafe/OS/libs/coreinit/alarm_table.c $(SRC)/Cafe/OS/libs/coreinit/host_thread_table.c

os_tables_test$(EXE): $(CORE_SRCS) $(LRC_SRCS) $(SRC)/Cafe/OS/libs/coreinit/virt_range.h $(SRC)/Cafe/OS/libs/coreinit/fsa_clients.h $(SRC)/Cafe/OS/libs/coreinit/alarm_table.h $(SRC)/Cafe/OS/libs/coreinit/host_thread_table.h
	mkdir -p obj
	for f in $(LRC_SRCS); do $(CC) $(FLAGS) -std=gnu99 -c $$f -o obj/$$(basename $$f .c).o || exit 1; done
	for f in $(CORE_SRCS); do $(CC) $(FLAGS) $(C89_FLAGS) -c $$f -o obj/$$(basename $$f .c).o || exit 1; done
	$(CC) $(FLAGS) obj/*.o $(LIBS) -o $@
	rm -rf obj

clean:
	rm -rf os_tables_test os_tables_test.exe obj
