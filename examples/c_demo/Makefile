CC ?= clang
CFLAGS ?= -O2 -Wall -I../../crates/phone-buddy-ffi/include

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS_STATIC ?= ../../target/release/libphone_buddy_ffi.a -lpthread -ldl -lm -framework Security -framework CoreFoundation
else
    LDFLAGS_STATIC ?= ../../target/release/libphone_buddy_ffi.a -lpthread -ldl -lm
endif

all: demo

demo: main.c ../../target/release/libphone_buddy_ffi.a
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS_STATIC)

../../target/release/libphone_buddy_ffi.a:
	cd ../.. && cargo build --release -p phone-buddy-ffi

run: demo
	./demo

clean:
	rm -f demo demo_dylib
