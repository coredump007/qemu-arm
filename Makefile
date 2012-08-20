ARCH := $(shell arch)
TOP_DIR := $(shell pwd)
BUILD_STATIC := false

export TOP_DIR BUILD_STATIC

.PHONY: all build config clean

all: build

config: configs
	ln -sf configs/config_$(ARCH) config

build: config
	make -C core/arm-linux-user

clean:
	find -name "*.d" | xargs rm -rf
	find -name "*.o" | xargs rm -rf
	rm -rf core/arm-linux-user/qemu-arm
	rm -rf config/*.h
	rm -rf config
