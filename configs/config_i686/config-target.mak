# Automatically generated by configure - do not modify
TARGET_SHORT_ALIGNMENT=2
TARGET_INT_ALIGNMENT=4
TARGET_LONG_ALIGNMENT=4
TARGET_LLONG_ALIGNMENT=4
TARGET_ARCH=arm
TARGET_ARM=y
TARGET_ARCH2=arm
TARGET_BASE_ARCH=arm
TARGET_ABI_DIR=arm
CONFIG_NO_XEN=y
CONFIG_USER_ONLY=y
CONFIG_QEMU_INTERP_PREFIX="/usr/gnemul/qemu-arm"
CONFIG_LINUX_USER=y
TARGET_XML_FILES= /home/dell/qemu/qemu/gdb-xml/arm-core.xml /home/dell/qemu/qemu/gdb-xml/arm-vfp.xml /home/dell/qemu/qemu/gdb-xml/arm-vfp3.xml /home/dell/qemu/qemu/gdb-xml/arm-neon.xml
TARGET_HAS_BFLT=y
CONFIG_USE_NPTL=y
CONFIG_USE_GUEST_BASE=y
CONFIG_I386_DIS=y
CONFIG_ARM_DIS=y
LDFLAGS+=-Wl,-T$(SRC_PATH)/config/config-host.ld -Wl,-T,$(SRC_PATH)/config/$(ARCH).ld
QEMU_CFLAGS+=
QEMU_INCLUDES+=-I$(SRC_PATH)/include/tcg -I$(SRC_PATH)/include/tcg/i386