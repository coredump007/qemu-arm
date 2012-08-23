# -*- Mode: makefile -*-

GENERATED_HEADERS = $(TOP_DIR)/config/config-target.h $(TOP_DIR)/config/config-host.h

include $(TOP_DIR)/config/config-host.mak
include $(TOP_DIR)/config/config-devices.mak
include $(TOP_DIR)/config/config-target.mak
include $(TOP_DIR)/rules.mak

QEMU_CFLAGS += -I.. -I$(SRC_PATH)/include/target/arm -DNEED_CPU_H

include $(TOP_DIR)/Makefile.objs

QEMU_PROG=qemu-$(TARGET_ARCH2)
PROGS=$(QEMU_PROG)

config/config-host.h: config-host.h-timestamp
config/config-host.h-timestamp: config-host.mak

config/config-target.h: config/config-target.h-timestamp
config/config-target.h-timestamp: config/config-target.mak

all: $(PROGS)
# Dummy command so that make thinks it has done something
	@true

QEMU_CFLAGS += -I$(SRC_PATH)/include/tcg 
QEMU_CFLAGS += -I$(SRC_PATH)/include/linux-user
QEMU_CFLAGS += -I$(SRC_PATH)/include/linux-user/arm
QEMU_CFLAGS += -I$(SRC_PATH)/include/linux-user/arm/nwfpe
QEMU_CFLAGS += -I$(SRC_PATH)/include/fpu

#########################################################
# cpu emulator library
libobj-y = exec.o translate-all.o cpu-exec.o translate.o
libobj-y += tcg/tcg.o tcg/optimize.o
libobj-$(CONFIG_TCG_INTERPRETER) += tci.o
libobj-y += fpu/softfloat.o
libobj-y += op_helper.o
libobj-y += helper.o
libobj-y += cpu.o
libobj-y += neon_helper.o iwmmxt_helper.o
libobj-y += disas.o
libobj-y += gdbstub-xml.o
libobj-$(CONFIG_TCI_DIS) += tci-dis.o

tci-dis.o: QEMU_CFLAGS += -I$(SRC_PATH)/include/tcg/tci

$(libobj-y): $(GENERATED_HEADERS)

# HELPER_CFLAGS is used for all the legacy code compiled with static register
# variables
op_helper.o: QEMU_CFLAGS += $(HELPER_CFLAGS)
user-exec.o: QEMU_CFLAGS += $(HELPER_CFLAGS)

# Note: this is a workaround. The real fix is to avoid compiling
# cpu_signal_handler() in user-exec.c.
signal.o: QEMU_CFLAGS += $(HELPER_CFLAGS)

#########################################################
# Linux user emulator target

obj-y = main.o syscall.o strace.o mmap.o signal.o thunk.o \
      elfload.o linuxload.o uaccess.o gdbstub.o cpu-uname.o \
      user-exec.o

nwfpe-obj-y = fpa11.o fpa11_cpdo.o fpa11_cpdt.o fpa11_cprt.o fpopcode.o
nwfpe-obj-y += single_cpdo.o double_cpdo.o extended_cpdo.o
obj-arm-y +=  $(addprefix nwfpe/, $(nwfpe-obj-y))
obj-arm-y += arm-semi.o

$(obj-y) $(obj-$(TARGET_BASE_ARCH)-y): $(GENERATED_HEADERS)

obj-y += $(addprefix ../../util/qobject/, $(qobject-obj-y))
obj-y += $(addprefix ../../util/qapi/, $(qapi-obj-y))
obj-y += $(addprefix ../../util/qom/, $(qom-obj-y))
obj-y += $(addprefix ../libuser/, $(user-obj-y))
obj-y += $(addprefix ../libdis-user/, $(libdis-y))
obj-y += $(libobj-y)

$(QEMU_PROG): $(obj-y) $(obj-$(TARGET_BASE_ARCH)-y)
	$(call LINK,$^)

clean:
	rm -f *.o *.a *~ $(PROGS) nwfpe/*.o fpu/*.o
	rm -f *.d */*.d tcg/*.o ide/*.o 9pfs/*.o kvm/*.o
	rm -f hmp-commands.h qmp-commands-old.h

# Include automatically generated dependency files
-include $(wildcard *.d */*.d)
