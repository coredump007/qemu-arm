# Automatically generated by configure - do not modify
# Configured with: './configure' '--prefix=/q' '--target-list=arm-linux-user' '--disable-vnc' '--disable-sdl' '--disable-curses' '--disable-guest-agent' '--disable-system' '--disable-opengl' '--disable-docs' '--disable-vhost-net'
all:
prefix=/q
bindir=${prefix}/bin
libdir=${prefix}/lib
includedir=${prefix}/include
mandir=${prefix}/share/man
sysconfdir=${prefix}/etc
qemu_confdir=${prefix}/etc/qemu
qemu_datadir=${prefix}/share/qemu
qemu_docdir=${prefix}/share/doc/qemu
libexecdir=${prefix}/libexec
CONFIG_QEMU_HELPERDIR="/q/libexec"
ARCH=x86_64
STRIP=strip
CONFIG_POSIX=y
CONFIG_LINUX=y
CONFIG_SLIRP=y
CONFIG_SMBD_COMMAND="/usr/sbin/smbd"
CONFIG_AC97=y
CONFIG_ES1370=y
CONFIG_SB16=y
CONFIG_HDA=y
CONFIG_AUDIO_DRIVERS=oss
CONFIG_OSS=y
CONFIG_BDRV_WHITELIST=
CONFIG_FNMATCH=y
VERSION=1.1.1
PKGVERSION=
SRC_PATH=$(TOP_DIR)
TARGET_DIRS=arm-linux-user
CONFIG_ATFILE=y
CONFIG_UTIMENSAT=y
CONFIG_PIPE2=y
CONFIG_ACCEPT4=y
CONFIG_SPLICE=y
CONFIG_EVENTFD=y
CONFIG_FALLOCATE=y
CONFIG_SYNC_FILE_RANGE=y
CONFIG_FIEMAP=y
CONFIG_DUP3=y
CONFIG_EPOLL=y
CONFIG_EPOLL_CREATE1=y
CONFIG_EPOLL_PWAIT=y
CONFIG_INOTIFY=y
CONFIG_INOTIFY1=y
CONFIG_BYTESWAP_H=y
GLIB_CFLAGS=-pthread -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include  
CONFIG_ATTR=y
CONFIG_IOVEC=y
CONFIG_PREADV=y
CONFIG_SIGNALFD=y
CONFIG_FDATASYNC=y
CONFIG_MADVISE=y
CONFIG_POSIX_MADVISE=y
CONFIG_SMARTCARD=y
CONFIG_UNAME_RELEASE=""
CONFIG_ZERO_MALLOC=y
CONFIG_UCONTEXT_COROUTINE=y
CONFIG_LINUX_MAGIC_H=y
CONFIG_HAS_ENVIRON=y
HOST_USB=linux
TRACE_BACKEND=nop
CONFIG_TRACE_NOP=y
CONFIG_TRACE_FILE=trace
CONFIG_TRACE_DEFAULT=y
TOOLS=
ROMS=
MAKE=make
INSTALL=install
INSTALL_DIR=install -d -m 0755
INSTALL_DATA=install -c -m 0644
INSTALL_PROG=install -c -m 0755
PYTHON=python
CC=gcc
CC_I386=$(CC) -m32
HOST_CC=gcc
AR=ar
OBJCOPY=objcopy
LD=ld
WINDRES=windres
LIBTOOL=
CFLAGS=-O2 -g 
QEMU_CFLAGS=-fPIE -DPIE -m64 -D_FORTIFY_SOURCE=2 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -Wstrict-prototypes -Wredundant-decls -Wall -Wundef -Wwrite-strings -Wmissing-prototypes -fno-strict-aliasing  -fstack-protector-all -Wendif-labels -Wmissing-include-dirs -Wempty-body -Wnested-externs -Wformat-security -Wformat-y2k -Winit-self -Wignored-qualifiers -Wold-style-declaration -Wold-style-definition -Wtype-limits
QEMU_INCLUDES=-I. -I$(SRC_PATH)/config -I$(SRC_PATH)/include/common -I$(SRC_PATH)/include
HELPER_CFLAGS=

ifeq ($(BUILD_STATIC),true)
LDFLAGS=-Wl,--warn-common --static -m64 -g 
else
LDFLAGS=-Wl,--warn-common -Wl,-z,relro -Wl,-z,now -pie -m64 -g 
endif

ARLIBS_BEGIN=
ARLIBS_END=
LIBS+=-lrt -pthread -lgthread-2.0 -lrt -lglib-2.0   
LIBS_TOOLS+=
EXESUF=
LIBS_QGA+=-pthread -lgthread-2.0 -lrt -lglib-2.0   
POD2MAN=pod2man --utf8
